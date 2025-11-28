#include <cv_bridge/cv_bridge.h>
#include <cmath>
#include <geometry_msgs/msg/point.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <algorithm>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>
#include <referee_pkg/msg/object.hpp>
#include <referee_pkg/msg/multi_object.hpp>
#include <referee_pkg/msg/race_stage.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/header.hpp>
#include "sensor_msgs/msg/image.hpp"

using namespace std;
using namespace rclcpp;
using namespace cv;
const double EPS_RATIO = 0.04; // approxPolyDP eps = EPS_RATIO * perimeter
const int DIGIT_SIZE = 300;    // 数字比对时的统一大小（像素）
// 归一化 SAD 阈值：0 表示完全相同，1 表示完全相反。模板仅1个时设宽松一些；多模板可选择最小值
const double SAD_THRESHOLD = 0.40; // 根据实际效果调整，减小更严格

// ----------------- 工具函数 -----------------

// 在多个点中找到最左下（x 最小，若 x 相同则 y 最大）点的索引
// 用于把四个点旋转，使左下为起点（便于统一输出顺序）
static int leftBottomIdx(const vector<Point2f> &pts)
{
  int idx = 0;
  for (size_t i = 1; i < pts.size(); ++i)
  {
    if (pts[i].x < pts[idx].x - 1e-6)
      idx = i;
    else if (fabs(pts[i].x - pts[idx].x) < 1e-6 && pts[i].y > pts[idx].y)
      idx = i;
  }
  return idx;
}
// 计算两个矩形的 IoU（交并比）
float rectIoU(const Rect2f &a, const Rect2f &b)
{
  Rect2f inter = a & b; // 计算交集矩形
  float interArea = inter.area();
  float unionArea = a.area() + b.area() - interArea;
  return interArea / unionArea;
}

struct
{
  string type;
  vector<Point> armorpoints;
  bool exist = false;
} armor[6];

class VisionNode : public rclcpp::Node
{
public:
  VisionNode(string name) : Node(name)
  {
    RCLCPP_INFO(this->get_logger(), "Initializing VisionNode");
    // 加载装甲板template
    const string TPL_DIR = "src/target_model_pkg/urdf/armor/textures";
    for (int i = 1; i <= 5; ++i)
    {
      string p = TPL_DIR + "/" + "small_num" + to_string(i) + ".png";
      Mat t = imread(p, IMREAD_GRAYSCALE);
      if (t.empty())
      {
        RCLCPP_ERROR(this->get_logger(), "Failed to load template image: %s", p.c_str());
        continue;
      }
      threshold(t, t, 128, 255, THRESH_BINARY);
      morphologyEx(t, t, MORPH_CLOSE, rect_k);
      morphologyEx(t, t, MORPH_OPEN, rect_k);
      RCLCPP_INFO(this->get_logger(), "success Loaded template: %s", p.c_str());
      // 从模板中提取最大的白色连通域（应对应数字区域）
      Rect tplBox;
      Mat a = extractLargestWhiteComponent(t, tplBox);
      if (a.empty())
      {
        RCLCPP_WARN(this->get_logger(), "Failed to find number in template");
      }
      resize(a, a, Size(DIGIT_SIZE, DIGIT_SIZE));
      tpl[i] = a;
    }

    cv::namedWindow("Detection Result", cv::WINDOW_AUTOSIZE);
    Image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", 10,
        bind(&VisionNode::callback_camera, this, std::placeholders::_1));

    Target_pub = this->create_publisher<referee_pkg::msg::MultiObject>(
        "/vision/target", 10);

    RCLCPP_INFO(this->get_logger(), "VisionNode initialized successfully");
  }

  ~VisionNode() { cv::destroyWindow("Detection Result"); }

private:
  void callback_camera(sensor_msgs::msg::Image::SharedPtr msg);
  Mat tpl[6]; // 模板图像数组，索引1-5有效
  Mat rect_k = getStructuringElement(MORPH_RECT, Size(3, 3));
  // 稳定的球体点计算方法
  vector<Point2f> calculateStableSpherePoints(const Point2f &center,
                                              float radius);
  vector<vector<Point2f>> findQuadrilateralsFromBinary(const Mat &bin);
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr Image_sub;
  rclcpp::Publisher<referee_pkg::msg::MultiObject>::SharedPtr Target_pub;
  // 归一化绝对差（SAD）用于像素比对
  double normalizedSAD(const Mat &a, const Mat &b);
  // 从二值图像（白色为前景）中提取面积最大的白色连通区域并返回其裁剪图像。
  Mat extractLargestWhiteComponent(const Mat &bw, Rect &outBox);
  // 前一针图像中的rect位置，用于简单跟踪
  Rect2f rect = {0, 0, 0, 0};
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VisionNode>("VisionNode");
  RCLCPP_INFO(node->get_logger(), "Starting VisionNode");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

vector<Point2f> VisionNode::calculateStableSpherePoints(const Point2f &center,
                                                        float radius)
{
  vector<Point2f> points;

  // 简单稳定的几何计算，避免漂移
  // 左、下、右、上
  points.push_back(Point2f(center.x - radius, center.y)); // 左点 (1)
  points.push_back(Point2f(center.x, center.y + radius)); // 下点 (2)
  points.push_back(Point2f(center.x + radius, center.y)); // 右点 (3)
  points.push_back(Point2f(center.x, center.y - radius)); // 上点 (4)

  return points;
}

void draw4points(vector<Point2f> p, Mat result_image)
{
  vector<cv::Scalar> point_colors = {
      cv::Scalar(255, 0, 0),   // 蓝色 - 左
      cv::Scalar(0, 255, 0),   // 绿色 - 下
      cv::Scalar(0, 255, 255), // 黄色 - 右
      cv::Scalar(255, 0, 255)  // 紫色 - 上
  };

  for (int j = 0; j < 4; j++)
  {
    cv::circle(result_image, p[j], 6, point_colors[j], -1);
    cv::circle(result_image, p[j], 6, cv::Scalar(0, 0, 0), 2);

    // 标注序号
    string point_text = to_string(j + 1);
    cv::putText(
        result_image, point_text,
        cv::Point(p[j].x + 10, p[j].y - 10),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 3);
    cv::putText(
        result_image, point_text,
        cv::Point(p[j].x + 10, p[j].y - 10),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, point_colors[j], 2);
  }
}
// 从阈值二值图中找到四边形轮廓并返回四个点（按左下起逆时针）
vector<vector<Point2f>> VisionNode::findQuadrilateralsFromBinary(const Mat &bin)
{
  vector<vector<Point2f>> quads;
  vector<vector<Point>> contours;
  findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
  for (size_t i = 0; i < contours.size(); ++i)
  {
    double a = fabs(contourArea(contours[i]));
    if (a < 500)
      continue;
    // 多边形逼近
    double peri = arcLength(contours[i], true);
    vector<Point> approx;
    approxPolyDP(contours[i], approx, EPS_RATIO * peri, true);
    // 需要 4 个顶点并近似矩形
    if (approx.size() == 4)
    {
      if (!isContourConvex(approx))
        continue;
      std::vector<cv::Point2f> approx2f;
      approx2f.reserve(approx.size());
      for (const auto &p : approx)
      {
        approx2f.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
      }
      int lb = leftBottomIdx(approx2f);
      vector<Point2f> quad;
      for (int j = 0; j < 4; j++)
        quad.push_back(Point2f(static_cast<float>(approx[(lb + j) % 4].x), static_cast<float>(approx[(lb + j) % 4].y)));
      quads.push_back(quad);
      RCLCPP_INFO(this->get_logger(), "find possible armor");
    }
  }
  return quads;
}
void VisionNode::callback_camera(sensor_msgs::msg::Image::SharedPtr msg)
{
  try
  {
    // 图像转换
    cv_bridge::CvImagePtr cv_ptr;

    if (msg->encoding == "rgb8" || msg->encoding == "R8G8B8")
    {
      cv::Mat image(msg->height, msg->width, CV_8UC3,
                    const_cast<unsigned char *>(msg->data.data()));
      cv::Mat bgr_image;
      cv::cvtColor(image, bgr_image, cv::COLOR_RGB2BGR);
      cv_ptr = std::make_shared<cv_bridge::CvImage>(); // 分配内存
      cv_ptr->header = msg->header;                    // 时间戳
      cv_ptr->encoding = "bgr8";
      cv_ptr->image = bgr_image;
    }
    else
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }

    cv::Mat image = cv_ptr->image;

    if (image.empty())
    {
      RCLCPP_WARN(this->get_logger(), "Received empty image");
      return;
    }
    // 创建结果图像
    cv::Mat result_image = image.clone();

    // 初始化返回消息
    referee_pkg::msg::MultiObject msg_object;
    msg_object.header = msg->header;
    msg_object.num_objects = 0;

    // 重置armor数组
    for (int i = 1; i < 6; i++)
    {
      armor[i].exist = false;
      armor[i].type = "";
      armor[i].armorpoints.clear();
    }

    cv::Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    Mat blackMask;
    Scalar lowerBlack(0, 0, 0);
    Scalar upperBlack(180, 255, 60);
    inRange(hsv, lowerBlack, upperBlack, blackMask);
    Mat greenMask;

    Scalar lowerGreen(40, 80, 80);
    Scalar upperGreen(105, 255, 255);
    inRange(hsv, lowerGreen, upperGreen, greenMask);

    // find black rect
    Mat rect_k = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(blackMask, blackMask, MORPH_CLOSE, rect_k);
    morphologyEx(blackMask, blackMask, MORPH_OPEN, rect_k);

    auto blackQuads = findQuadrilateralsFromBinary(blackMask);
    // find armor
    for (size_t i = 0; i < blackQuads.size(); i++)
    {
      auto contour = blackQuads[i];
      Mat inside(blackMask, boundingRect(contour));
      Rect digitBox;
      bitwise_not(inside, inside);
      Mat roi = extractLargestWhiteComponent(inside, digitBox);
      if (roi.empty())
        continue;
      // 将候选数字缩放到与模板相同的大小进行像素比对
      resize(roi, roi, Size(DIGIT_SIZE, DIGIT_SIZE));

      double best = 100;
      int flag = 0, j = 1;
      for (j = 1; j < 6; j++)
      {
        // 计算候选与模板的归一化 SAD（像素差平均值）
        double sad = normalizedSAD(roi, tpl[j]);
        // maybe颜色反转,尝试反色比对
        /*Mat rroi;
        bitwise_not(roi, rroi);
        double sadInv = normalizedSAD(rroi, tpl[j]);
        if (best < min(sad, sadInv))
        {
          best = min(sad, sadInv);
          flag = j;
        }*/
        if (sad < best)
        {
          best = sad;
          flag = j;
        }
      }
      RCLCPP_INFO(this->get_logger(), "bestSAD=%f for digit %d", best, flag);
      // 根据阈值判断是否匹配。
      if (best > SAD_THRESHOLD)
        continue;

      armor[flag].type = "armor_red_" + std::to_string(flag);
      armor[flag].exist = true;

      // 输出与绘制
      // 计算中心
      Point2f center(0.f, 0.f);
      for (auto &p : blackQuads[i])
        center += p;
      center *= (1.0f / 4.0f);
      // 计算原矩形在 x 方向的水平宽度（投影宽度）
      float xmin = blackQuads[i][0].x, xmax = blackQuads[i][0].x;
      for (int j = 1; j < 4; ++j)
      {
        xmin = min(xmin, blackQuads[i][j].x);
        xmax = max(xmax, blackQuads[i][j].x);
      }
      double targetW = double(xmax - xmin);

      // 根据宽高比计算目标高度
      double targetH = targetW * (0.230 / 0.705);

      // 构造轴对齐矩形的四个角（以 center 为中心），坐标系 y 向下为正
      // 左下, 左上, 右上, 右下（局部）
      vector<Point> pts(4);
      pts[0] = Point((int)(center.x - targetW / 2.0), (int)(center.y + targetH / 2.0)); // 左下
      pts[1] = Point((int)(center.x - targetW / 2.0), (int)(center.y - targetH / 2.0)); // 左上
      pts[2] = Point((int)(center.x + targetW / 2.0), (int)(center.y - targetH / 2.0)); // 右上
      pts[3] = Point((int)(center.x + targetW / 2.0), (int)(center.y + targetH / 2.0)); // 右下
      armor[flag].armorpoints = pts;

      for (int j = 0; j < 4; j++)
      {
        cv::circle(result_image, pts[j], 6, cv::Scalar(0, 0, 255), 2);
        // 标注序号
        string point_text = to_string(j + 1);
        cv::putText(
            result_image, point_text,
            cv::Point(pts[j].x + 10, pts[j].y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
      }
    }
    // 填写装甲板消息
    for (int i = 1; i < 6; i++)
    {
      if (armor[i].exist)
      {
        msg_object.num_objects++;
        referee_pkg::msg::Object obj;
        obj.target_type = armor[i].type;
        for (int j = 0; j < 4; j++)
        {
          geometry_msgs::msg::Point corner;
          corner.x = (int)(armor[i].armorpoints[j].x);
          corner.y = (int)(armor[i].armorpoints[j].y);
          corner.z = 0.0;
          obj.corners.push_back(corner);
        }
        msg_object.objects.push_back(obj);
        RCLCPP_INFO(this->get_logger(), "Found armor: %s", armor[i].type.c_str());
      }
    }

    // find green rect
    morphologyEx(greenMask, greenMask, MORPH_CLOSE, rect_k);
    morphologyEx(greenMask, greenMask, MORPH_OPEN, rect_k);
    auto greenQuads = findQuadrilateralsFromBinary(greenMask);
    if (greenQuads.size() > 0)
    {
      // output msg
      msg_object.num_objects++;
      referee_pkg::msg::Object obj;
      for (size_t m = 0; m < 4; m++)
      {
        geometry_msgs::msg::Point corner;
        corner.x = (int)(greenQuads.at(0).at(m).x);
        corner.y = (int)(greenQuads.at(0).at(m).y);
        corner.z = 0.0;
        obj.corners.push_back(corner);
      }
      Rect nowrect(boundingRect(greenQuads.at(0)));
      float iou = rectIoU(rect, nowrect);
      if (iou < 0.95)
      {
        obj.target_type = "rect_move";
        RCLCPP_INFO(this->get_logger(), "find rect_move,iou=%f", iou);
      }
      else
      {
        obj.target_type = "rect";
        RCLCPP_INFO(this->get_logger(), "find rect");
      }
      msg_object.objects.push_back(obj);

      // 绘制
      // 将 Point2f 转为 Point
      vector<Point> poly;
      poly.reserve(greenQuads.at(0).size());
      for (const Point2f &p : greenQuads.at(0))
        poly.emplace_back(cvRound(p.x), cvRound(p.y));
      // drawContours 要求 contours 的类型是 vector<vector<Point>>
      vector<vector<Point>> contours;
      contours.push_back(poly);
      drawContours(result_image, contours, 0, Scalar(0, 0, 255), 2);
      draw4points(greenQuads.at(0), result_image);
      rect = nowrect;
    }

    // 红色检测 - 使用稳定的范围
    cv::Mat mask1, mask2, redmask;
    cv::inRange(hsv, cv::Scalar(0, 120, 70), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(170, 120, 70), cv::Scalar(180, 255, 255), mask2);
    redmask = mask1 | mask2;

    // 适度的形态学操作
    cv::Mat sphere_k = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                 cv::Size(5, 5));
    cv::morphologyEx(redmask, redmask, cv::MORPH_CLOSE, sphere_k);
    cv::morphologyEx(redmask, redmask, cv::MORPH_OPEN, sphere_k);

    // 找轮廓
    std::vector<std::vector<cv::Point>> Scontours;
    cv::findContours(redmask, Scontours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    int valid_spheres = 0;

    for (size_t i = 0; i < Scontours.size(); i++)
    {
      double area = cv::contourArea(Scontours.at(i));
      if (area < 500)
        continue;

      // 计算最小外接圆
      Point2f center;
      float radius = 0;
      minEnclosingCircle(Scontours.at(i), center, radius);

      // 计算圆形度
      double perimeter = arcLength(Scontours.at(i), true);
      double circularity = 4 * CV_PI * area / (perimeter * perimeter);

      if (circularity > 0.7 && radius > 15 && radius < 200)
      {
        vector<Point2f> sphere_points =
            calculateStableSpherePoints(center, radius);

        // 绘制检测到的球体
        cv::circle(result_image, center, static_cast<int>(radius),
                   cv::Scalar(0, 255, 0), 2); // 绿色圆圈
        cv::circle(result_image, center, 3, cv::Scalar(0, 0, 255),
                   -1); // 红色圆心

        // 绘制球体上的四个点
        draw4points(sphere_points, result_image);

        // 显示半径信息
        string info_text = "R:" + to_string((int)radius);
        cv::putText(
            result_image, info_text, cv::Point(center.x - 15, center.y + 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);

        valid_spheres++;
        RCLCPP_INFO(this->get_logger(),
                    "Found sphere: (%.1f, %.1f) R=%.1f C=%.3f", center.x,
                    center.y, radius, circularity);
        // output msg
        msg_object.num_objects++;
        referee_pkg::msg::Object obj;
        obj.target_type = "sphere";
        for (int j = 0; j < 4; j++)
        {
          geometry_msgs::msg::Point corner;
          corner.x = (int)(sphere_points.at(j).x);
          corner.y = (int)(sphere_points.at(j).y);
          corner.z = 0.0;
          obj.corners.push_back(corner);
        }
        msg_object.objects.push_back(obj);
      }
    }

    imshow("tpl1", tpl[1]);
    waitKey(10);

    // 显示结果图像
    cv::imshow("Detection Result", result_image);
    cv::waitKey(1);

    // 发送最终消息
    Target_pub->publish(msg_object);
    RCLCPP_INFO(this->get_logger(), "Published %d  targets",
                msg_object.num_objects);
  }
  catch (const cv_bridge::Exception &e)
  {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
  }
}
/*
  从二值图像（白色为前景）中提取面积最大的白色连通区域并返回其裁剪图像。
  同时用 outBox 返回该连通区域的 bounding box（相对于原图）。
  若没有找到，返回空 Mat。
  这个函数在模板预处理和在帧处理中（提取多边形内部数字）会用到。
  返回裁剪后的图像是bw的指针,仅限bw作用域使用
*/
Mat VisionNode::extractLargestWhiteComponent(const Mat &bw, Rect &outBox)
{
  vector<vector<Point>> cnts;
  // 查找外部轮廓（白色区域）
  findContours(bw, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
  if (cnts.empty())
    return Mat();
  int imax = 0;
  double amax = 200;
  for (size_t i = 1; i < cnts.size(); ++i)
  {
    double a = contourArea(cnts[i]);
    if (a > amax)
    {
      amax = a;
      imax = (int)i;
    }
  }
  Rect b = boundingRect(cnts[imax]);
  // 防止越界
  outBox = b & Rect(0, 0, bw.cols, bw.rows);
  if (outBox.area() <= 0)
    return Mat();
  return bw(outBox);
}
/*
  归一化绝对差（SAD）用于像素比对：
  - 输入 a,b 必须都是单通道 8-bit (0/255) 且大小相同
  - 返回值在 [0,1] 之间，0 表示完全相同，1 表示完全不同（每像素差 255）
  该方法简单直观，适用于二值数字模板匹配（当无旋转、尺度差小且图像干净时效果好）
*/
double VisionNode::normalizedSAD(const Mat &a, const Mat &b)
{
  if (a.size() != b.size())
    RCLCPP_WARN(this->get_logger(), "SAD: size mismatch");
  double sum = 0.0;
  for (int y = 0; y < a.rows; ++y)
  {
    const uchar *pa = a.ptr<uchar>(y);
    const uchar *pb = b.ptr<uchar>(y);
    for (int x = 0; x < a.cols; ++x)
    {
      sum += fabs((double)pa[x] - (double)pb[x]) / 255.0;
    }
  }
  double maxv = double(a.rows) * double(a.cols);
  return sum / maxv; // 返回 0..1
}

//_______工具函数_______
// 中心 ROI
/*Rect centerROI(const Rect &r)
{
  int w = int(r.width);
  int h = int(r.height);
  int x = r.x + (r.width - w) / 2;
  int y = r.y + (r.height - h) / 2;
  return Rect(x, y, w, h);
}
// cosine similarity between two row vectors (CV_32F)
double cosineSim(const Mat &a, const Mat &b)
{
  if (a.empty() || b.empty())
    return -1;
  CV_Assert(a.type() == CV_32F && b.type() == CV_32F);
  double dot = a.dot(b);
  double na = norm(a, NORM_L2);
  double nb = norm(b, NORM_L2);
  if (na < 1e-6 || nb < 1e-6)
    return -1;
  return dot / (na * nb);
}

// 返回图像坐标系中的左下点索引
static int findLBidx(const vector<Point2f> &pts)
{
  int idx = 0;
  for (size_t i = 1; i < pts.size(); ++i)
  {
    if (pts[i].x < pts[idx].x - 1e-6)
      idx = (int)i;
    else if (fabs(pts[i].x - pts[idx].x) < 1e-6 && pts[i].y > pts[idx].y)
      idx = (int)i;
  }
  return idx;
}

// 找armor 阈值 -> 形态学 -> findContours）
vector<Rect> findarmor(const Mat &gray)
{
  Mat bw;
  // 目标为黑色：阈值并取反，使目标为白
  threshold(gray, bw, BLACK_THRESH, 255, THRESH_BINARY_INV);
  // 适度的形态学操作
  cv::Matksphere = cv::getStructuringElement(cv::MORPH_RECT,
                                             cv::Size(5, 5));
  cv::morphologyEx(bw, bw, cv::MORPH_CLOSE, sphere_k);
  cv::morphologyEx(bw, bw, cv::MORPH_OPEN, sphere_k);
  vector<vector<Point>> Scontours;
  findContours(bw, Scontours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  for (auto &c : Scontours)
  {
    double area = contourArea(c);
    if (area > MIN_ARMOR_AREA)
    {
      vector<Point> hullPts;
      vector<int> hullIdx;
      convexHull(c, hullIdx);
    }
  }
  return rects;
}*/
