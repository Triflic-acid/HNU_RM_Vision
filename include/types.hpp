#pragma once
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <array>
#include <chrono>
#include <optional>

namespace training {
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
constexpr double kPi = 3.14159265358979323846;

// IO 提供的算法输入；错误、连接状态和设备模式留在 IO 层。
struct CameraCalibration {
  cv::Mat matrix; // 3x3 CV_64F，相机光学坐标系
  cv::Mat distortion;
  Eigen::Matrix3d camera_to_gimbal = Eigen::Matrix3d::Identity();
  Eigen::Vector3d camera_to_gimbal_translation = Eigen::Vector3d::Zero();
  unsigned width=0, height=0;
  bool valid() const { return !matrix.empty(); }
};

struct SensorData {
  cv::Mat image;  // CV_8UC3，BGR，独立拥有像素内存
  TimePoint timestamp{};
  Eigen::Quaterniond imu_q = Eigen::Quaterniond::Identity();
  double bullet_speed = 0;  // m/s
};

// 第三讲的两种识别模式：
//   kArmor : 装甲板（14 类，kpt_shape [4,2]，输出 [1,26,8400]）
//   kRune  : 能量机关叶片（1 类，kpt_shape [4,2]，输出 [1,13,8400]）
enum class DetMode { kArmor, kRune };

// 第三讲输出：一个目标的图像观测（装甲板或符文叶片共用）。
//
// corners 点序随模式不同，务必看清：
//   kArmor : 左上、右上、右下、左下（顺时针）
//   kRune  : 左上、左下、右下、右上（逆时针，与符文数据集一致）
struct Detection {
  std::array<cv::Point2f, 4> corners{};
  cv::Rect2f box{};
  int class_id = -1;        // 装甲板：编号 0~6；符文：恒为 0
  bool large_armor = false; // 仅装甲板有效（大板 225x125 / 小板 135x125）
  double confidence = 0;
  // 仅对能量机关有效：R 标圆心（模板匹配得到，全帧共享）。
  // 为什么不让网络回归：圆心距叶片约 260 px，网络预测会塌到叶片附近，
  // 把 Pose mAP50-95 从 0.995 压到 0.597。
  std::optional<cv::Point2f> center;
};

// 兼容旧名字：第三讲原来的返回值类型
using Armor = Detection;

// 第四讲输出：当前观测时刻的单目标状态。无目标用 std::nullopt 表示。
struct Target {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();  // m，IMU稳定系中相对云台原点的目标向量
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();  // m/s，同一坐标系
};

// 第五讲输出：绝对角使用 rad，IO 边界统一转换单位和方向。
struct Command {
  bool control = false;
  bool shoot = false;
  double yaw = 0;
  double pitch = 0;
};
}  // namespace training