#pragma once
#include "types.hpp"
#include "tracking_state.hpp"
#include <optional>
#include <vector>

namespace training
{
    // lesson3 检测装甲板（与 mode 相关）
    std::vector<Detection> lesson03_detection(const cv::Mat &image, DetMode mode = DetMode::kArmor);
    // lesson4 跟踪目标（与 mode 相关）
    std::optional<Target> lesson04_tracking(const std::vector<Detection> &detections, DetMode mode,
                                            TimePoint timestamp, const Eigen::Quaterniond &imu_q, const CameraCalibration &calibration, TrackingState &state);
    // lesson5 弹道解算（与 mode 无关）
    Command lesson05_aiming(const Target &target, TimePoint timestamp, double bullet_speed);
} // namespace training
