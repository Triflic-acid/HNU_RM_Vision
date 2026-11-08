#pragma once
#include "types.hpp"
#include "tracking_state.hpp"
#include <optional>
#include <vector>

namespace training
{
    // lesson3 检测装甲板（与 mode 相关）
    std::vector<Detection> lesson03_detection(const cv::Mat &image, DetMode mode = DetMode::kArmor);

    // ROI 放大推理
    std::vector<Detection> lesson03_detection_roi(const cv::Mat &image,
                                                  cv::Point2f center,
                                                  float zoom,
                                                  DetMode mode = DetMode::kArmor);

    // 亚像素精修
    void lesson03_refine_subpix(const cv::Mat &image,
                                std::vector<Detection> &detections,
                                int win = 5,
                                float max_shift = 6.0F);
    // 四边拟合直线求交
    void lesson03_refine_quad_lines(const cv::Mat &image,
                                    std::vector<Detection> &detections,
                                    int pts_per_edge = 15,
                                    float max_shift = 8.0F);
    // 帧间相位
    std::optional<cv::Point2f> lesson03_phase_shift(const cv::Mat &prev,
                                                    const cv::Mat &cur,
                                                    cv::Rect roi,
                                                    double min_response = 0.05);

    // lesson4 跟踪目标（与 mode 相关）
    std::optional<Target> lesson04_tracking(const std::vector<Detection> &detections,
                                            DetMode mode,
                                            TimePoint timestamp,
                                            const Eigen::Quaterniond &imu_q,
                                            const CameraCalibration &calibration,
                                            TrackingState &state);

    // 通过本帧观测更新跨帧状态
    void lesson04_observe(TrackingState &state, const std::vector<Detection> &detections);

    // lesson5 弹道解算（与 mode 无关）
    Command lesson05_aiming(const Target &target,
                            TimePoint timestamp,
                            double bullet_speed);

    // ROI实现
    struct RoiPlan
    {
        bool use = false;
        cv::Point2f center{};
        float side = 0.0F;
    };

    RoiPlan lesson04_roi_plan(const TrackingState &state,
                              const CameraCalibration &calibration,
                              int img_width,
                              int img_height,
                              float target_width_m);

} // namespace training
