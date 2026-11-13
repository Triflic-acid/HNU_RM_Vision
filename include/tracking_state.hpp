#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include <deque>

#include "types.hpp"
#include "kalman_cv.hpp"

namespace training
{
  // 仅第四讲使用；由学员设计跨帧状态，其余讲不接收它。
  //
  // 下面几个字段是**初值**，为 lesson04_roi_plan()（决定是否进行 ROI）提供输入。
  // 实现 ROI 判断时使用 PnP 距离
  struct TrackingState
  {

    // 选板状态机
    enum class Phase
    {
      kLost,
      kDetecting,
      kTracking,
      kTempLost
    };
    Phase phase = Phase::kLost;
    int same_target_frames{0}; // 同一装甲板连续命中帧数
    int lost_frames{0};        // 连续丢帧数

    // 当前目标
    int target_class_id{-1};
    cv::Point2f target_px{-1.0F, -1.0F}; // 上一帧图像的位置（单位：px）
    float target_box_px{0.0F};

    // Kalman
    KalmanCV kf;
    TimePoint last_stamp{};

    // 最近一次解算结果（世界系，单位：m）
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    double distance_m{0.0};
    double reproj_err_px{0.0};

    // 最近一次 PnP 得到的目标距离（单位：m）， 0 表示未测得
    double last_distance_m{0.0};
    // 最近一次目标在图像里的框宽（单位：px），作为 PnP 不可用时的退化依据
    float last_box_px{0.0F};
    // 最近一次目标中心（单位：px）
    cv::Point2f last_center{-1.0F, -1.0F};

    // 预测框及其误差
    cv::Point2f pred_corners[4] {};
    bool has_pred {false};
    double pred_err_mm {0.0};
    bool has_pred_err {false};

    // 预测历史
    struct PredSample
    {
      TimePoint t;       // 时刻
      Eigen::Vector3d p; // 世界坐标
    };
    std::deque<PredSample> pred_hist;
  };
} // namespace training
