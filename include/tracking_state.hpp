#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include <deque>

#include "types.hpp"
#include "kalman_cv.hpp"
#include "ekf_ctrv.hpp"
#include "spin_model.hpp"
#include "ekf_spin.hpp"

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
    float target_box_px{0.0F};  // 最近一次目标的框宽（px）
    float locked_face{0.0F};    // 锁定装甲板的正面度 cos，用于切板死区
    int selected_idx{-1};                // 选中目标的id

    // PnP 解算出的上一帧的姿态（用于 IPPE 双解消歧）
    cv::Vec3d last_rvec{0, 0, 0};
    bool has_rvec{false};
    
    // Kalman
    KalmanCV kf;
    // 旋转目标（小陀螺/前哨站）用 CTRV-EKF：状态含角速度 ω，能描述绕车体中心的圆周运动
    EkfCtrv ekf;
    // 旋转目标模型：中心 + 半径 + 相位 + 角速度，换板不影响中心估计
    SpinModel spin;          // 旧的滑窗圆周模型（保留做对比/兜底）
    EkfSpin spin_ekf;        // 圆周 EKF：θ 由装甲板 yaw 直接观测，ω 可收敛
    bool spin_active{false};   // 本帧是否用旋转模型出预测

    // ---- 对比用：匀速卡尔曼（CV）的结果 ----
    // 只在 dbg::cv_filter 打开时更新；关闭时这些字段保持默认值，不参与任何逻辑。
    Eigen::Vector3d cv_position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d cv_velocity{Eigen::Vector3d::Zero()};
    cv::Point2f cv_pred_corners[4]{};
    bool cv_valid{false};
    bool cv_has_pred{false};
    TimePoint last_stamp{};
    // 上一帧云台姿态：用于估计云台角速度（快速转动时放大量测噪声）
    Eigen::Quaterniond prev_q{Eigen::Quaterniond::Identity()};
    bool has_prev_q{false};
    TimePoint prev_q_stamp{};
    double gimbal_rate{0.0};   // rad/s，云台角速度（供日志观察）
    double spin_rate{0.0};     // rad/s，目标自转角速度（EKF 的 ω 状态）
    // 图像时刻 -> 本帧决策完成的延迟（s）。lesson05 预测时应把它加到飞行时间上。
    double pipeline_latency_s{0.0};

    // 最近一次解算结果（世界系，单位：m）
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    double distance_m{0.0}; // 最近一次解算得到的目标距离，0 表示未测得
    double reproj_err_px{0.0};

    // 像素速度（单位：px/s）：用于 ROI 预测，不需要 imu 和外参即可外推
    cv::Point2f px_velocity{0.0F, 0.0F};

    // 预测框及其误差
    cv::Point2f pred_corners[4]{};
    bool has_pred{false};
    double pred_err_mm{0.0};
    bool has_pred_err{false};

    // 预测历史
    struct PredSample
    {
      TimePoint t;       // 时刻
      Eigen::Vector3d p; // 预测坐标（世界系）

      // 探测到的目标（预测帧选中的板）
      int class_id{-1};
      float confidence{0.0F};
      float box_w{0.0F};
      float dist_m{0.0F};
      float reproj_px{0.0F};

      // 锁定状态
      Phase phase{Phase::kLost};
      int same_frames{0};
      int lost_frames{0};

      // 运动状态（诊断旋转/上下抖动用）
      float spin_rate{0.0F};   // rad/s，EKF/圆周模型的 ω
      float vz{0.0F};          // m/s，EKF 的垂直速度
      float spin_r{0.0F};      // m，圆周模型的半径估计
      int spin_on{0};          // 1 = 本帧使用圆周模型
      float latency_ms{0.0F};  // 图像时刻 -> 决策完成的延迟（ms）
    };
    std::deque<PredSample> pred_hist;
  };
} // namespace training
