#pragma once

#include <opencv2/core.hpp>

namespace training {
// 仅第四讲使用；由学员设计跨帧状态，其余讲不接收它。
//
// 下面几个字段是**初值**，为 lesson04_roi_plan()（决定是否进行 ROI）提供输入。
// 实现 ROI 判断时使用 PnP 距离
struct TrackingState {
  // 最近一次 PnP 得到的目标距离（单位：m）， 0 表示未测得
  double last_distance_m {0.0};
  // 最近一次目标在图像里的框宽（单位：px），作为 PnP 不可用时的退化依据
  float last_box_px {0.0F};
  // 最近一次目标中心（单位：px）
  cv::Point2f last_center{-1.0F, -1.0F};
  // 连续丢帧数
  int lost_frames {0};
};
}  // namespace training
