#include "lessons.hpp"

namespace training {
// 第五讲：弹道解算
//
// 输入格式：
//   target       : const Target&；position 单位 m，velocity 单位 m/s，二者均采用
//                  第四讲规定的 x 向前、y 向左、z 向上坐标系。
//                  装甲板与能量机关共用该结构，区别只在于第四讲怎么算出它。
//   timestamp    : TimePoint，target 对应的图像时刻，用于计算预测时间。
//   bullet_speed : double，弹丸初速度，单位 m/s，必须为有限正数。
//
// 输出格式：
//   Command：
//   - control : true 表示请求云台跟随；无法产生安全解时必须为 false；
//   - shoot   : true 表示请求发射，最终仍受 IO 层 allow_fire 和有效性门控；
//   - yaw     : 世界/IMU 参考系下的绝对偏航角，单位 rad；
//   - pitch   : 世界/IMU 参考系下的绝对俯仰角，单位 rad。
//
// 验收：仿真器完成静止靶、平移靶任务。
//
// 打符额外注意：
//   能量机关是圆周运动，目标点在 t_flight 内会转过 ω·t_flight，
//   所以 position 应当由第四讲**已经预测到击打时刻**再传进来；
//   也可以在这里用 velocity 做线性外推，但圆周运动下线性外推误差较大。
Command lesson05_aiming(const Target& target, TimePoint timestamp, double bullet_speed) {
  (void)target; (void)timestamp; (void)bullet_speed;
  return Command{};
}
}  // namespace training
