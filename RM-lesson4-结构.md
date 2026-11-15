# 第四讲 模块详解（当前版本）

> 代码：`HNU_RM_Vision` 第四讲部分
> 最后更新：本文件随代码同步，行号以 `week4/` 副本为准

---

# 第 1 部分 · 当前程序结构

## 1.1 文件清单与状态

| 文件 | 行数 | 职责 | 状态 |
|---|---:|---|---|
| `include/tracking_state.hpp` | 108 | 跨帧状态容器（唯一写者 lesson04） | ✅ 在用 |
| `include/ekf_ctrv.hpp` | ~230 | **主滤波器**：CTRV 扩展卡尔曼 | ✅ 在用 |
| `lessons/lesson04_tracking.cpp` | 683 | 选板 → PnP → 状态机 → 滤波 → 预测 | ✅ 在用 |
| `include/kalman_cv.hpp` | ~115 | 匀速卡尔曼 | 🔶 对比用（`dbg::cv_filter`） |
| `include/ekf_spin.hpp` | ~230 | 圆周 EKF（中心+相位+半径） | 🔶 实验（`kEnableSpinEkf=false`） |
| `include/spin_model.hpp` | ~260 | 旧滑窗圆周模型 | ❌ **已停用（死文件）** |
| `src/main.cpp` | 730 | ROI、绘制、日志、真值比对 | ✅ 在用 |
| `include/debug_flags.hpp` | 18 | 日志/实验开关 | ✅ 在用 |

## 1.2 一帧的数据流

```
Detection[]（lesson3：4 角点 + class_id + large_armor + 置信度）
   │
   ├─① 选板      框宽<40px 跳过 → 正面度最大 → 死区 0.17 锁定
   ├─② PnP       solvePnPGeneric(IPPE) → 取重投影最小 → >3px 失败
   ├─③ 变换      p_cam → p_gimbal → p_world
   │    └ 门 2   PnP 距离 / 单目距离 ∈ [0.55, 1.8]
   ├─④ 噪声      σ_lat ∝ Z, σ_dep ∝ Z²；旋转到世界系；云台转速放大
   ├─⑤ 状态机    LOST→DETECTING(5帧)→TRACKING⇄TEMP_LOST(10帧)
   ├─⑥ 滤波      EkfCtrv 常开；（可选）KalmanCV 对比；（可选）EkfSpin
   └─⑦ 预测      250 ms → 反投影 → pred_corners / pred_hist
```

## 1.3 `tracking_state.hpp` 字段分组（清理后）

### 状态机
```cpp
Phase phase;  int same_target_frames;  int lost_frames;
```

### 当前目标
```cpp
int target_class_id;   cv::Point2f target_px;
float target_box_px;   float locked_face;   int selected_idx;
```

### 滤波器（三个）
```cpp
KalmanCV kf;       // 对比
EkfCtrv  ekf;      // 主
EkfSpin  spin_ekf; // 实验
bool spin_active;
Eigen::Vector3d cv_position, cv_velocity;   // 对比结果
cv::Point2f cv_pred_corners[4];  bool cv_valid, cv_has_pred;
```

### 解算结果
```cpp
Eigen::Vector3d position, velocity;
double distance_m;  double reproj_err_px;
```

### 时间 / 延迟
```cpp
TimePoint last_stamp;
Eigen::Quaterniond prev_q;  bool has_prev_q;  TimePoint prev_q_stamp;
double pipeline_latency_s;
double spin_rate;
```

### 可视化 / 日志
```cpp
cv::Point2f pred_corners[4];  bool has_pred;
cv::Point2f px_velocity;      // ROI 用
double pred_err_mm;  bool has_pred_err;
struct PredSample { ... 15 字段 ... };  std::deque<PredSample> pred_hist;
```

## 1.4 `lesson04_tracking.cpp` 的函数

| 函数 | 作用 |
|---|---|
| `armor_ring_prior(class_id)` | 各车型装甲环半径先验 |
| `estimate_distance(d, fx)` | 单目距离 `fx·W/w` |
| `frontality(d, fx)` | 正面度 `≈ cos(法向与视线夹角)` |
| `center_of(corners)` | 四角均值 |
| `armor_yaw_world(rvec, cal, imu_q, C, p)` | 装甲板法向世界 yaw + 朝外消歧（实验路径用） |
| `object_points(large)` | 板 3D 四角 |
| `reproj_error(...)` | 重投影误差 |
| `solve_armor(d, cal, rvec, tvec, err)` | IPPE 双解 → 取重投影最小 |
| `lesson04_tracking(...)` | 主流程 |
| `lesson04_roi_plan(...)` | ROI 计划 |

## 1.5 关键参数

| 组 | 参数 | 值 |
|---|---|---:|
| 运动学 | `kLargeW/kSmallW/kArmorH` | 0.225/0.135/0.125 m |
| | `kPredLeadS` | 0.25 s |
| | `kPredMaxShift` | 0.80 m |
| 卡尔曼 | `kQAccel / kQAccelZ / kQTheta / kQOmega` | 0.30 / 0.02 / 0.05 / 0.40 |
| 观测门限 | `kMinBoxPx` | 40 px |
| | `kDistRatioMax / Min` | 1.8 / 0.55 |
| 噪声模型 | `kCornerNoisePx` | 3 px |
| | `kSigLatMin/Max`、`kSigDepMin/Max` | 2~50 mm / 10~600 mm |
| 选板 | `kSwitchDeadZone` | 0.17（≈10°） |
| 状态机 | `kDetectFrames / kTempLostMax / kMaxReprojPx` | 5 / 10 / 3.0 |
| 实验 | `kEnableSpinEkf` | **false** |
| | `kSpinOmegaThresh / kAngleSigma` | 1.0 rad/s / 0.20 rad |

---

# 第 2 部分 · 死代码检查结果

## 2.1 已清理 ✅

| 项 | 处理 |
|---|---|
| `TrackingState::spin`（`SpinModel` 成员） | 已删 |
| `tracking_state.hpp` 里 `#include "spin_model.hpp"` | 已删 |
| `solve_armor` 的 `rvec_prev` / `has_prev` 参数 | 已删（原本 `(void)` 丢弃） |
| `TrackingState::last_rvec` / `has_rvec` | 已删（只写不读） |
| `TrackingState::gimbal_rate` | 已删（只写不读） |
| `lesson04_observe` 声明 | 已从 `lessons.hpp` 删除（函数体与调用早已没有） |

**清理后 `lesson04_tracking.cpp` 编译：零错误、零警告。**

## 2.2 仍保留（有意为之）🔶

| 项 | 为什么保留 | 怎么启用 |
|---|---|---|
| `include/spin_model.hpp` | 旧模型的完整实现，可作为"EKF 发散时"的参考/退路 | 目前**没有代码引用**，属于死文件，可随时删除 |
| `include/ekf_spin.hpp` + `state.spin_ekf` | 实验性旋转模型 | `kEnableSpinEkf = true` |
| `include/kalman_cv.hpp` + `state.cv_*` | 讲义任务 3「对比不同跟踪方案」 | `dbg::cv_filter = true` |
| `main.cpp` 的 `draw_cv_prediction` | 配合上一条 | 同上 |

## 2.3 结论

```
必删：include/spin_model.hpp（无任何引用）
可留：ekf_spin.hpp、kalman_cv.hpp（开关控制，用于对比实验）
```

> `spin_model.hpp` 建议直接删除；如果舍不得，至少在文件头标 `// DEPRECATED`。

---

# 第 3 部分 · 整车观测方案（解决旋转靶）

## 3.1 为什么要"整车"而不是"单板"

现在的链路是：**选一块板 → 对该板做 PnP → 用该板的状态预测未来**。

小陀螺时这个前提破了：

```
4 块板绕车体中心公转，半径 r ≈ 0.2 m
一块板转出视野 → 必须换板 → 位置瞬间跳 2r ≈ 0.4 m
单板模型把这 0.4 m 当成"真实运动" → 速度/角速度被污染
```

**整车观测的核心**：估计**车**的状态（中心、朝向、转速、半径），
装甲板只是它的"观测点"。换板不影响车的状态。

## 3.2 状态设计（三家开源的做法一致）

| 队伍 | 状态 |
|---|---|
| 深大 | `X_Center, V_X, Y_Armor, V_Y, Z_Center, V_Z, Theta, V_W, Radius`（9 维） |
| 同济 | `x vx y vy z vz a w r l h`（11 维，多 `l=r2−r1`、`h=z2−z1`） |
| 吉大 | GTSAM 因子图：整车位姿 + 速度 + 角速度 |

**建议我们的实现（9 维起步）**：

```
x = [ xc, vxc, yc, vyc, zc, vzc, θ, ω, r ]
```

- 装甲板 i 的位置：`P_i = C + r·[cos(θ + i·2π/N), sin(θ + i·2π/N), 0]`，N=4
- 量测：每块被检测到的板的 `(位置, 法向 yaw)`

## 3.3 当前 `EkfSpin` 已经具备什么

| 能力 | 状态 |
|---|---|
| 9 维状态（含 r） | ✅ 已有 |
| 位置量测 | ✅ 已有 |
| **装甲板 yaw 量测**（让 ω 可观测） | ✅ 已有（`update_with_angle`） |
| 多板关联（用所有检测，不只选中那块） | ❌ **缺** |
| 板数 N / 半径先验按车型 | 🔶 半径有，N 固定 4 |

## 3.4 缺的那一步：**多板关联 + 多量测更新**

### 现状

```cpp
// 只用选中的那一块板
spin_ekf.update_with_angle(p_world, plate_yaw, kAngleSigma);
```

### 改成

```cpp
// 对当帧所有检测到的板，逐个关联并更新
for (const auto &d2 : detections_this_frame) {
    // 1) 由状态预测每块板应有的相位
    //    θ_i = θ + i·2π/N  (i = 0..N-1)
    // 2) 用该板的 PnP 位置算实测相位
    //    φ_meas = atan2(p2.y - yc, p2.x - xc)
    // 3) 选角度残差最小的 i，作为它的 id
    // 4) 用 (p2, yaw2) 做一次 update_with_angle
}
```

**为什么有效**：

| 场景 | 单板 | 多板 |
|---|---|---|
| 一帧看到 2 块 | 只用 1 块 | 2 次更新，方差 ↓ |
| 换板瞬间 | 位置跳 0.4 m 被当运动 | 新板自动关联到正确 id，**跳变被吸收** |
| 板被遮挡 | 无观测 | 仍可用另一块 |

### 还要做的三件小事

1. **N（板数）从配置读**：步兵/英雄/哨兵 = 4，前哨站 = 3。
2. **关联门限**：角度残差 > 45° 视为不匹配，丢弃该检测。
3. **权重**：不同板的量测噪声可以按"正面度"加权（正对的板 yaw 更准）。

## 3.5 实施步骤（预计 200 行以内）

| 步骤 | 内容 | 文件 |
|---|---|---|
| 1 | `EkfSpin` 加 `update_multi(plates, N)`：内部做关联 + 多次 `update_with_angle` | `ekf_spin.hpp` |
| 2 | `lesson04` 把**当帧全部检测**（已 PnP 的）传进去，而不只是选中的 | `lesson04` |
| 3 | 半径与 N 按车型配置（`armor_ring_prior` 扩展） | `lesson04` |
| 4 | 打开 `kEnableSpinEkf`，与 CTRV 对比 | 开关 |
| 5 | 验收：旋转段 90 分位 < 150 mm、`spin_rate` 稳定在 ±2.0 | 日志 |

## 3.6 风险

| 风险 | 说明 |
|---|---|
| PnP 对每块板都要做 | 一帧 4 块 = 4 次 PnP（约 +60 µs），可接受 |
| 关联错误 | 两块板角度接近时可能配错，需要门限 |
| 半径仍不可观测 | 静态不转的目标永远估不出 r，必须靠先验 |
| 前哨站 3 层 | 还需 `h`（高度差），暂不做 |

---

# 第 4 部分 · 当前实测成绩

| 场景 | 中位 | 90 分位 |
|---|---:|---:|
| 静止 / 平移 | **1 mm** | 29 mm |
| 旋转 | 86 mm | 267 mm |
| 全段 | 70 mm | 241 mm |
| 径向抖动（距离稳定帧） | 3 mm | 29 mm |

对比历史：

| 版本 | 旋转段 90 分位 |
|---|---:|
| 纯 CTRV（`163217`） | 87 mm |
| 旋转 EKF（`215459`） | 845 mm |
| **本轮（关旋转 EKF）** | **267 mm** |

---

# 第 5 部分 · 已知问题

| # | 问题 | 影响 | 状态 |
|---|---|---|---|
| 1 | 换板时位置跳 2r | 旋转段 90 分位偏大 | 待整车观测 |
| 2 | IPPE 双解未消歧 | 偶发姿态翻转 | 已关闭消歧，靠重投影 |
| 3 | 半径不可观测（静态目标） | `spin_r` 乱跳 | 靠先验 |
| 4 | 前哨站三层高度未建模 | 打前哨站偏 | 未做 |
| 5 | 真值未接入（`truth=false`） | `err_mm` 全 NaN | 未接 |
| 6 | `spin_model.hpp` 死文件 | 仓库冗余 | 建议删 |
