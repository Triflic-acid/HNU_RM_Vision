#include "lessons.hpp"

#include "ballistics.hpp"
#include "debug_flags.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>

namespace training
{

  namespace
  {
    // 开火角误差阈值。仿真器 gimbal_command_deadband_deg = 0.05°，
    // 但那是"指令死区"，开火判据要宽一些：0.5° ≈ 5 m 处 44 mm，仍在装甲板内。
    constexpr double kFireAngleTol = 0.5 * M_PI / 180.0;
    // pitch 指令符号。
    // 由仿真器源码推：pitch_bevy = −cmd.pitch，反馈 pitch_deg = −rot.1
    //   ⇒ 反馈 = +指令（yaw 同理，yaw 不取反）
    // 日志佐证（2026-11-17 02:29 那次）：
    //   [aim] cmd=(-89.77,+1.26) → [IO] yaw=-89.90 pitch= 13.77→10.65（同向变化）
    //   [aim] cmd=(-53.18,+8.63) → [IO] yaw=-69.77 pitch=  8.07（yaw 向 -53 收敛、pitch≈8.6）
    // 所以这里**不取反**。
    constexpr double kPitchSign = +1.0;
    // 光轴误差的可信范围：目标被检测到时通常不会偏轴超过这个角度
    constexpr double kElevLimit = 25.0 * M_PI / 180.0;
    // yaw 死区：静止靶时目标不会动，误差小于这个角度就不动，
    // 否则 PnP 噪声会让云台一直左右微摆（实测静止靶帧间 cx 跳动 p90 = 18 px ⇒ 0.80°）。
    constexpr double kYawDeadband = 1.0 * M_PI / 180.0;
    // 云台的"水平位"：反馈 pitch = +25° 对应枪口水平（枪口相对云台体固定仰角 25°）。
    // 反馈与指令同号 ⇒ 要保持水平就下发 c = +25°。
    constexpr double kLevelPitchDeg = 25.0;
    constexpr double kLevelCmd = +kLevelPitchDeg * M_PI / 180.0;
    // 讲义说枪口/相机固定仰角 25°，但实测（弹丸从上方飞过）对不上，
    // 所以不再硬编码，改用 training::g_pitch_offset 连续可调（u/i 键 ±1°，o 键归零）。
  } // namespace

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
  Command lesson05_aiming(const Target &target, TimePoint timestamp, double bullet_speed)
  {
    (void)timestamp;

    Command cmd;

    // ---- 保持上次指令（关键）----
    // IO 层在 control=false 时会下发 yaw=0/pitch=0，等于命令云台立刻回正。
    // 于是"有解→无解→有解"会让云台在目标角和 0° 之间来回猛甩（实测就是抬头+抖动）。
    // 因此本帧解不出时**不放弃控制**，而是沿用上一次的指令。
    static Command last_cmd;
    static bool has_last = false;
    static training::TimePoint last_ok_t{};

    const auto hold_last = [&]() -> Command
    {
      const double age = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - last_ok_t)
                             .count();
      if (has_last && age < 0.5)
      {
        Command keep = last_cmd;
        keep.shoot = false; // 保持指向，但不要开火
        return keep;
      }
      // 真的丢了：不再"放开控制"，而是把云台回到水平位（反馈 +25° ⇔ 指令 −25°）
      Command level;
      level.control = true;
      level.shoot = false;
      level.yaw = has_last ? last_cmd.yaw : 0.0; // 方位保持不动
      level.pitch = kLevelCmd;                   // 俯仰回水平
      return level;
    };

    // 参数校验：弹速必须是有限正数
    if (!(bullet_speed > 1.0) || !std::isfinite(bullet_speed))
      return hold_last();

    // ---- 弹道参数：直接从仿真器模型读 ----
    //   simulation/simulator/config.toml
    //     [projectile] speed = 25.0, diameter = 0.017, mass = 0.0032
    //     [projectile.aerodynamics] enabled = false, air_density = 1.225, Cd = 0.22516
    ProjectileParams pp;
    pp.v0 = bullet_speed; // 用 IO 层读到的真实初速
    pp.diameter = 0.017;  // 17 mm
    pp.mass = 0.0032;     // 3.2 g
    pp.Cd = 0.22516;      // 仿真配置里的球体阻力系数
    pp.rho = 1.225;
    pp.drag = g_ballistics_drag; // 开关（默认关，与仿真一致）
    pp.g = 9.8;
    // Δt₁：观测时刻 → 现在（讲义原文用 Clock::now() − timestamp）
    {
      const auto now_tp = std::chrono::steady_clock::now();
      pp.latency = std::clamp(
          std::chrono::duration<double>(now_tp - timestamp).count(), 0.0, 0.5);
    }

    // ---- 解算 ----
    // 关键：用**云台系**的目标位置（lesson04 写入，未经世界系变换），
    // 解出的角度是相对云台当前指向的增量，再叠加当前云台角输出绝对角。
    // 这样"云台系→世界系"无论用什么约定（含 90° 偏置），都不会影响结果。
    // solve_aim 约定 **x 轴 = 正前方**，所以位置必须用**云台系**（x 前 / y 左 / z 上）。
    // 注意：相机系是 z 朝前、y 朝下，直接把 p_cam 传进去会算错（曾导致"云台不转"：
    //       目标在光轴上时 p_cam=(0,0,3) → 水平距离 0 → 直接返回空指令）。
    Eigen::Vector3d p = g_target_gimbal_valid ? g_target_gimbal : target.position;

    // 速度：用 lesson04 从**世界系速度**旋回云台系的值。
    // 不能用"相机系 6 帧拟合"——云台一转，p_cam 跟着变，拟合出来的是
    // 云台角速度而不是目标速度（实测 drop 在 −5.7 与 +1.6 之间跳，就是它造成的）。
    Eigen::Vector3d v = g_target_gimbal_valid ? g_target_gimbal_vel : target.velocity;
    // 物理上限保护：地面机器人不可能超过 6 m/s
    const double vn = v.norm();
    if (vn > 6.0)
      v *= 6.0 / vn;

    // 目标太近：不给解
    if (p.norm() < 0.5 || std::hypot(p.x(), p.y()) < 0.30)
      return hold_last();

    const AimSolution sol = solve_aim(p, v, pp);
    if (!sol.ok)
      return hold_last();

    // ---- 输出：伺服式（只用"目标相对光轴的偏差"+弹道补偿）----
    constexpr double kPitchLimit = 0.785; // 与仿真器 gimbal_pitch_limit 一致（±45°）

    // 目标相对光轴的方位/俯仰偏差，直接由 PnP 的相机系坐标算出。
    // 优点：完全不需要知道相机/枪口装了多少度，也不需要世界系。
    double azim = 0.0, elev = 0.0;
    {
      const Eigen::Vector3d pc = g_target_cam_valid ? g_target_cam : Eigen::Vector3d(0, 0, p.z());
      const double z = std::max(pc.z(), 0.1);
      // azim 必须取负：相机 x 轴（图像向右）在云台系里是 **−y**（见 R_camera2gimbal），
      // 而 ROS 的 yaw **逆时针为正（向左）**。所以"目标在图像右侧"⇒ yaw 要**减小**。
      // 之前写成 +atan2(pc.x, z)，等于把云台往右偏的方向转，形成正反馈：
      //   gy −89.9 → −85.7 → −71.8，azim 0.13 → 0.78 → 4.24 → 18.62（越大越偏）—— 就是"向左旋转"。
      azim = -std::atan2(pc.x(), z); // 图像右为正 ⇒ 云台需向右（yaw 减小）
      if (std::abs(azim) < kYawDeadband)
        azim = 0.0;                  // 死区：静止靶不许左右微摆
      elev = -std::atan2(pc.y(), z); // 上为正（图像 y 向下）

      // 钳位：单帧 PnP 野值可能给出几十度的"偏差"，直接照做就是"一进自瞄猛抬头"
      azim = std::clamp(azim, -kElevLimit, kElevLimit);
      elev = std::clamp(elev, -kElevLimit, kElevLimit);
    }

    // ==================== 静态标定模式 ====================
    // 触发：按 c（main 里置 training::g_calib = true）
    // 逻辑：对 pitch 注入 +3° / −3° 的台阶，测量 elev 的稳态变化，
    //       直接算出【环路增益】与【符号】—— 不必再靠猜。
    //       预期：cmd 增大 ⇒ 光轴下压 ⇒ 目标在画面里"上移" ⇒ elev 增大
    //            所以正常闭环下 Δelev/Δcmd ≈ +1（同号 = 需要取反才能使误差收敛）
    //       同时记录"对准时的 gp"，作为零位参考。
    if (training::g_calib)
    {
      static int step = 0;
      static training::TimePoint t_step{};
      static double gp0 = 0, elev0 = 0, gp_at = 0, elev_at = 0;
      const auto now_tp = std::chrono::steady_clock::now();
      const double el = std::chrono::duration<double>(now_tp - t_step).count();

      Command c;
      c.control = true;
      c.shoot = false;
      c.yaw = training::g_gimbal_yaw; // 标定期间不动 yaw

      switch (step)
      {
      case 0: // 记录基准
        c.pitch = training::g_gimbal_pitch;
        if (el > 0.6)
        {
          gp0 = g_gimbal_pitch;
          elev0 = elev;
          step = 1;
          t_step = now_tp;
        }
        break;
      case 1: // +3°
        c.pitch = gp0 + 3.0 * M_PI / 180.0;
        if (el > 0.8)
        {
          gp_at = g_gimbal_pitch;
          elev_at = elev;
          std::printf("[calib] +3.0°  gp %.2f→%.2f  elev %.2f→%.2f  Δelev/Δcmd = %+.2f\n",
                      gp0, gp_at, elev0 * 180 / M_PI, elev_at * 180 / M_PI,
                      (elev_at - elev0) / (3.0 * M_PI / 180.0));
          step = 2;
          t_step = now_tp;
        }
        break;
      case 2: // 回基准
        c.pitch = gp0;
        if (el > 0.6)
        {
          step = 3;
          t_step = now_tp;
        }
        break;
      case 3: // −3°
        c.pitch = gp0 - 3.0 * M_PI / 180.0;
        if (el > 0.8)
        {
          gp_at = g_gimbal_pitch;
          elev_at = elev;
          std::printf("[calib] -3.0°  gp %.2f→%.2f  elev %.2f→%.2f  Δelev/Δcmd = %+.2f\n",
                      gp0, gp_at, elev0 * 180 / M_PI, elev_at * 180 / M_PI,
                      (elev_at - elev0) / (-3.0 * M_PI / 180.0));
          step = 4;
          t_step = now_tp;
        }
        break;
      default: // 回基准并结束
        c.pitch = gp0;
        if (el > 0.6)
        {
          std::printf("[calib] 结束：基准 gp=%.2f°  elev=%.2f°  对准误差=%.2f°\n",
                      gp0, elev0 * 180 / M_PI, elev0 * 180 / M_PI);
          std::printf("[calib] 判读：|Δelev/Δcmd|≈1 → 环路耦合正常；"
                      "符号 + → cmd 与 elev 同向（需取反收敛）；符号 − → 已同向收敛\n");
          std::fflush(stdout);
          training::g_calib = false;
          step = 0;
        }
        break;
      }
      return c;
    }

    // 弹道补偿：发射仰角 θ 与"直线指向目标"的仰角之差（通常 1~3°）
    const double phi_target = std::atan2(p.z(), std::hypot(p.x(), p.y()));
    double drop = sol.pitch - phi_target;

    // drop 是"重力补偿"，物理上只有几十毫弧度；一旦超过 5° 说明速度估计被带偏了，
    // 此时按静止目标重算（速度置零），保证至少打"当前所在位置"。
    if (std::abs(drop) > 5.0 * M_PI / 180.0)
    {
      const AimSolution still = solve_aim(p, Eigen::Vector3d::Zero(), pp);
      if (still.ok)
        drop = still.pitch - phi_target;
    }
    if (std::abs(drop) > 10.0 * M_PI / 180.0)
      return hold_last(); // 仍不合理 → 沿用上次指令

    // 模式 0：cmd = 当前 + drop（**没有误差反馈**，会每帧叠加 drop → 一路抬头，勿用）
    // 模式 1：cmd = 当前 + 光轴偏差 + drop（elev 就是"还差多少"，自动收敛）
    if (training::g_aim_mode == 0)
    {
      cmd.yaw = std::remainder(g_gimbal_yaw + sol.yaw + training::g_yaw_offset, 2.0 * M_PI);
      cmd.pitch = g_gimbal_pitch + drop;
    }
    else
    {
      // 模式 1：伺服式，用"目标相对光轴的偏差"，与安装角无关
      const double dyaw_total =
          std::remainder(azim + training::g_yaw_offset, 2.0 * M_PI);

      // 死区放在**指令增量**上：静止靶时误差本来就只有 0.1~0.8°，
      // 若照单全收，云台会跟着 PnP 噪声左右微摆。这里小于 1.0° 直接不动。
      if (std::abs(dyaw_total) < kYawDeadband)
        cmd.yaw = g_gimbal_yaw;
      else
        cmd.yaw = std::remainder(g_gimbal_yaw + dyaw_total, 2.0 * M_PI);

      cmd.pitch = g_gimbal_pitch - elev - drop;
    }
    // 符号：反馈与指令同号（kPitchSign = +1，见文件头注释），偏置最后叠加。
    cmd.pitch = kPitchSign * cmd.pitch + training::g_pitch_offset;
    cmd.pitch = std::clamp(cmd.pitch, -kPitchLimit, kPitchLimit);
    cmd.control = true;

    last_cmd = cmd;
    has_last = true;
    last_ok_t = std::chrono::steady_clock::now();

    // ---- 调试打点：一眼看出到底哪一步不对 ----
    if (training::dbg::aim)
    {
      static int n = 0;
      if (++n % 15 == 0)
      {
        std::printf("[aim] gy=%+7.2f gp=%+7.2f | pg=(%+5.2f,%+5.2f,%+5.2f) "
                    "pc=(%+5.2f,%+5.2f,%+5.2f) cam_ok=%d | "
                    "azim=%+6.2f elev=%+6.2f drop=%+5.2f off=%+5.1f | "
                    "cmd=(%+7.2f,%+6.2f) shoot=%d\n",
                    g_gimbal_yaw * 180.0 / M_PI, g_gimbal_pitch * 180.0 / M_PI,
                    p.x(), p.y(), p.z(),
                    g_target_cam.x(), g_target_cam.y(), g_target_cam.z(),
                    g_target_cam_valid ? 1 : 0,
                    azim * 180.0 / M_PI, elev * 180.0 / M_PI, drop * 180.0 / M_PI,
                    training::g_pitch_offset * 180.0 / M_PI,
                    cmd.yaw * 180.0 / M_PI, cmd.pitch * 180.0 / M_PI,
                    cmd.shoot ? 1 : 0);
        std::fflush(stdout);
      }
    }

    // ---- 低通 + 限速：打断"追过头→更偏"的正反馈 ----
    // 1) 一阶低通，抑制单帧测量抖动
    // 2) 每帧允许的角度变化 ≤ max_rate·dt，避免一步甩到错误方向
    {
      static double prev_yaw = 0.0;
      static double prev_pitch = 0.0;
      static training::TimePoint prev_t{};
      static bool valid = false;
      static int state_lock_frames = 0;

      // 首次调用时用当前云台角作为起点，避免从 0 开始慢慢爬
      if (!valid)
      {
        prev_yaw = g_gimbal_yaw;
        prev_pitch = g_gimbal_pitch;
      }

      const double alpha = std::clamp(training::g_gimbal_lpf, 0.05, 1.0);
      const double yaw_lp = prev_yaw + alpha * std::remainder(cmd.yaw - prev_yaw, 2.0 * M_PI);
      const double pitch_lp = prev_pitch + alpha * (cmd.pitch - prev_pitch);

      double dt = 0.05;
      if (valid)
        dt = std::chrono::duration<double>(timestamp - prev_t).count();
      dt = std::clamp(dt, 0.005, 0.20);

      // ---- 分轴限速 ----
      // 俯仰：恒定低速（0.15 rad/s）。俯仰对噪声最敏感，一旦被带跑就会"抬头"丢目标。
      // 偏航：静止/自旋用低档，平移用高档（平移时目标横向移动快，需要跟得上）。
      const double yaw_rate = (training::g_motion == 1) // 1 = 平移
                                  ? training::g_yaw_rate_high
                                  : training::g_yaw_rate_low;
      const double max_step_yaw = yaw_rate * dt;
      const double max_step_pitch = training::g_pitch_max_rate * dt;

      const double dy = std::remainder(yaw_lp - prev_yaw, 2.0 * M_PI);
      const double dp = pitch_lp - prev_pitch;

      cmd.yaw = std::remainder(prev_yaw + std::clamp(dy, -max_step_yaw, max_step_yaw),
                               2.0 * M_PI);

      // 静止靶锁定后冻结俯仰：|elev| 连续很小 ⇒ 认为已对准，不再修正俯仰
      if (training::g_motion == 0 && std::abs(elev) < 0.5 * M_PI / 180.0)
      {
        if (++state_lock_frames >= 10)
          training::g_pitch_locked = true;
      }
      else if (training::g_motion != 0 || std::abs(elev) > 2.0 * M_PI / 180.0)
      {
        state_lock_frames = 0;
        training::g_pitch_locked = false;
      }

      if (training::g_pitch_locked)
        cmd.pitch = prev_pitch; // 冻结
      else
        cmd.pitch = prev_pitch + std::clamp(dp, -max_step_pitch, max_step_pitch);

      prev_yaw = cmd.yaw;
      prev_pitch = cmd.pitch;
      prev_t = timestamp;
      valid = true;
    }

    // 开火期间"锁角"：命中瞬间靶车会短暂消失/换板，若继续追会让云台狂抖。
    // shoot 后 kHoldAfterFire 秒内保持上一帧指令，不更新。
    {
      static training::TimePoint hold_until{};
      static double hold_yaw = 0.0, hold_pitch = 0.0;
      static bool holding = false;

      const auto now_tp = std::chrono::steady_clock::now();
      if (holding && now_tp < hold_until)
      {
        cmd.yaw = hold_yaw;
        cmd.pitch = hold_pitch;
      }
      else
      {
        holding = false;
      }

      // 开火门控（讲义：比较"云台实测角"和"指令角"）
      const double dyaw = std::remainder(cmd.yaw - g_gimbal_yaw, 2.0 * M_PI);
      const double dpitch = cmd.pitch - g_gimbal_pitch;
      cmd.shoot = std::abs(dyaw) < kFireAngleTol && std::abs(dpitch) < kFireAngleTol;

      if (cmd.shoot && !holding)
      {
        holding = true;
        hold_yaw = cmd.yaw;
        hold_pitch = cmd.pitch;
        hold_until = now_tp + std::chrono::milliseconds(150);
      }

    } // 锁角作用域结束

    return cmd;
  }
} // namespace training
