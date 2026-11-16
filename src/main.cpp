#include "lessons.hpp"
#include "io_context.hpp"

#include "ballistics.hpp" // g_ballistics_drag / g_gimbal_yaw
#include "debug_flags.hpp"
#include "talos_truth.hpp"

#include <opencv2/calib3d.hpp> // cv::projectPoints（画蓝准星要用）
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>
#include <deque>
#include <fstream>

namespace
{
  volatile std::sig_atomic_t interrupted = 0;
  void on_signal(int) { interrupted = 1; }

  // 统一画法：四边 + 编号角点 1~4 + 标签
  //   armor 点序 左上、右上、右下、左下（顺时针）
  //   rune  点序 左上、左下、右下、右上（逆时针）
  void draw_quad(cv::Mat &canvas,
                 const std::array<cv::Point2f, 4> &corners,
                 const cv::Scalar &edge, const cv::Scalar &dot,
                 const std::string &tag)
  {
    for (int i = 0; i < 4; ++i)
      cv::line(canvas, corners[i], corners[(i + 1) % 4], edge, 2, cv::LINE_AA);

    for (int i = 0; i < 4; ++i)
    {
      const cv::Point pt(cvRound(corners[i].x), cvRound(corners[i].y));
      cv::circle(canvas, pt, 3, dot, -1, cv::LINE_AA);
      cv::putText(canvas, std::to_string(i + 1), pt + cv::Point(5, -5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, dot, 2, cv::LINE_AA);
    }

    if (!tag.empty())
      cv::putText(canvas, tag,
                  cv::Point(cvRound(corners[0].x), cvRound(corners[0].y) - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, edge, 2, cv::LINE_AA);
  }

  void draw_armors(cv::Mat &canvas, const std::vector<training::Detection> &dets)
  {
    for (const auto &d : dets)
    {
      char text[64];
      std::snprintf(text, sizeof(text), "id=%d %.2f", d.class_id, d.confidence);
      draw_quad(canvas, d.corners, cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), text);
    }
  }

  // 250 ms 后的预测框（黄色）
  void draw_prediction(cv::Mat &canvas, const training::TrackingState &st)
  {
    if (!st.has_pred)
      return;

    for (int i = 0; i < 4; ++i)
      cv::line(canvas, st.pred_corners[i], st.pred_corners[(i + 1) % 4],
               cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

    cv::putText(canvas, "pred 250ms",
                cv::Point(cvRound(st.pred_corners[0].x), cvRound(st.pred_corners[0].y) - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
  }

  // 对比用：匀速卡尔曼(CV)的预测框（品红）。仅 dbg::cv_filter 打开时有内容。
  void draw_cv_prediction(cv::Mat &canvas, const training::TrackingState &st)
  {
    if (!st.cv_has_pred)
      return;
    for (int i = 0; i < 4; ++i)
      cv::line(canvas, st.cv_pred_corners[i], st.cv_pred_corners[(i + 1) % 4],
               cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
    cv::putText(canvas, "cv 250ms",
                cv::Point(cvRound(st.cv_pred_corners[0].x), cvRound(st.cv_pred_corners[0].y) - 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
  }

  // 云台/相机的固定偏置（与 lesson04 的 gimbal_yaw_offset 保持一致）
  double gimbal_yaw_offset()
  {
    static const double v = []
    {
      double deg = 90.0;
      if (const char *e = std::getenv("GIMBAL_YAW_OFFSET_DEG"))
        deg = std::atof(e);
      return deg * M_PI / 180.0;
    }();
    return v;
  }

  Eigen::Matrix3d r_gimbal_to_world(const Eigen::Quaterniond &imu_q)
  {
    const double off = gimbal_yaw_offset();
    if (std::abs(off) < 1e-9)
      return imu_q.toRotationMatrix();
    Eigen::Matrix3d Rz = Eigen::Matrix3d::Identity();
    Rz(0, 0) = std::cos(off);
    Rz(0, 1) = -std::sin(off);
    Rz(1, 0) = std::sin(off);
    Rz(1, 1) = std::cos(off);
    return Rz * imu_q.toRotationMatrix();
  }

  // 世界系点 → 像素。用于把弹道解算出的击打点画到画面上。
  bool world_to_pixel(const Eigen::Vector3d &p_world,
                      const Eigen::Quaterniond &imu_q,
                      const training::CameraCalibration &cal,
                      cv::Point2f &px)
  {
    if (!cal.valid())
      return false;
    const Eigen::Matrix3d R_g2w = r_gimbal_to_world(imu_q);
    const Eigen::Vector3d p_g = R_g2w.transpose() * p_world; // → 云台系
    const Eigen::Vector3d p_c = cal.camera_to_gimbal.transpose() *
                                (p_g - cal.camera_to_gimbal_translation); // → 相机系
    if (p_c.z() <= 0.1)
      return false; // 在相机后方

    std::vector<cv::Point3f> obj{cv::Point3f(static_cast<float>(p_c.x()),
                                             static_cast<float>(p_c.y()),
                                             static_cast<float>(p_c.z()))};
    std::vector<cv::Point2f> img;
    cv::projectPoints(obj, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0),
                      cal.matrix, cal.distortion, img);
    if (img.empty())
      return false;
    px = img[0];
    return true;
  }

  // 击打点：蓝色准星 + yaw/pitch/τ 文字
  void draw_aim_point(cv::Mat &canvas, const training::TrackingState &st,
                      const Eigen::Quaterniond &imu_q,
                      const training::CameraCalibration &cal)
  {
    if (!st.aim_valid)
      return;

    cv::Point2f px;
    if (!world_to_pixel(st.aim_point, imu_q, cal, px))
      return;
    if (px.x < 0 || px.y < 0 || px.x >= canvas.cols || px.y >= canvas.rows)
      return;

    const cv::Scalar kBlue(255, 128, 0); // BGR
    const cv::Point c(cvRound(px.x), cvRound(px.y));
    cv::drawMarker(canvas, c, kBlue, cv::MARKER_CROSS, 26, 2, cv::LINE_AA);
    cv::circle(canvas, c, 11, kBlue, 2, cv::LINE_AA);

    char buf[96];
    std::snprintf(buf, sizeof(buf), "AIM %.2f %.2f %.0fms",
                  st.aim_yaw * 180.0 / M_PI,
                  st.aim_pitch * 180.0 / M_PI,
                  st.aim_tau * 1000.0);
    cv::putText(canvas, buf, c + cv::Point(16, -14),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, kBlue, 2, cv::LINE_AA);
  }

  // 打符：叶片四角 + 靶心 + 圆心
  void draw_runes(cv::Mat &canvas, const std::vector<training::Detection> &dets)
  {
    const cv::Scalar kLogo(255, 200, 0); // 青
    const cv::Scalar kAim(0, 0, 255);    // 红

    for (size_t i = 0; i < dets.size(); ++i)
    {
      char text[64];
      std::snprintf(text, sizeof(text), "RUNE%zu %.2f", i + 1, dets[i].confidence);
      draw_quad(canvas, dets[i].corners, cv::Scalar(0, 255, 0), cv::Scalar(0, 165, 255), text);
    }
    if (dets.empty())
      return;

    const auto &center = dets.front().center;

    // 标注圆心
    if (center)
    {
      const cv::Point c(cvRound(center->x), cvRound(center->y));
      cv::drawMarker(canvas, c, kLogo, cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
      cv::circle(canvas, c, 10, kLogo, 2, cv::LINE_AA);
      cv::putText(canvas, "R", c + cv::Point(12, -12),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, kLogo, 2, cv::LINE_AA);
    }

    // 靶心（每片叶子的中心）
    for (const auto &d : dets)
    {
      const cv::Point2f t = (d.corners[0] + d.corners[1] + d.corners[2] + d.corners[3]) * 0.25F;
      const cv::Point c(cvRound(t.x), cvRound(t.y));
      cv::drawMarker(canvas, c, kAim, cv::MARKER_TILTED_CROSS, 22, 2, cv::LINE_AA);
      cv::circle(canvas, c, 12, kAim, 2, cv::LINE_AA);

      if (center)
      {
        const double r = cv::norm(t - *center);
        cv::line(canvas, c, cv::Point(cvRound(center->x), cvRound(center->y)),
                 kAim, 1, cv::LINE_AA);
        char text[64];
        std::snprintf(text, sizeof(text), "r=%.0fpx", r);
        cv::putText(canvas, text, c + cv::Point(14, 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, kAim, 2, cv::LINE_AA);
      }
    }
  }

  // 圆心时序滤波：在原图坐标里做，避免与 ROI 裁剪坐标混用
  // 连续 kEscape 次都判为"跳变"就认为旧值错了，接受新值（否则会永久锁死）
  struct RuneCenterFilter
  {
    static constexpr int kLostMax{3};
    static constexpr int kEscape{5};

    cv::Point2f last{-1.F, -1.F};
    int lost{0};
    int rejects{0};

    std::optional<cv::Point2f> apply(const std::optional<cv::Point2f> &in, float limit)
    {
      if (in)
      {
        if (last.x >= 0.0F && cv::norm(*in - last) > limit)
        {
          if (++rejects < kEscape)
            return last;
          rejects = 0;
        }
        else
        {
          rejects = 0;
        }
        last = *in;
        lost = 0;
        return last;
      }
      if (last.x >= 0.0F && ++lost <= kLostMax)
        return last;
      last = cv::Point2f(-1.0F, -1.0F);
      lost = 0;
      return std::nullopt;
    }

    void reset()
    {
      last = cv::Point2f(-1.0F, -1.0F);
      lost = 0;
      rejects = 0;
    }
  };

  // 按键切换事件日志：写 stdout + gui_events.log，便于事后回溯"当时开的哪个开关"
  std::ofstream &gui_event_log()
  {
    static std::ofstream ofs;
    if (!ofs.is_open())
      ofs.open("simulation/outputs/gui_events.log", std::ios::app);
    return ofs;
  }

  void log_gui(const char *key, const char *name, const char *state)
  {
    const auto now = std::chrono::system_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch())
                        .count();
    std::printf("[gui] %s -> %s\n", name, state);
    std::fflush(stdout);
    auto &ofs = gui_event_log();
    if (ofs.is_open())
    {
      ofs << ms << ',' << key << ',' << name << ',' << state << '\n';
      ofs.flush();
    }
  }

  enum class RoiMode
  {
    kOff,
    kAuto,
    kManual
  };
  enum class RefineMode
  {
    kOff,
    kSubpix,
    kLines
  };

  const char *refine_name(RefineMode m)
  {
    switch (m)
    {
    case RefineMode::kSubpix:
      return "subpix";
    case RefineMode::kLines:
      return "lines";
    default:
      return "off";
    }
  }
  // 左下角：预测误差 + 距离 + 状态
  void draw_error_panel(cv::Mat &canvas, const training::TrackingState &st)
  {
    char line1[160];
    char line2[160];

    if (st.has_pred_err)
      std::snprintf(line1, sizeof(line1), "pred_err(250ms) = %6.1f mm", st.pred_err_mm);
    else
      std::snprintf(line1, sizeof(line1), "pred_err(250ms) =    ---");

    const char *ph{(st.phase == training::TrackingState::Phase::kTracking)    ? "TRACK"
                   : (st.phase == training::TrackingState::Phase::kDetecting) ? "DETECT"
                   : (st.phase == training::TrackingState::Phase::kTempLost)  ? "TEMP_LOST"
                                                                              : "LOST"};

    std::snprintf(line2, sizeof(line2), "dist=%5.2f m  reproj=%4.1f px  %s",
                  st.distance_m, st.reproj_err_px, ph);

    const int y{canvas.rows - 46};
    cv::putText(canvas, line1, cv::Point(12, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    cv::putText(canvas, line2, cv::Point(12, y + 26),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
  }
}

// 真值采样：按预测的目标时刻回查
struct TruthSample
{
  training::TimePoint t;
  Eigen::Vector3d p;
  float yaw{0.F};
  int label{0};
};
constexpr double kBaselineMm{0.0};

// 预测值 / 真值 CSV 日志
// 与 dbg::pred_log 配合：关闭时不打开文件、不产生任何 IO
class PredLog
{
public:
  bool open(const char *path)
  {
    fs_.open(path, std::ios::out | std::ios::trunc);
    if (!fs_.is_open())
      return false;
    fs_ << "time_ms,pred_t_ms,px,py,pz,tx,ty,tz,err_mm,label,"
           "dist_m,reproj_px,phase,class_id,conf,box_w,same_frames,lost_frames,"
           "spin_rate,vz,spin_r,spin_on,latency_ms,aim_x,aim_y,aim_z,aim_tau\n";
    return true;
  }

  void write(const training::TrackingState::PredSample &rec,
             const Eigen::Vector3d *truth, double err_mm, int label,
             long long now_ms, long long pred_t_ms)
  {
    if (!fs_.is_open())
      return;

    fs_ << now_ms
        << ','
        << pred_t_ms
        << ','
        << rec.p.x()
        << ','
        << rec.p.y()
        << ','
        << rec.p.z()
        << ',';

    if (truth)
      fs_ << truth->x()
          << ','
          << truth->y()
          << ','
          << truth->z()
          << ','
          << err_mm
          << ','
          << label
          << ',';
    else
      fs_ << ",,,NaN,-1,";

    fs_ << rec.dist_m
        << ','
        << rec.reproj_px
        << ','
        << phase_name(rec.phase)
        << ','
        << rec.class_id
        << ','
        << rec.confidence
        << ','
        << rec.box_w
        << ','
        << rec.same_frames
        << ','
        << rec.lost_frames
        << ','
        << rec.spin_rate
        << ','
        << rec.vz
        << ','
        << rec.spin_r
        << ','
        << rec.spin_on
        << ','
        << rec.latency_ms
        << ','
        << rec.aim_x
        << ','
        << rec.aim_y
        << ','
        << rec.aim_z
        << ','
        << rec.aim_tau
        << '\n';

    if (++count_ % 64 == 0)
      fs_.flush();
  }

  void close()
  {
    if (fs_.is_open())
    {
      fs_.flush();
      fs_.close();
    }
  }

private:
  static const char *phase_name(training::TrackingState::Phase p)
  {
    switch (p)
    {
    case training::TrackingState::Phase::kTracking:
      return "TRACK";
    case training::TrackingState::Phase::kDetecting:
      return "DETECT";
    case training::TrackingState::Phase::kTempLost:
      return "TEMP_LOST";
    default:
      return "LOST";
    }
  }

  std::ofstream fs_;
  int count_{0};
};

// 每帧每个检测输出一行日志
class DetLog
{
public:
  bool open(const char *path)
  {
    fs_.open(path, std::ios::out | std::ios::trunc);
    if (!fs_.is_open())
      return false;
    fs_ << "time_ms,n_det,idx,class_id,conf,box_w,cx,cy,est_dist,selected\n";
    return true;
  }

  void write(long long now_ms, int idx, int n_det,
             const training::Detection &d, double est_dist, bool selected)
  {
    if (!fs_.is_open())
      return;
    const cv::Point2f c{(d.corners[0] + d.corners[1] + d.corners[2] + d.corners[3]) * 0.25F};
    fs_ << now_ms
        << ','
        << n_det
        << ','
        << idx
        << ','
        << d.class_id
        << ','
        << d.confidence
        << ','
        << d.box.width
        << ','
        << c.x
        << ','
        << c.y
        << ','
        << est_dist
        << ','
        << (selected ? 1 : 0)
        << '\n';
    if (++count_ % 256 == 0)
      fs_.flush();
  }

  void close()
  {
    if (fs_.is_open())
    {
      fs_.flush();
      fs_.close();
    }
  }

private:
  std::ofstream fs_;
  int count_{0};
};

namespace training::dbg
{
  bool io{true};          // [IO]   帧率/姿态、[read] nullopt、[Write]
  bool det{false};        // [det]  每帧检测耗时、模型/模板加载
  bool rune{true};        // [rune] 圆心 r/w、模板分数
  bool gui{true};         // [gui]  按键切换提示
  bool cv_filter{false};  // 对比：同时运行匀速卡尔曼(CV)，画出品红色预测框
  bool demo_write{false}; // 演示：无视算法输出，向云台发出按正弦函数运动的指令
  bool truth{true};       // 返回仿真器数据
  bool pred_log{true};    // 输出预测值 & 真实值日志（依赖 truth）
} // namespace training::dbg

int main(int argc, char **argv)
{
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  // 空气阻力默认关（与仿真器 config.toml 一致），可用环境变量或按 d 切换
  if (const char *e = std::getenv("BALLISTIC_DRAG"); e != nullptr && e[0] == '1')
  {
    training::g_ballistics_drag = true;
    std::printf("[init] 空气阻力已通过 BALLISTIC_DRAG=1 打开\n");
  }

  const auto config = training::load_config(argc > 1 ? argv[1] : "configs/simulation.yaml");
  const auto calibration = training::load_calibration(config);
  training::io::IoContext devices{config};
  training::TrackingState tracking;

  training::truth::ShmTruth truth;
  bool truth_ok{training::dbg::truth && truth.open()};
  std::vector<training::truth::Target> truth_targets;
  std::vector<training::truth::Rune> truth_runes;
  std::deque<TruthSample> truth_hist;

  PredLog pred_log;
  const bool pred_log_ok{training::dbg::pred_log && pred_log.open("simulation/outputs/pred_log.csv")};
  DetLog det_log;
  const bool det_log_ok{training::dbg::pred_log && det_log.open("simulation/outputs/det_log.csv")};

  devices.open();

  // 把启动默认值写进事件日志（这样日志里总能找到"本次运行是什么配置"）
  {
    auto &ofs = gui_event_log();
    if (!ofs.is_open())
      ofs.open("simulation/outputs/gui_events.log", std::ios::app);
    if (ofs.is_open())
    {
      const auto now = std::chrono::system_clock::now();
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch())
                          .count();
      ofs << ms << ",start,默认配置," << "ROI=off,refine=subpix,drag="
          << (training::g_ballistics_drag ? "on" : "off") << '\n';
      ofs.flush();
    }
    std::printf("[init] 默认配置: ROI=off, refine=subpix, drag=%s\n",
                training::g_ballistics_drag ? "on" : "off");
    std::fflush(stdout);
  }

  training::DetMode mode{training::DetMode::kArmor}; // 默认自瞄；按 m 切换

  RoiMode roi_mode = RoiMode::kOff; // 默认关闭 ROI 裁剪
  const float manual_zoom = 2.0F;
  RefineMode refine_mode = RefineMode::kSubpix; // 默认开启亚像素精修
  RuneCenterFilter rune_filter;
  const int subpix_win = 5;
  double last_ms = 0.0;

  while (!interrupted)
  {
    const auto input = devices.read();

    training::Command command{};
    std::vector<training::Detection> detections;
    bool has_logo = false;

    if (input)
    {

      // ROI
      const auto t0 = std::chrono::steady_clock::now();

      bool used_roi = false;
      if (roi_mode == RoiMode::kAuto)
      {
        const float w_m = (mode == training::DetMode::kArmor) ? 0.225F : 0.0F;
        const auto plan = training::lesson04_roi_plan(
            tracking, calibration, input->image.cols, input->image.rows, w_m);
        if (plan.use)
        {
          detections = training::lesson03_detection_roi(
              input->image, plan.center,
              static_cast<float>(input->image.cols) / plan.side, mode);
          used_roi = true;
        }
      }
      else if (roi_mode == RoiMode::kManual)
      {
        const cv::Point2f c{(tracking.target_px.x >= 0.F)
                                ? tracking.target_px
                                : cv::Point2f(input->image.cols * 0.5F,
                                              input->image.rows * 0.5F)};
        detections = training::lesson03_detection_roi(input->image, c, manual_zoom, mode);
        used_roi = true;
      }

      if (!used_roi)
        detections = training::lesson03_detection(input->image, mode);

      // 亚像素精修
      if (!detections.empty())
      {
        if (refine_mode == RefineMode::kSubpix)
          training::lesson03_refine_subpix(input->image, detections, subpix_win);
        else if (refine_mode == RefineMode::kLines)
          training::lesson03_refine_quad_lines(input->image, detections);
      }

      // 能量机关圆心：在原图坐标里做时序滤波
      // lesson03 只负责单帧
      // 若把时序放进 lesson03，ROI 裁剪坐标会和原图坐标混用
      // 表现为圆心被锁到画面角落无法归位
      if (mode == training::DetMode::kRune && !detections.empty())
      {
        const float w = std::max(detections.front().box.width, 4.F);
        const auto filtered = rune_filter.apply(detections.front().center,
                                                std::max(0.6F * w, 12.F));
        for (auto &d : detections)
          d.center = filtered;
      }

      // 计时
      last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

      // 真值采集，供 250 ms 后按目标时刻回查
      if (truth_ok)
      {
        truth.poll(truth_targets, truth_runes);
        if (!truth_targets.empty())
        {
          truth_hist.push_back({input->timestamp,
                                Eigen::Vector3d(truth_targets[0].position[0],
                                                truth_targets[0].position[1],
                                                truth_targets[0].position[2]),
                                truth_targets[0].yaw,
                                static_cast<int>(truth_targets[0].armor_label)});
          while (truth_hist.size() > 120)
            truth_hist.pop_front();
        }
      }

      // 写回
      // 状态只由 lesson4 更新
      // main 不直接写入 TrackingState 字段
      // lesson4 目标跟踪
      const auto target{training::lesson04_tracking(detections,
                                                    mode,
                                                    input->timestamp,
                                                    input->imu_q,
                                                    calibration,
                                                    tracking)};

      // 写入det_log
      if (det_log_ok)
      {
        const long long now_ms{std::chrono::duration_cast<std::chrono::milliseconds>(
                                   input->timestamp.time_since_epoch())
                                   .count()};
        const int n_det{static_cast<int>(detections.size())};
        for (int i = 0; i < n_det; ++i)
        {
          const auto &dd{detections[static_cast<size_t>(i)]};
          const double est{(calibration.valid() && dd.box.width > 1.F)
                               ? calibration.matrix.at<double>(0, 0) * (dd.large_armor ? 0.225 : 0.135) / dd.box.width
                               : 0.0};
          det_log.write(now_ms, i, n_det, dd, est, /*selected=*/i == tracking.selected_idx);
        }
      }

      // lesson5 开火
      if (target)
      {
        // 把当前云台角同步给 lesson05（开火门控用）
        // imu_q = Rz(yaw)·Ry(−pitch)·Rx(roll)，所以反解：
        {
          const Eigen::Matrix3d R = input->imu_q.toRotationMatrix();
          training::g_gimbal_yaw = std::atan2(R(1, 0), R(0, 0));
          training::g_gimbal_pitch = -std::asin(std::clamp(R(2, 0), -1.0, 1.0));
        }

        command = training::lesson05_aiming(*target,
                                            input->timestamp,
                                            input->bullet_speed);
      }

      // 250 ms 前对当前状态的预测值与当前测量值的对比
      //   dbg::truth  开：与真值比，写完整行（仿真器中）
      //   dbg::truth  关：只记预测值（实车）
      while (!tracking.pred_hist.empty())
      {
        const auto &rec{tracking.pred_hist.front()};
        const auto age{input->timestamp - rec.t};

        if (age < std::chrono::milliseconds(250))
          break; // 没到 250 ms

        if (age > std::chrono::milliseconds(310))
        {
          tracking.pred_hist.pop_front(); // 丢弃过期样本
          continue;
        }

        const auto t_target{rec.t + std::chrono::milliseconds(250)};

        const TruthSample *best{nullptr};
        double best_dt{1e9};
        if (truth_ok)
        {
          for (const auto &s : truth_hist)
          {
            const double dt_ms{std::abs(std::chrono::duration<double, std::milli>(
                                            s.t - t_target)
                                            .count())};
            if (dt_ms < best_dt)
            {
              best_dt = dt_ms;
              best = &s;
            }
          }
          if (best && best_dt >= 40.0)
            best = nullptr; // 真值离目标时刻太远
        }

        if (best)
        {
          tracking.pred_err_mm = (best->p - rec.p).norm() * 1000.0 - kBaselineMm;
          tracking.has_pred_err = true;
        }

        if (pred_log_ok)
        {
          const auto ms = [](const training::TimePoint &t)
          {
            return static_cast<long long>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    t.time_since_epoch())
                    .count());
          };
          pred_log.write(rec,
                         best ? &best->p : nullptr,
                         tracking.pred_err_mm,
                         best ? best->label : -1,
                         ms(input->timestamp),
                         ms(t_target));
        }

        tracking.pred_hist.pop_front();
      }

      if (mode == training::DetMode::kRune && !detections.empty())
        has_logo = detections.front().center.has_value();
    }
    if (interrupted)
      break;

    if (training::dbg::demo_write)
    {
      const double t{std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()};

      command.control = true;
      command.shoot = false;
      command.yaw = -1.5708 + 0.6 * std::sin(t * 0.5);
      command.pitch = 0.4363 + 0.10 + 0.05 * std::sin(t * 0.3);
    }

    devices.write(command);

    if (devices.config.display)
    {
      // 没有新帧时也刷新窗口，否则窗口会假死、按键无响应
      cv::Mat canvas;
      if (input)
      {
        canvas = input->image.clone();
        if (mode == training::DetMode::kArmor)
        {
          draw_armors(canvas, detections);
          draw_prediction(canvas, tracking);
          draw_cv_prediction(canvas, tracking);
          draw_aim_point(canvas, tracking, input->imu_q, calibration);
        }
        else
        {
          draw_runes(canvas, detections);
        }
      }
      else
      {
        canvas = cv::Mat(240, 640, CV_8UC3, cv::Scalar(30, 30, 30));
        cv::putText(canvas, "waiting for /image_raw ...", cv::Point(20, 130),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 200, 255), 2);
      }

      {
        const char *roi_txt = (roi_mode == RoiMode::kAuto)     ? "auto"
                              : (roi_mode == RoiMode::kManual) ? "manual"
                                                               : "off";
        char hud[200];
        std::snprintf(hud, sizeof(hud),
                      "[%s] %zu targets  %.1fms  ROI:%s  refine:%s  drag:%s  m:mode z:roi p:refine d:drag",
                      mode == training::DetMode::kArmor ? "ARMOR" : "RUNE",
                      detections.size(), last_ms, roi_txt, refine_name(refine_mode),
                      training::g_ballistics_drag ? "on" : "off");
        cv::putText(canvas, hud, cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    mode == training::DetMode::kArmor ? cv::Scalar(0, 255, 0)
                                                      : cv::Scalar(0, 128, 255),
                    2);
      }

      if (mode == training::DetMode::kArmor)
        draw_error_panel(canvas, tracking);

      cv::imshow("RoboMaster training", canvas);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q')
        break;
      if (key == 'm' || key == 'M' || key == '\t')
      {
        mode = (mode == training::DetMode::kArmor) ? training::DetMode::kRune
                                                   : training::DetMode::kArmor;
        tracking = training::TrackingState{};
        rune_filter.reset();
        log_gui("m", "检测模式",
                mode == training::DetMode::kArmor ? "装甲板" : "能量机关");
      }
      else if (key == 'z' || key == 'Z')
      {
        roi_mode = (roi_mode == RoiMode::kOff)    ? RoiMode::kAuto
                   : (roi_mode == RoiMode::kAuto) ? RoiMode::kManual
                                                  : RoiMode::kOff;
        log_gui("z", "ROI",
                roi_mode == RoiMode::kAuto     ? "自动(PnP)"
                : roi_mode == RoiMode::kManual ? "手动 2.0x"
                                               : "关闭");
      }
      else if (key == 'd' || key == 'D')
      {
        training::g_ballistics_drag = !training::g_ballistics_drag;
        log_gui("d", "空气阻力",
                training::g_ballistics_drag ? "开" : "关（抛体模型）");
      }
      else if (key == 'p' || key == 'P')
      {
        refine_mode = (refine_mode == RefineMode::kOff)      ? RefineMode::kSubpix
                      : (refine_mode == RefineMode::kSubpix) ? RefineMode::kLines
                                                             : RefineMode::kOff;
        log_gui("p", "亚像素精修", refine_name(refine_mode));
      }
    }
  }

  pred_log.close();
  det_log.close();
  devices.close();
}
