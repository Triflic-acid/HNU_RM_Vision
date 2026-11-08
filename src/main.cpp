#include "lessons.hpp"
#include "io_context.hpp"

#include "debug_flags.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <csignal>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <optional>

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

  // 圆心时序滤波：在【原图坐标】里做，避免与 ROI 裁剪坐标混用。
  // 连续 kEscape 次都判为"跳变"就认为旧值错了，接受新值（否则会永久锁死）。
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
        if (last.x >= 0.F && cv::norm(*in - last) > limit)
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
      if (last.x >= 0.F && ++lost <= kLostMax)
        return last;
      last = cv::Point2f(-1.F, -1.F);
      lost = 0;
      return std::nullopt;
    }

    void reset()
    {
      last = cv::Point2f(-1.F, -1.F);
      lost = 0;
      rejects = 0;
    }
  };

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
}



namespace training::dbg
{
  bool io{true};          // [IO] 帧率/姿态、[read] nullopt、[Write]
  bool det{false};        // [det] 每帧检测耗时、模型/模板加载
  bool rune{true};       // [rune] 圆心 r/w、模板分数
  bool gui{true};         // [gui] 按键切换提示
  bool demo_write{false}; // 演示：无视算法输出，向云台发出按正弦函数运动的指令
} // namespace training::dbg



int main(int argc, char **argv)
{
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  const auto config = training::load_config(argc > 1 ? argv[1] : "configs/simulation.yaml");
  const auto calibration = training::load_calibration(config);
  training::io::IoContext devices{config};
  training::TrackingState tracking;
  devices.open();

  training::DetMode mode{training::DetMode::kArmor}; // 默认自瞄；按 m 切换

  RoiMode roi_mode = RoiMode::kOff;
  const float manual_zoom = 2.0F;
  RefineMode refine_mode = RefineMode::kOff;
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

      // 为ROI预留
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
        const cv::Point2f c = (tracking.last_center.x >= 0.F)
                                  ? tracking.last_center
                                  : cv::Point2f(input->image.cols * 0.5F, input->image.rows * 0.5F);
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

      // 能量机关圆心：在【原图坐标】里做时序滤波。
      // lesson03 只负责单帧；若把时序放进 lesson03，ROI 裁剪坐标会和原图坐标混用，
      // 表现为圆心被锁到画面角落再也回不来。
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

      // 写回
      // 状态只由 lesson4 更新
      // main 不直接写入 TrackingState 字段
      training::lesson04_observe(tracking, detections);

      // lesson 4/5 目标跟踪/击打
      const auto target = training::lesson04_tracking(
          detections, mode, input->timestamp, input->imu_q, calibration, tracking);
      if (target)
        command = training::lesson05_aiming(*target, input->timestamp, input->bullet_speed);

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
          draw_armors(canvas, detections);
        else
          draw_runes(canvas, detections);
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
                      "[%s] %zu targets  %.1fms  ROI:%s  refine:%s  m:mode z:roi p:refine",
                      mode == training::DetMode::kArmor ? "ARMOR" : "RUNE",
                      detections.size(), last_ms, roi_txt, refine_name(refine_mode));
        cv::putText(canvas, hud, cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    mode == training::DetMode::kArmor ? cv::Scalar(0, 255, 0)
                                                      : cv::Scalar(0, 128, 255),
                    2);
      }

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
      }
      else if (key == 'z' || key == 'Z')
      {
        roi_mode = (roi_mode == RoiMode::kOff)    ? RoiMode::kAuto
                   : (roi_mode == RoiMode::kAuto) ? RoiMode::kManual
                                                  : RoiMode::kOff;
        std::printf("[gui] ROI: %s\n",
                    roi_mode == RoiMode::kAuto     ? "自动(PnP)"
                    : roi_mode == RoiMode::kManual ? "手动 2.0x"
                                                   : "关闭");
        std::fflush(stdout);
      }
      else if (key == 'p' || key == 'P')
      {
        refine_mode = (refine_mode == RefineMode::kOff)      ? RefineMode::kSubpix
                      : (refine_mode == RefineMode::kSubpix) ? RefineMode::kLines
                                                             : RefineMode::kOff;
        std::printf("[gui] 精修: %s\n", refine_name(refine_mode));
        std::fflush(stdout);
      }
    }
  }

  devices.close();
}
