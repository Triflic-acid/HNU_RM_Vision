#include "lessons.hpp"
#include "io_context.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <csignal>
#include <string>
#include <vector>
#include <cmath>

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

  constexpr bool kDemoWrite{false}; // write()函数测试开关
}

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

  while (!interrupted)
  {
    const auto input = devices.read();

    training::Command command{};
    std::vector<training::Detection> detections;
    bool has_logo = false;

    if (input)
    {
      detections = training::lesson03_detection(input->image, mode);
      const auto target = training::lesson04_tracking(
          detections, mode, input->timestamp, input->imu_q, calibration, tracking);
      if (target)
        command = training::lesson05_aiming(*target, input->timestamp, input->bullet_speed);

      if (mode == training::DetMode::kRune && !detections.empty())
        has_logo = detections.front().center.has_value();
    }
    if (interrupted)
      break;

    if (kDemoWrite)
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
      // 没有新帧时也应当刷新窗口，否则窗口会假死、按键无响应
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
        char hud[160];
        if (mode == training::DetMode::kArmor)
          std::snprintf(hud, sizeof(hud), "[MODE] ARMOR   %zu targets    (m: switch)",
                        detections.size());
        else
          std::snprintf(hud, sizeof(hud), "[MODE] RUNE    %zu blades   logo:%s    (m: switch)",
                        detections.size(), has_logo ? "Y" : "N");
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
        std::printf("[gui] 切换到 %s 模式\n",
                    mode == training::DetMode::kArmor ? "自瞄(ARMOR)" : "打符(RUNE)");
        std::fflush(stdout);
      }
    }
  }

  devices.close();
}
