#include "lessons.hpp"
#include "io_context.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <csignal>
#include <vector>
#include <cmath>

namespace
{
  volatile std::sig_atomic_t interrupted = 0;
  void on_signal(int) { interrupted = 1; }
  // 画检测框：四边 + 四角 + 类别&置信度
  void draw_armors(cv::Mat &canvas, const std::vector<training::Armor> &armors)
  {
    for (const auto &a : armors)
    {
      for (int i = 0; i < 4; i++)
        cv::line(canvas, a.corners[i], a.corners[(i + 1) % 4],
                 cv::Scalar(0, 255, 0), 2);
      for (const auto &p : a.corners)
        cv::circle(canvas, cv::Point(cvRound(p.x), cvRound(p.y)), 3,
                   cv::Scalar(0, 0, 255), -1);
      char text[64];
      std::snprintf(text, sizeof(text), "id=%d %.2f", a.class_id, a.confidence);
      cv::putText(canvas, text,
                  cv::Point(cvRound(a.corners[0].x), cvRound(a.corners[0].y) - 8),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
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

  while (!interrupted)
  {
    const auto input = devices.read();

    training::Command command{};
    std::vector<training::Armor> armors;

    if (input)
    {
      armors = training::lesson03_detection(input->image);
      const auto target = training::lesson04_tracking(
          armors, input->timestamp, input->imu_q, calibration, tracking);
      if (target)
        command = training::lesson05_aiming(*target, input->timestamp, input->bullet_speed);
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
        draw_armors(canvas, armors);
      }
      else
      {
        canvas = cv::Mat(240, 640, CV_8UC3, cv::Scalar(30, 30, 30));
        cv::putText(canvas, "waiting for /image_raw ...", cv::Point(20, 130),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 200, 255), 2);
      }

      cv::imshow("RoboMaster training", canvas);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q')
        break;
    }
  }

  devices.close();
}
