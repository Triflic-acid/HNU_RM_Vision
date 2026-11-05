// 第三讲：装甲板检测（YOLO26-Pose + ONNX Runtime）
//
// 由于 ROS2 Humble 的 OpenCV 4.5.4 太旧，无法解析 ONNX，故不使用 cv::dnn:
// （解析 DFL 解码部分的 Add 节点时报 "blob_0.size == blob_1.size"），
// 换 opset 10/11/12/13、开不开 simplify 都不行；OpenCV 5.0 才支持。
// ONNX Runtime 的 C++ API 没有这个限制，而且推理更快。

#include "lessons.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace training
{
  namespace
  {

    constexpr int kInput{640};
    constexpr float kConfThres{0.25F};
    constexpr float kNmsThres{0.45F};
    constexpr bool kDetVerbose{false};
    constexpr const char *kModelPath{"models/best.onnx"};

    constexpr const char *kClassNames[] = {
        "B1", "B2", "B3", "B4", "B5", "BO", "BS",
        "R1", "R2", "R3", "R4", "R5", "RO", "RS"};
    constexpr int kNumClasses{static_cast<int>(sizeof(kClassNames) / sizeof(kClassNames[0]))};

    struct ClassInfo
    {
      int id;
      bool large;
    };

    // 编号 / 大小板 映射
    //   id   : 装甲板编号（1~5），0=哨兵，6=前哨站，-1=未知
    //   large: 大板 225x125 / 小板 135x125
    //          据数据集标签实测宽高比（B1/R1=1.84/1.75，其余 ~1.1~1.3），
    //          只有 1 号英雄是大板。影响第四讲 solvePnP 的物理尺寸。
    constexpr ClassInfo kClassTable[kNumClasses] = {
        {1, true},  // B1 英雄（大板）
        {2, false}, // B2 工程
        {3, false}, // B3 步兵
        {4, false}, // B4 步兵
        {5, false}, // B5 步兵
        {6, false}, // BO 前哨站
        {0, false}, // BS 哨兵
        {1, true},  // R1 英雄（大板）
        {2, false}, // R2 工程
        {3, false}, // R3 步兵
        {4, false}, // R4 步兵
        {5, false}, // R5 步兵
        {6, false}, // RO 前哨站
        {0, false}, // RS 哨兵
    };

    cv::Mat letterbox(const cv::Mat &src, int size, float &scale, int &pad_x, int &pad_y)
    {
      const int width{src.cols}, height{src.rows};
      scale = std::min(size / static_cast<float>(width), size / static_cast<float>(height));

      const int new_width{static_cast<int>(std::round(width * scale))};
      const int new_height{static_cast<int>(std::round(height * scale))};

      pad_x = (size - new_width) / 2;
      pad_y = (size - new_height) / 2;

      cv::Mat out(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
      cv::Mat resized;
      cv::resize(src, resized, cv::Size(new_width, new_height));
      resized.copyTo(out(cv::Rect(pad_x, pad_y, new_width, new_height)));
      return out;
    }

    Ort::Session *session_ptr()
    {
      static std::unique_ptr<Ort::Env> env;
      static std::unique_ptr<Ort::Session> session;
      static bool failed = false;

      if (!session && !failed)
      {
        try
        {
          env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rm_det");
          Ort::SessionOptions opts;
          opts.SetIntraOpNumThreads(4);
          opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
          session = std::make_unique<Ort::Session>(*env, kModelPath, opts);
        }
        catch (const std::exception &e)
        {
          // 模型缺失/损坏时不要让进程直接 terminate
          std::printf("[det] 模型加载失败：%s\n[det] 请检查 %s（应在仓库根目录运行）\n",
                      e.what(), kModelPath);
          std::fflush(stdout);
          failed = true;
        }
      }
      return session.get(); // 失败时为 nullptr
    }

  } // namespace

  std::vector<Armor> lesson03_detection(const cv::Mat &image)
  {
    std::vector<Armor> armors;
    if (image.empty() || image.type() != CV_8UC3)
      return armors;

    const auto t0 = Clock::now();

    // ---- 1. 预处理 ----
    float scale{1.F};
    int pad_x{0}, pad_y{0};
    const cv::Mat padded = letterbox(image, kInput, scale, pad_x, pad_y);

    // HWC BGR uint8 -> CHW RGB float 0~1
    // static：复用同一块 4.9 MB 缓冲，避免每帧 malloc/free（单线程安全）
    static std::vector<float> input(3 * kInput * kInput);
    const int plane = kInput * kInput;
    for (int y = 0; y < kInput; ++y)
    {
      const auto *row = padded.ptr<cv::Vec3b>(y);
      for (int x = 0; x < kInput; ++x)
      {
        const auto &bgr = row[x];
        const int idx = y * kInput + x;
        input[0 * plane + idx] = bgr[2] / 255.F; // R
        input[1 * plane + idx] = bgr[1] / 255.F; // G
        input[2 * plane + idx] = bgr[0] / 255.F; // B
      }
    }

    // ---- 2. 推理 ----
    auto *session = session_ptr();
    if (!session)
      return armors;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> in_shape{1, 3, kInput, kInput};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        mem, input.data(), input.size(), in_shape.data(), in_shape.size());

    const char *in_names[] = {"images"};
    const char *out_names[] = {"output0"};

    auto outputs = session->Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);

    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape(); // [1, C, N]
    if (shape.size() != 3 || shape[0] != 1)
      return armors;

    const int C{static_cast<int>(shape[1])};
    const int N{static_cast<int>(shape[2])};
    const int nc{C - 4 - 8}; // 4 框 + 4 关键点×2
    if (nc <= 0 || nc > kNumClasses)
      return armors;

    const float *p = outputs[0].GetTensorData<float>();

    // ---- 3. 解码 ----
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::array<cv::Point2f, 4>> kpts;

    for (int i = 0; i < N; ++i)
    {
      int best_class{0};
      float best_score{0.F};
      for (int c = 0; c < nc; ++c)
      {
        const float s = p[(4 + c) * N + i];
        if (s > best_score)
        {
          best_score = s;
          best_class = c;
        }
      }
      if (best_score < kConfThres)
        continue;

      const float cx = p[0 * N + i];
      const float cy = p[1 * N + i];
      const float w = p[2 * N + i];
      const float h = p[3 * N + i];

      std::array<cv::Point2f, 4> kp{};
      for (int k = 0; k < 4; ++k)
      {
        const float x = p[(4 + nc + k * 2 + 0) * N + i];
        const float y = p[(4 + nc + k * 2 + 1) * N + i];
        kp[k] = cv::Point2f((x - pad_x) / scale, (y - pad_y) / scale);
      }

      const float x0 = (cx - w / 2.F - pad_x) / scale;
      const float y0 = (cy - h / 2.F - pad_y) / scale;
      const float bw = w / scale;
      const float bh = h / scale;
      if (bw < 2.F || bh < 2.F)
        continue;

      boxes.emplace_back(static_cast<int>(x0), static_cast<int>(y0),
                         static_cast<int>(bw), static_cast<int>(bh));
      scores.push_back(best_score);
      class_ids.push_back(best_class);
      kpts.push_back(kp);
    }

    if (boxes.empty())
    {
      if (kDetVerbose)
      {
        const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        std::printf("[det] %.1f ms, 0 armors\n", ms);
      }
      return armors;
    }

    // ---- 4. NMS ----
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, kConfThres, kNmsThres, keep);

    armors.reserve(keep.size());
    for (int idx : keep)
    {
      Armor a;
      // Armor 要求：左上、右上、右下、左下
      a.corners = {kpts[idx][0], kpts[idx][3], kpts[idx][2], kpts[idx][1]}; // 注意顺序
      a.confidence = scores[idx];
      a.class_id = kClassTable[class_ids[idx]].id;
      a.large_armor = kClassTable[class_ids[idx]].large;
      armors.push_back(a);
    }

    if (kDetVerbose)
    {
      const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      std::printf("[det] %.1f ms, %zu armors\n", ms, armors.size());
      std::fflush(stdout);
    }

    return armors;
  }

} // namespace training
