#pragma once
// 读取仿真器真值（talos_ipc 共享内存）
//
// 布局来源：simulation/simulator/crates/talos-ipc/src/layout.rs
//   共享内存文件：/tmp/talos_ipc_meta（memmap 文件，不是 POSIX shm）
//   ShmMetaRegion 总长 3712 字节，各段偏移：
//       0     ShmHeader            64
//       64    ImageTripleBuffer   192
//       256   PoseTripleBuffer[5] 1280
//       1536  GimbalTripleBuffer  192
//       1728  CameraInfo          128
//       1856  ChassisObservation  128
//       1984  GroundTruthBatch    1664   ← 我们只要这一段
//       3648  RuntimeState         64
//
// GroundTruthBatch 内部偏移：
//       0     frame_seq(u64)  8  timestamp_ns(u64)
//       16    target_count(u32)  20 rune_count(u32)
//       32    targets[16]  每个 64 字节
//       1088  runes[4]     每个 128 字节
//
// GroundTruthTarget(64B) 内部偏移：
//       0 frame_seq(u64) | 8 timestamp_ns(u64) | 16 team(u8) | 17 armor_label(u8)
//       18 is_outpost(u8) | 20 position(f32[3]) | 32 vyaw(f32) | 36 yaw(f32)
//
// 只读映射，不写、不修改仿真器的任何数据。

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace training::truth
{
  struct Target
  {
    uint8_t team = 0;        // 0=红 1=蓝
    uint8_t armor_label = 0; // 1~5 编号，7 哨兵，0 前哨站
    uint8_t is_outpost = 0;
    float position[3]{};     // 世界系（ROS Z-up），m
    float vyaw = 0.F;
    float yaw = 0.F;
    float distance_to(const float p[3]) const
    {
      const float dx = position[0] - p[0];
      const float dy = position[1] - p[1];
      const float dz = position[2] - p[2];
      return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
  };

  struct Rune
  {
    uint8_t team = 0;
    uint8_t rune_mode = 0;
    uint8_t mechanism_state = 0;
    float r_center_odom[3]{};
    float radius = 0.F;
    float current_angle = 0.F;
    float v_roll = 0.F;
    int32_t blade_id = -1;
  };

  class ShmTruth
  {
  public:
    static constexpr const char *kPath = "/tmp/talos_ipc_meta";
    static constexpr uint32_t kMagic = 0x54414C05;
    static constexpr uint32_t kVersion = 2;
    static constexpr size_t kMetaSize = 3712;
    static constexpr size_t kGtOffset = 1984;

    ShmTruth() = default;
    ~ShmTruth() { close(); }
    ShmTruth(const ShmTruth &) = delete;
    ShmTruth &operator=(const ShmTruth &) = delete;

    // 返回 false 表示仿真器没在跑 / 版本不符
    bool open()
    {
      close();
      fd_ = ::open(kPath, O_RDONLY);
      if (fd_ < 0)
        return false;

      struct stat st {};
      if (::fstat(fd_, &st) != 0 || static_cast<size_t>(st.st_size) < kMetaSize)
      {
        ::close(fd_);
        fd_ = -1;
        return false;
      }

      base_ = static_cast<const uint8_t *>(
          ::mmap(nullptr, kMetaSize, PROT_READ, MAP_SHARED, fd_, 0));
      if (base_ == MAP_FAILED)
      {
        base_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        return false;
      }

      uint32_t magic = 0, ver = 0;
      std::memcpy(&magic, base_ + 0, 4);
      std::memcpy(&ver, base_ + 4, 4);
      if (magic != kMagic || ver != kVersion)
      {
        close();
        return false;
      }
      return true;
    }

    void close()
    {
      if (base_)
      {
        ::munmap(const_cast<uint8_t *>(base_), kMetaSize);
        base_ = nullptr;
      }
      if (fd_ >= 0)
      {
        ::close(fd_);
        fd_ = -1;
      }
    }

    bool valid() const { return base_ != nullptr; }

    // 读一帧真值
    bool poll(std::vector<Target> &targets, std::vector<Rune> &runes,
              uint64_t *frame_seq = nullptr) const
    {
      targets.clear();
      runes.clear();
      if (!base_)
        return false;

      const uint8_t *gt = base_ + kGtOffset;
      uint32_t n_targets = 0, n_runes = 0;
      std::memcpy(&n_targets, gt + 16, 4);
      std::memcpy(&n_runes, gt + 20, 4);
      if (frame_seq)
        std::memcpy(frame_seq, gt + 0, 8);

      n_targets = std::min<uint32_t>(n_targets, 16);
      n_runes = std::min<uint32_t>(n_runes, 4);

      for (uint32_t i = 0; i < n_targets; ++i)
      {
        const uint8_t *p = gt + 32 + i * 64;
        Target t;
        std::memcpy(&t.team, p + 16, 1);
        std::memcpy(&t.armor_label, p + 17, 1);
        std::memcpy(&t.is_outpost, p + 18, 1);
        std::memcpy(t.position, p + 20, sizeof(t.position));
        std::memcpy(&t.vyaw, p + 32, 4);
        std::memcpy(&t.yaw, p + 36, 4);
        targets.push_back(t);
      }

      for (uint32_t i = 0; i < n_runes; ++i)
      {
        const uint8_t *p = gt + 1088 + i * 128;
        Rune r;
        std::memcpy(&r.team, p + 16, 1);
        std::memcpy(&r.rune_mode, p + 17, 1);
        std::memcpy(&r.mechanism_state, p + 18, 1);
        std::memcpy(r.r_center_odom, p + 20, sizeof(r.r_center_odom));
        std::memcpy(&r.radius, p + 32, 4);
        std::memcpy(&r.current_angle, p + 36, 4);
        std::memcpy(&r.v_roll, p + 40, 4);
        std::memcpy(&r.blade_id, p + 44, 4);
        runes.push_back(r);
      }
      return true;
    }

  private:
    int fd_{-1};
    const uint8_t *base_{nullptr};
  };
} // namespace training::truth
