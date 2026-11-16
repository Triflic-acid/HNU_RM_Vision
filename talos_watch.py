#!/usr/bin/env python3
"""实时读取仿真器（talos_ipc）共享内存：云台位姿 / 真值 / 底盘状态。

用法：
  python3 talos_watch.py                 # 实时刷屏
  python3 talos_watch.py --csv out.csv   # 同时写 CSV
  python3 talos_watch.py --once          # 只打一帧
  python3 talos_watch.py --hz 5          # 刷新频率

坐标系（重要）：
  Bevy(Y 上, 右手) → ROS(Z 上) 的换算矩阵 M_ALIGN 的列是
      (0,−1,0) (0,0,1) (−1,0,0)
  即  ROS_x = −bevy_z,  ROS_y = −bevy_x,  ROS_z = bevy_y
  所以靶车 yaw=90°(π/2) 是**正常的右手系朝向**，不是程序算错。
  另外发布"云台四元数"时还额外乘了 Rz(90°)：
      gimbal_rot * muzzle_local.rot * Rz(π/2)
  （见 src/talos/capture.rs::captured_pose_data）

pose 索引的含义（很容易弄错）：
  0 Gimbal : 位置恒为 (0,0,0)，只有**姿态**
  1 Odom   : 云台在**世界系的位置**（四元数为单位）
  2 Muzzle : 枪口**相对云台**的位置   ← 安装偏移
  3 Camera : 相机**相对云台**的位置   ← 安装偏移

布局来源：simulation/simulator/crates/talos-ipc/src/layout.rs
  /tmp/talos_ipc_meta   3712 B
    0     ShmHeader            64
    64    ImageTripleBuffer    192
    256   PoseTripleBuffer[5]  1280   (0=gimbal 1=odom 2=muzzle 3=camera 4=chassis)
    1536  GimbalTripleBuffer   192
    1728  CameraInfo           128
    1856  ChassisObservation   128
    1984  GroundTruthBatch     1664
    3648  RuntimeState         64
"""
from __future__ import annotations

import argparse
import csv
import math
import mmap
import os
import struct
import sys
import time

SHM_CANDIDATES = [
    "/tmp/talos_ipc_meta",
    "/dev/shm/talos_ipc_meta",
]
SHM_PATH = SHM_CANDIDATES[0]
SHM_SIZE = 3712
SHM_MAGIC = 0x54414C05

POSES_OFF = 256
POSE_STRIDE = 256          # 一个 PoseTripleBuffer = 256 B
POSE_SLOT_OFF = 64         # 槽位数组起点
POSE_SLOT_SIZE = 64
GIMBAL_OFF = 1536
CAMERA_INFO_OFF = 1728
CHASSIS_OFF = 1856
GT_OFF = 1984
RUNTIME_OFF = 3648

POSE_NAMES = ["gimbal", "odom", "muzzle", "camera", "chassis"]


def _f32(buf, off, n=1):
    return struct.unpack_from("<" + "f" * n, buf, off)


def _u64(buf, off):
    return struct.unpack_from("<Q", buf, off)[0]


def _u32(buf, off):
    return struct.unpack_from("<I", buf, off)[0]


def quat_to_rpy(q):
    """(x, y, z, w) → (roll, pitch, yaw)，ZYX 顺序，单位 rad。"""
    x, y, z, w = q
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def quat_rotate(q, v):
    """用四元数 (x,y,z,w) 旋转向量 v。"""
    x, y, z, w = q
    vx, vy, vz = v
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


class TalosShm:
    def __init__(self, path=SHM_PATH):
        self.path = path
        self.fd = None
        self.mm = None

    def open(self):
        if not os.path.exists(self.path):
            return False
        self.fd = os.open(self.path, os.O_RDONLY)
        size = max(os.fstat(self.fd).st_size, SHM_SIZE)
        self.mm = mmap.mmap(self.fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
        return True

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    # ---------- 基础段 ----------
    def header(self):
        magic = _u32(self.mm, 0)
        version = _u32(self.mm, 4)
        created = _u64(self.mm, 8)
        heartbeat = _u64(self.mm, 16)
        return magic, version, created, heartbeat

    def pose(self, idx):
        base = POSES_OFF + idx * POSE_STRIDE
        state = self.mm[base]
        write_idx = self.mm[base + 1] & 0x03
        slot = base + POSE_SLOT_OFF + write_idx * POSE_SLOT_SIZE
        seq = _u64(self.mm, slot)
        pos = _f32(self.mm, slot + 8, 3)
        quat = _f32(self.mm, slot + 20, 4)
        ts = _u64(self.mm, slot + 36)
        return {"seq": seq, "pos": pos, "quat": quat, "ts": ts,
                "state": state, "slot": write_idx}

    def gimbal_cmd(self):
        state = self.mm[GIMBAL_OFF]
        write_idx = self.mm[GIMBAL_OFF + 1] & 0x03
        slot = GIMBAL_OFF + 64 + write_idx * 32
        ts = _u64(self.mm, slot)
        yaw, pitch, dist = _f32(self.mm, slot + 8, 3)
        fire = self.mm[slot + 20]
        return {"ts": ts, "yaw_deg": yaw, "pitch_deg": pitch,
                "distance_m": dist, "fire": fire, "state": state}

    def camera_info(self):
        o = CAMERA_INFO_OFF
        fx, fy, cx, cy = struct.unpack_from("<4d", self.mm, o + 8)
        dist = struct.unpack_from("<5d", self.mm, o + 40)
        w, h = struct.unpack_from("<2I", self.mm, o + 80)
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "distortion": dist, "width": w, "height": h}

    def chassis(self):
        o = CHASSIS_OFF
        seq = _u64(self.mm, o)
        ts = _u64(self.mm, o + 8)
        dt = _f32(self.mm, o + 16)[0]
        v_body = _f32(self.mm, o + 20, 2)
        wz = _f32(self.mm, o + 28)[0]
        rpy = _f32(self.mm, o + 76, 3)
        gyro = _f32(self.mm, o + 88, 3)
        accel = _f32(self.mm, o + 100, 3)
        return {"seq": seq, "ts": ts, "dt": dt, "v_body": v_body, "wz": wz,
                "rpy": rpy, "gyro": gyro, "accel": accel}

    def ground_truth(self):
        o = GT_OFF
        seq = _u64(self.mm, o)
        ts = _u64(self.mm, o + 8)
        n_t = _u32(self.mm, o + 16)
        n_r = _u32(self.mm, o + 20)
        targets = []
        for i in range(min(n_t, 16)):
            b = o + 32 + i * 64
            targets.append({
                "seq": _u64(self.mm, b),
                "ts": _u64(self.mm, b + 8),
                "team": self.mm[b + 16],
                "label": self.mm[b + 17],
                "outpost": self.mm[b + 18],
                "pos": _f32(self.mm, b + 20, 3),
                "vyaw": _f32(self.mm, b + 32)[0],
                "yaw": _f32(self.mm, b + 36)[0],
            })
        runes = []
        for i in range(min(n_r, 4)):
            b = o + 1088 + i * 128
            runes.append({
                "team": self.mm[b + 16],
                "mode": self.mm[b + 17],
                "state": self.mm[b + 18],
                "center": _f32(self.mm, b + 20, 3),
                "radius": _f32(self.mm, b + 32)[0],
                "angle": _f32(self.mm, b + 36)[0],
                "v_roll": _f32(self.mm, b + 40)[0],
                "blade": struct.unpack_from("<i", self.mm, b + 68)[0],
            })
        return {"seq": seq, "ts": ts, "n_targets": n_t, "n_runes": n_r,
                "targets": targets, "runes": runes}

    def runtime(self):
        return {"ts": _u64(self.mm, RUNTIME_OFF), "following": self.mm[RUNTIME_OFF + 8]}


# ---------------- 统计与显示 ----------------
class Stat:
    """记录一段时间内的位姿，用于显示抖动。"""

    def __init__(self, n=200):
        self.buf = []
        self.n = n

    def push(self, v):
        self.buf.append(v)
        if len(self.buf) > self.n:
            self.buf.pop(0)

    def std_mm(self):
        if len(self.buf) < 5:
            return float("nan")
        m = [sum(c) / len(self.buf) for c in zip(*self.buf)]
        s = sum((x - m[0]) ** 2 + (y - m[1]) ** 2 + (z - m[2]) ** 2
                for x, y, z in self.buf) / len(self.buf)
        return math.sqrt(s) * 1000.0


def fmt_pos(p):
    return f"({p[0]:+7.3f},{p[1]:+7.3f},{p[2]:+7.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="把每帧写到 CSV")
    ap.add_argument("--once", action="store_true", help="只打印一帧")
    ap.add_argument("--hz", type=float, default=5.0, help="刷新频率（Hz）")
    ap.add_argument("--path", default=None,
                    help="共享内存路径（默认自动在 /tmp 与 /dev/shm 里找）")
    ap.add_argument("--wait", type=float, default=15.0,
                    help="等共享内存出现的最长秒数（默认 15）")
    args = ap.parse_args()

    # 选路径：显式指定优先，否则在两个候选里找
    path = args.path
    if path is None:
        for c in SHM_CANDIDATES:
            if os.path.exists(c):
                path = c
                break
        if path is None:
            path = SHM_CANDIDATES[0]

    # 等仿真器把共享内存建出来
    deadline = time.time() + max(args.wait, 0.0)
    waited = False
    while not os.path.exists(path) and time.time() < deadline:
        if not waited:
            print(f"[等待] 还没出现 {path}，最多等 {args.wait:.0f} s"
                  f"（仿真器启动后会自动继续）…")
            waited = True
        time.sleep(0.2)

    shm = TalosShm(path)
    if not shm.open():
        print(f"[错误] 找不到 {path}")
        for c in SHM_CANDIDATES:
            print(f"       已检查：{c}{'  ← 存在' if os.path.exists(c) else ''}")
        print("       先启动仿真器：./scripts/run_simulator.sh，再运行本脚本")
        return 1
    print(f"[信息] 使用 {path}")

    magic, version, created, _ = shm.header()
    if magic != SHM_MAGIC:
        print(f"[警告] magic=0x{magic:08X}，期望 0x{SHM_MAGIC:08X}（版本可能不同）")
    print(f"[信息] magic=0x{magic:08X} version={version} 已映射 {SHM_SIZE} B")

    csv_f = csv_w = None
    if args.csv:
        csv_f = open(args.csv, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow([
            "t_ms", "gz_x", "gz_y", "gz_z", "gyaw", "gpitch", "groll",
            "mz_x", "mz_y", "mz_z", "cam_x", "cam_y", "cam_z",
            "nt", "t0_label", "t0_x", "t0_y", "t0_z", "t0_yaw",
            "dist_g2t", "yaw_g2t_deg", "pitch_g2t_deg",
            "chassis_vx", "chassis_vy", "chassis_wz",
            "std_mm",
        ])

    stat = Stat()
    period = 1.0 / max(args.hz, 0.5)
    t0 = time.time()
    last_print = 0.0
    try:
        while True:
            now = time.time()
            odom = shm.pose(1)      # 云台在世界系的位置（identity 姿态）
            gq = shm.pose(0)        # 云台姿态（位置恒为 0）
            mz = shm.pose(2)        # 枪口相对云台
            cam = shm.pose(3)       # 相机相对云台
            g = {"pos": odom["pos"], "quat": gq["quat"], "seq": gq["seq"]}
            gt = shm.ground_truth()
            ch = shm.chassis()
            gcmd = shm.gimbal_cmd()

            stat.push(g["pos"])
            std_mm = stat.std_mm()

            # 云台 → 目标（世界系方位角/俯仰角，供和程序里的 log 对照）
            yaw_d = pitch_d = dist = float("nan")
            t0t = gt["targets"][0] if gt["targets"] else None
            if t0t:
                d = [t0t["pos"][i] - g["pos"][i] for i in range(3)]
                dist = math.sqrt(sum(x * x for x in d))
                yaw_d = math.degrees(math.atan2(d[1], d[0]))
                pitch_d = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))

            if csv_w:
                csv_w.writerow([
                    f"{(now - t0) * 1000:.0f}",
                    *(f"{v:.4f}" for v in g["pos"]),
                    *(f"{math.degrees(v):.3f}" for v in quat_to_rpy(g["quat"])),
                    *(f"{v:.4f}" for v in mz["pos"]),
                    *(f"{v:.4f}" for v in cam["pos"]),
                    gt["n_targets"],
                    t0t["label"] if t0t else -1,
                    *(f"{v:.4f}" for v in (t0t["pos"] if t0t else (0, 0, 0))),
                    f"{t0t['yaw'] if t0t else 0:.4f}",
                    f"{dist:.4f}", f"{yaw_d:.3f}", f"{pitch_d:.3f}",
                    f"{ch['v_body'][0]:.4f}", f"{ch['v_body'][1]:.4f}", f"{ch['wz']:.4f}",
                    f"{std_mm:.1f}",
                ])

            if now - last_print >= period:
                last_print = now
                roll, pitch, yaw = quat_to_rpy(g["quat"])
                lines = []
                lines.append(f"── {time.strftime('%H:%M:%S')} ─────────────────────────────")
                lines.append(
                    f"云台(世界) pos={fmt_pos(g['pos'])} "
                    f"yaw={math.degrees(yaw):+7.2f}° pitch={math.degrees(pitch):+6.2f}° "
                    f"roll={math.degrees(roll):+6.2f}°")
                lines.append(
                    f"枪口相对云台 {fmt_pos(mz['pos'])}   相机相对云台 {fmt_pos(cam['pos'])}"
                    f"   std={std_mm:5.1f} mm")
                lines.append(
                    f"底盘 v=({ch['v_body'][0]:+.3f},{ch['v_body'][1]:+.3f}) m/s "
                    f"wz={ch['wz']:+.3f} rad/s  rpy="
                    f"({math.degrees(ch['rpy'][0]):+.1f},{math.degrees(ch['rpy'][1]):+.1f},"
                    f"{math.degrees(ch['rpy'][2]):+.1f})°")
                lines.append(
                    f"上位机下发  yaw={gcmd['yaw_deg']:+7.2f}° pitch={gcmd['pitch_deg']:+6.2f}° "
                    f"dist={gcmd['distance_m']:.2f} m fire={gcmd['fire']}")
                lines.append(f"真值：targets={gt['n_targets']} runes={gt['n_runes']}")
                for i, t in enumerate(gt["targets"][:6]):
                    lines.append(
                        f"  [{i}] label={t['label']:2d} team={t['team']} "
                        f"pos={fmt_pos(t['pos'])} yaw={math.degrees(t['yaw']):+7.1f}° "
                        f"vyaw={t['vyaw']:+.2f}")
                if t0t:
                    lines.append(
                        f"  云台→[0] 距离={dist:.2f} m 方位={yaw_d:+.1f}° 俯仰={pitch_d:+.1f}°")
                for i, r in enumerate(gt["runes"]):
                    lines.append(
                        f"  符文[{i}] state={r['state']} r={r['radius']:.3f} "
                        f"angle={math.degrees(r['angle']):+.1f}° blade={r['blade']}")

                if not args.once:
                    sys.stdout.write("\033[H\033[J")   # 清屏
                print("\n".join(lines))
                sys.stdout.flush()

            if args.once:
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[信息] 已停止")
    finally:
        if csv_f:
            csv_f.close()
            print(f"[信息] CSV 已写入 {args.csv}")
        shm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
