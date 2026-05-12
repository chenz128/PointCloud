"""
2Hz PCD → 10Hz PCD 升频插值主流程

输出目录结构：
  data/output/series_A/   — 系列 A 原始帧 + 插值中间帧
  data/output/series_B/   — 系列 B 原始帧 + 插值中间帧
"""

import os
import re
import shutil
import json
import numpy as np

from pcd_reader import read_pcd
from pcd_writer import write_pcd
from transform import load_ego_poses, transform_points


# ── 路径配置 ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'point_cloud_segmentation')
EGO_POSE_PATH = os.path.join(BASE_DIR, 'data', 'ego_pose.json')
OUT_A = os.path.join(BASE_DIR, 'data', 'output', 'series_A')
OUT_B = os.path.join(BASE_DIR, 'data', 'output', 'series_B')


def scan_series(input_dir):
    """
    扫描 input_dir，将 PCD 文件分成系列 A（step-1）和系列 B（step-5）。

    区分方法：有配套 .bin 文件的是系列 B，否则是系列 A。
    系列 B：编号直接等于 ego_pose sample_idx。
    系列 A：编号是顺序计数，sample_idx = 编号 × 5。

    返回:
        series_a: list of (sample_idx, timestamp_ns, filepath)，按 sample_idx 排序
        series_b: list of (sample_idx, timestamp_ns, filepath)，按 sample_idx 排序
    """
    pattern = re.compile(r'^(\d{4})_(\d+)\.pcd$')

    series_a = []
    series_b = []

    for fname in os.listdir(input_dir):
        m = pattern.match(fname)
        if not m:
            continue
        idx = int(m.group(1))
        ts = int(m.group(2))
        fpath = os.path.join(input_dir, fname)

        # 判断：有同名 .bin 文件 → 系列 B
        bin_fname = fname.replace('.pcd', '.bin')
        has_bin = os.path.exists(os.path.join(input_dir, bin_fname))

        if has_bin:
            # 系列 B：编号即 sample_idx
            series_b.append((idx, ts, fpath))
        else:
            # 系列 A：sample_idx = 编号 × 5
            series_a.append((idx * 5, ts, fpath))

    series_a.sort(key=lambda x: x[0])
    series_b.sort(key=lambda x: x[0])

    return series_a, series_b


def interpolate_series(frames, poses, out_dir, series_name):
    """
    对一个系列的真值帧列表做插值，生成中间帧并写出 PCD。

    参数:
        frames: list of (sample_idx, timestamp_ns, filepath)
        poses:  dict[timestamp_ns, (4,4) ndarray]
        out_dir: 输出目录
        series_name: 日志用，'A' 或 'B'
    """
    os.makedirs(out_dir, exist_ok=True)

    # 复制原始真值帧到输出目录
    for sample_idx, ts, fpath in frames:
        dst = os.path.join(out_dir, os.path.basename(fpath))
        if not os.path.exists(dst):
            shutil.copy2(fpath, dst)

    total_pairs = len(frames) - 1
    generated = 0

    for i in range(total_pairs):
        idx_a, ts_a, path_a = frames[i]
        idx_b, ts_b, path_b = frames[i + 1]

        # 两帧之间应有 4 个中间 sample_idx
        if idx_b - idx_a != 5:
            print(f"[{series_name}] 跳过非连续帧对: sample_idx {idx_a} → {idx_b}")
            continue

        if ts_a not in poses or ts_b not in poses:
            print(f"[{series_name}] ego_pose 缺失: ts_a={ts_a} or ts_b={ts_b}")
            continue

        T_a = poses[ts_a]
        T_b = poses[ts_b]

        pts_a = read_pcd(path_a)  # (N, 5)
        pts_b = read_pcd(path_b)  # (N, 5)

        xyz_a = pts_a[:, :3]
        xyz_b = pts_b[:, :3]

        # 4 个中间帧：sample_idx = idx_a+1, idx_a+2, idx_a+3, idx_a+4
        for step in range(1, 5):
            mid_idx = idx_a + step
            # 时间戳：在 ts_a 和 ts_b 之间均匀插值
            mid_ts = ts_a + (ts_b - ts_a) * step // 5

            if mid_ts not in poses:
                print(f"[{series_name}] ego_pose 缺失中间帧: sample_idx={mid_idx} ts={mid_ts}")
                continue

            T_t = poses[mid_ts]

            # 帧 A 运动补偿到 t
            xyz_a_warp = transform_points(xyz_a, T_a, T_t)
            pts_a_warp = np.hstack([
                xyz_a_warp.astype(np.float32),
                pts_a[:, 3:5]  # label + intensity 原样保留
            ])

            # 帧 B 运动补偿到 t
            xyz_b_warp = transform_points(xyz_b, T_b, T_t)
            pts_b_warp = np.hstack([
                xyz_b_warp.astype(np.float32),
                pts_b[:, 3:5]
            ])

            # 合并
            merged = np.vstack([pts_a_warp, pts_b_warp])

            out_fname = f"{mid_idx:04d}_{mid_ts}.pcd"
            out_path = os.path.join(out_dir, out_fname)
            write_pcd(out_path, merged)
            generated += 1

        if (i + 1) % 10 == 0 or (i + 1) == total_pairs:
            print(f"[{series_name}] 进度: {i+1}/{total_pairs} 帧对完成")

    print(f"[{series_name}] 完成：共生成 {generated} 个中间帧，写入 {out_dir}")


def main():
    print("加载 ego_pose ...")
    poses = load_ego_poses(EGO_POSE_PATH)
    print(f"  已加载 {len(poses)} 条位姿记录")

    print("扫描数据目录 ...")
    series_a, series_b = scan_series(INPUT_DIR)
    print(f"  系列 A: {len(series_a)} 帧  系列 B: {len(series_b)} 帧")

    print("\n── 处理系列 A ──")
    interpolate_series(series_a, poses, OUT_A, 'A')

    print("\n── 处理系列 B ──")
    interpolate_series(series_b, poses, OUT_B, 'B')

    print("\n全部完成。")


if __name__ == '__main__':
    main()
