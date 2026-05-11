"""将 PCD + BIN 转换为带颜色的 PLY 文件"""

import os
import sys
import struct
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pcd_reader import read_pcd
from bin_io import read_bin

# 类别 → RGB 颜色映射
LABEL_COLORS = {
    0:  (128, 128, 128),  # 背景/未知 - 灰色
    1:  (255, 0, 0),      # 红色
    2:  (0, 255, 0),      # 绿色
    3:  (0, 0, 255),      # 蓝色
    4:  (255, 255, 0),    # 黄色
    5:  (255, 0, 255),    # 品红
    6:  (0, 255, 255),    # 青色（道路）
    7:  (255, 128, 0),    # 橙色
    11: (128, 0, 255),    # 紫色
    12: (0, 128, 255),    # 天蓝
    13: (255, 128, 128),  # 浅红
    17: (128, 255, 0),    # 黄绿
    24: (0, 128, 128),    # 深青
    32: (128, 64, 0),     # 棕色
    33: (0, 128, 0),      # 深绿（植被）
}

DEFAULT_COLOR = (200, 200, 200)


def bin_to_ply(pcd_path, bin_path, ply_path):
    """将 PCD + BIN 转换为带颜色的 PLY 文件"""
    pts = read_pcd(pcd_path)
    labels = read_bin(bin_path)

    # BIN 可能比 PCD 长，取前 N 个
    labels = labels[:pts.shape[0]]

    xyz = pts[:, :3]
    N = xyz.shape[0]

    # 生成颜色
    colors = np.zeros((N, 3), dtype=np.uint8)
    for label, color in LABEL_COLORS.items():
        mask = labels == label
        colors[mask] = color
    # 未在映射中的标签用默认颜色
    unmapped = ~np.isin(labels, list(LABEL_COLORS.keys()))
    colors[unmapped] = DEFAULT_COLOR

    # 写 PLY
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
                    f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")

    print(f"  保存: {ply_path} ({N} 点)")


if __name__ == '__main__':
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    PCD_DIR = os.path.join(DATA_DIR, 'point_cloud_segmentation')
    PLY_DIR = os.path.join(DATA_DIR, 'ply')
    os.makedirs(PLY_DIR, exist_ok=True)

    frames = [
        '0015_1775728137200007680',
        '0020_1775728137700007680',
    ]

    for name in frames:
        pcd_path = os.path.join(PCD_DIR, name + '.pcd')
        bin_path = os.path.join(PCD_DIR, name + '.bin')
        ply_path = os.path.join(PLY_DIR, name + '.ply')
        print(f"转换 {name}...")
        bin_to_ply(pcd_path, bin_path, ply_path)

    print("\n完成！")
