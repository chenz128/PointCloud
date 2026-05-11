"""主流程：遍历无标注帧，生成 BIN 标签文件"""

import os
import sys
import glob

from pcd_reader import read_pcd
from bin_io import write_bin
from transform import load_ego_poses
from label_propagation import propagate_labels


# 数据目录配置
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PCD_DIR = os.path.join(DATA_DIR, 'point_cloud_segmentation')
EGO_POSE_PATH = os.path.join(DATA_DIR, 'ego_pose.json')

# 标注帧文件名（不含扩展名）
LABELED_FRAMES = [
    '0015_1775728137200007680',
    '0020_1775728137700007680',
]

# 无标注帧文件名（不含扩展名）
UNLABELED_FRAMES = [
    '0015_1775728143200007680',
    '0016_1775728143700007680',
    '0017_1775728144200007680',
    '0018_1775728144700007680',
    '0019_1775728145200007680',
    '0020_1775728145700007680',
    '0021_1775728146200007680',
    '0022_1775728146700007680',
    '0023_1775728147200007680',
    '0024_1775728147700007680',
]

# 距离阈值（米）
DISTANCE_THRESHOLD = 0.5


def main():
    print("加载 ego_pose...")
    ego_poses = load_ego_poses(EGO_POSE_PATH)
    print(f"  共 {len(ego_poses)} 条位姿记录")

    # 构建标注帧信息
    labeled_frames = []
    for name in LABELED_FRAMES:
        ts = int(name.split('_')[1])
        frame = {
            'pcd_path': os.path.join(PCD_DIR, name + '.pcd'),
            'bin_path': os.path.join(PCD_DIR, name + '.bin'),
            'timestamp': ts,
        }
        # 验证文件存在
        assert os.path.exists(frame['pcd_path']), f"标注帧 PCD 不存在: {frame['pcd_path']}"
        assert os.path.exists(frame['bin_path']), f"标注帧 BIN 不存在: {frame['bin_path']}"
        assert ts in ego_poses, f"标注帧时间戳 {ts} 不在 ego_pose 中"
        labeled_frames.append(frame)
        print(f"  标注帧: {name} (sample_idx 对应 ego_pose)")

    # 预加载标注帧点云和标签（避免每帧重复加载）
    print("\n加载标注帧点云...")
    for frame in labeled_frames:
        pts = read_pcd(frame['pcd_path'])
        labels = read_pcd(frame['pcd_path'])  # 仅验证读取
        print(f"  {os.path.basename(frame['pcd_path'])}: {pts.shape[0]} 点")

    # 遍历无标注帧
    print(f"\n开始处理 {len(UNLABELED_FRAMES)} 个无标注帧 (距离阈值={DISTANCE_THRESHOLD}m)...")
    for i, name in enumerate(UNLABELED_FRAMES):
        pcd_path = os.path.join(PCD_DIR, name + '.pcd')
        bin_path = os.path.join(PCD_DIR, name + '.bin')

        assert os.path.exists(pcd_path), f"无标注帧 PCD 不存在: {pcd_path}"

        ts = int(name.split('_')[1])
        assert ts in ego_poses, f"无标注帧时间戳 {ts} 不在 ego_pose 中"

        print(f"\n[{i+1}/{len(UNLABELED_FRAMES)}] 处理 {name}...")

        # 标签传播
        labels = propagate_labels(
            target_pcd_path=pcd_path,
            labeled_frames=labeled_frames,
            ego_poses=ego_poses,
            distance_threshold=DISTANCE_THRESHOLD,
        )

        # 验证点数匹配
        target_points = read_pcd(pcd_path)
        assert len(labels) == target_points.shape[0], \
            f"标签数 {len(labels)} != 点数 {target_points.shape[0]}"

        # 保存
        write_bin(labels, bin_path)
        print(f"  输出: {os.path.basename(bin_path)} ({len(labels)} 标签)")

        # 统计类别分布
        unique, counts = __import__('numpy').unique(labels, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"  类别分布: {dist}")

    print("\n全部完成！")


if __name__ == '__main__':
    main()
