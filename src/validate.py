"""验证脚本：用标注帧互推计算准确率"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pcd_reader import read_pcd
from bin_io import read_bin
from transform import load_ego_poses, transform_points
from label_propagation import propagate_labels


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PCD_DIR = os.path.join(DATA_DIR, 'point_cloud_segmentation')
EGO_POSE_PATH = os.path.join(DATA_DIR, 'ego_pose.json')


def validate_cross_prediction():
    """用标注帧 0020 推算 0015 的标签，反之亦然"""
    print("=" * 60)
    print("验证 1: 标注帧交叉推算准确率")
    print("=" * 60)

    ego_poses = load_ego_poses(EGO_POSE_PATH)

    frames = [
        {'name': '0015_1775728137200007680', 'timestamp': 1775728137200007680},
        {'name': '0020_1775728137700007680', 'timestamp': 1775728137700007680},
    ]

    for i, target in enumerate(frames):
        # 用另一个标注帧推算当前帧的标签
        source = frames[1 - i]
        labeled_frames = [{
            'pcd_path': os.path.join(PCD_DIR, source['name'] + '.pcd'),
            'bin_path': os.path.join(PCD_DIR, source['name'] + '.bin'),
            'timestamp': source['timestamp'],
        }]

        target_pcd_path = os.path.join(PCD_DIR, target['name'] + '.pcd')
        target_bin_path = os.path.join(PCD_DIR, target['name'] + '.bin')

        # 推算标签
        predicted = propagate_labels(
            target_pcd_path=target_pcd_path,
            labeled_frames=labeled_frames,
            ego_poses=ego_poses,
            distance_threshold=0.5,
        )

        # 真实标签（BIN 可能比 PCD 长，取前 N 个）
        ground_truth = read_bin(target_bin_path)[:len(predicted)]

        # 计算准确率
        total = len(ground_truth)
        correct = np.sum(predicted == ground_truth)
        accuracy = correct / total * 100

        # 排除默认类别 0 的干扰：只看预测非零的点
        nonzero_mask = predicted != 0
        nonzero_total = np.sum(nonzero_mask)
        nonzero_correct = np.sum(predicted[nonzero_mask] == ground_truth[nonzero_mask])
        nonzero_accuracy = nonzero_correct / nonzero_total * 100 if nonzero_total > 0 else 0

        # 在真实标签非零的点上计算
        gt_nonzero_mask = ground_truth != 0
        gt_nonzero_total = np.sum(gt_nonzero_mask)
        gt_nonzero_correct = np.sum(predicted[gt_nonzero_mask] == ground_truth[gt_nonzero_mask])
        gt_nonzero_accuracy = gt_nonzero_correct / gt_nonzero_total * 100 if gt_nonzero_total > 0 else 0

        print(f"\n用 {source['name']} 推算 {target['name']}:")
        print(f"  总点数: {total}")
        print(f"  整体准确率: {accuracy:.2f}% ({correct}/{total})")
        print(f"  预测非零点准确率: {nonzero_accuracy:.2f}% ({nonzero_correct}/{nonzero_total})")
        print(f"  真实非零点准确率: {gt_nonzero_accuracy:.2f}% ({gt_nonzero_correct}/{gt_nonzero_total})")

        # 距离阈值内匹配率
        within_threshold = np.sum(nonzero_mask)
        print(f"  阈值内匹配点数: {within_threshold}/{total} ({within_threshold/total*100:.1f}%)")


def validate_output_sizes():
    """检查输出 BIN 点数是否与 PCD 匹配"""
    print("\n" + "=" * 60)
    print("验证 2: 输出 BIN 点数与 PCD 匹配性检查")
    print("=" * 60)

    unlabeled = [
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

    all_ok = True
    for name in unlabeled:
        pcd_path = os.path.join(PCD_DIR, name + '.pcd')
        bin_path = os.path.join(PCD_DIR, name + '.bin')

        if not os.path.exists(bin_path):
            print(f"  [MISSING] {name}.bin 不存在")
            all_ok = False
            continue

        pts = read_pcd(pcd_path)
        labels = read_bin(bin_path)

        if pts.shape[0] == len(labels):
            print(f"  [OK] {name}: {pts.shape[0]} 点 == {len(labels)} 标签")
        else:
            print(f"  [FAIL] {name}: {pts.shape[0]} 点 != {len(labels)} 标签")
            all_ok = False

    if all_ok:
        print("  全部通过 ✓")


def validate_label_distribution():
    """对比输出 BIN 与源 BIN 的类别分布"""
    print("\n" + "=" * 60)
    print("验证 3: 类别分布对比")
    print("=" * 60)

    # 源标注帧分布
    print("\n源标注帧分布:")
    for name in ['0015_1775728137200007680', '0020_1775728137700007680']:
        labels = read_bin(os.path.join(PCD_DIR, name + '.bin'))
        unique, counts = np.unique(labels, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"  {name}: {dist}")

    # 输出帧分布
    print("\n输出帧分布（汇总）:")
    all_labels = []
    for name in ['0015_1775728143200007680', '0016_1775728143700007680',
                 '0017_1775728144200007680', '0018_1775728144700007680',
                 '0019_1775728145200007680', '0020_1775728145700007680',
                 '0021_1775728146200007680', '0022_1775728146700007680',
                 '0023_1775728147200007680', '0024_1775728147700007680']:
        bin_path = os.path.join(PCD_DIR, name + '.bin')
        if os.path.exists(bin_path):
            all_labels.append(read_bin(bin_path))

    if all_labels:
        merged = np.concatenate(all_labels)
        unique, counts = np.unique(merged, return_counts=True)
        total = len(merged)
        print(f"  总标签数: {total}")
        for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
            print(f"    类别 {u:2d}: {c:>8d} ({c/total*100:.2f}%)")


if __name__ == '__main__':
    validate_cross_prediction()
    validate_output_sizes()
    validate_label_distribution()
