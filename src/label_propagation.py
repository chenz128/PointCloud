"""标签传播：KD-Tree 最近邻 + 自车运动补偿"""

import numpy as np
from scipy.spatial import cKDTree

from pcd_reader import read_pcd
from bin_io import read_bin
from transform import transform_points


def propagate_labels(target_pcd_path, labeled_frames, ego_poses, distance_threshold=0.5):
    """
    为目标帧的每个点分配语义标签。

    参数:
        target_pcd_path: 目标帧 PCD 文件路径
        labeled_frames: list of dict, 每个包含:
            - 'pcd_path': 标注帧 PCD 路径
            - 'bin_path': 标注帧 BIN 路径
            - 'timestamp': 标注帧时间戳
        ego_poses: dict[int, np.ndarray], timestamp → 4x4 transform
        distance_threshold: 距离阈值（米），超过则标记为默认类别

    返回:
        labels: np.ndarray, shape (N,), dtype int32
    """
    # 提取目标帧时间戳
    target_ts = _extract_timestamp(target_pcd_path)
    target_T = ego_poses[target_ts]

    # 加载目标帧点云
    target_points = read_pcd(target_pcd_path)[:, :3]  # (N, 3)

    # 合并所有标注帧的变换后点云和标签
    all_points = []
    all_labels = []

    for frame in labeled_frames:
        # 加载标注帧点云和标签
        src_points = read_pcd(frame['pcd_path'])[:, :3]  # (M, 3)
        src_labels = read_bin(frame['bin_path'])
        src_T = ego_poses[frame['timestamp']]

        # 自车运动补偿：变换到目标帧坐标系
        transformed = transform_points(src_points, src_T, target_T)

        all_points.append(transformed)
        all_labels.append(src_labels)

    all_points = np.vstack(all_points)  # (M_total, 3)
    all_labels = np.concatenate(all_labels)  # (M_total,)

    # 构建 KD-Tree 并查询最近邻
    tree = cKDTree(all_points)
    distances, indices = tree.query(target_points, k=1)

    # 分配标签，距离超阈值的标为 0
    labels = all_labels[indices].copy()
    labels[distances > distance_threshold] = 0

    return labels.astype(np.int32)


def _extract_timestamp(filepath):
    """从文件路径中提取时间戳（文件名格式: {编号}_{时间戳}.{ext}）"""
    import os
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return int(basename.split('_')[1])
