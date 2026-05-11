"""ego_pose 加载 + 自车运动补偿（帧间坐标变换）"""

import json
import numpy as np


def load_ego_poses(json_path):
    """
    加载 ego_pose.json，返回 timestamp → 4x4 变换矩阵 的映射。

    返回:
        poses: dict[int, np.ndarray]  # timestamp → (4, 4) transform
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    poses = {}
    for entry in data:
        ts = entry['sample_timestamp']
        T = np.array(entry['transform'], dtype=np.float64)
        poses[ts] = T

    return poses


def transform_points(points_xyz, src_T, dst_T):
    """
    自车运动补偿：将点从源帧坐标系变换到目标帧坐标系。

    公式: p_dst = inv(T_dst) @ T_src @ p_src

    参数:
        points_xyz: (N, 3) 源帧中的点坐标
        src_T: (4, 4) 源帧的世界变换矩阵
        dst_T: (4, 4) 目标帧的世界变换矩阵

    返回:
        transformed: (N, 3) 目标帧中的点坐标
    """
    # 构造齐次坐标 (N, 4)
    N = points_xyz.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    pts_homo = np.hstack([points_xyz.astype(np.float64), ones])  # (N, 4)

    # inv(T_dst) @ T_src
    T = np.linalg.solve(dst_T, src_T)  # 比 inv(dst_T) @ src_T 数值更稳定

    # 变换: (4, 4) @ (4, N) → (4, N) → (N, 4)
    transformed = (T @ pts_homo.T).T

    return transformed[:, :3]
