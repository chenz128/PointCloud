"""PCD v0.7 binary 格式写出器"""

import numpy as np
import struct


def write_pcd(filepath, points):
    """
    写出 PCD v0.7 binary 格式文件。

    参数:
        filepath: 输出文件路径
        points: np.ndarray, shape (N, 5), dtype float32
                列: x, y, z, sensor_id, intensity
    """
    points = points.astype(np.float32)
    N = points.shape[0]

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z sensor_id intensity\n"
        "SIZE 4 4 4 4 4\n"
        "TYPE F F F F F\n"
        "COUNT 1 1 1 1 1\n"
        f"WIDTH {N}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {N}\n"
        "DATA binary\n"
    )

    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(points.tobytes())
