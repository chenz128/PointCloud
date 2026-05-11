"""PCD v0.7 binary_compressed 格式解析器"""

import struct
import numpy as np


def _lzf_decompress(compressed: bytes, uncompressed_size: int) -> bytes:
    """LZF 解压缩算法实现"""
    out = bytearray(uncompressed_size)
    i = 0  # input index
    o = 0  # output index
    in_len = len(compressed)

    while i < in_len:
        ctrl = compressed[i]
        i += 1
        if ctrl < 32:  # literal run
            count = ctrl + 1
            out[o:o + count] = compressed[i:i + count]
            i += count
            o += count
        else:  # back reference
            length = ctrl >> 5
            if length == 7:
                length += compressed[i]
                i += 1
            length += 2
            offset = ((ctrl & 0x1F) << 8) + compressed[i] + 1
            i += 1
            ref = o - offset
            for j in range(length):
                out[o] = out[ref]
                o += 1
                ref += 1

    return bytes(out)


def read_pcd(filepath):
    """
    解析 PCD v0.7 binary_compressed 文件。

    返回:
        points: np.ndarray, shape (N, 5), dtype float32
                列: x, y, z, sensor_id, intensity
    """
    with open(filepath, 'rb') as f:
        # 解析文本头部
        num_points = 0
        num_fields = 0
        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            if line.startswith('FIELDS'):
                num_fields = len(line.split()) - 1
            elif line.startswith('POINTS'):
                num_points = int(line.split()[1])
            elif line.startswith('DATA'):
                break

        # 读取压缩数据头：compressed_size, uncompressed_size
        compressed_size = struct.unpack('I', f.read(4))[0]
        uncompressed_size = struct.unpack('I', f.read(4))[0]

        # 读取并解压
        compressed_data = f.read(compressed_size)
        raw = _lzf_decompress(compressed_data, uncompressed_size)

    # binary_compressed 中数据按列优先存储：所有 x, 所有 y, ...
    point_size = 4  # float32
    points = np.empty((num_points, num_fields), dtype=np.float32)
    for i in range(num_fields):
        start = i * num_points * point_size
        end = start + num_points * point_size
        points[:, i] = np.frombuffer(raw[start:end], dtype=np.float32)

    return points
