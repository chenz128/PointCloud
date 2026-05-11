"""BIN 语义分割标签文件读写"""

import numpy as np


def read_bin(filepath):
    """读取 BIN 标签文件，返回 int32 数组"""
    return np.fromfile(filepath, dtype=np.int32)


def write_bin(labels, filepath):
    """将 int32 标签数组写入 BIN 文件"""
    labels.astype(np.int32).tofile(filepath)
