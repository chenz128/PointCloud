# PointCloud 点云分割标签插值项目

## 项目目标

通过 2 帧已标注的点云数据（BIN 语义分割标签），结合自车运动补偿（ego_pose），为 10 帧无标注点云（PCD）生成对应的语义分割标签（BIN）。

## 数据说明

### 文件格式

- **PCD 文件**（Point Cloud Data）：PCD v0.7 binary_compressed 格式，每个点包含 5 个 float32 字段：`x, y, z, sensor_id, intensity`。单帧约 40 万个点。
- **BIN 文件**（语义分割标签）：与 PCD 一一对应的 int32 数组，每个点一个标签值。标签值包括 `0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 17, 24, 32, 33` 等类别。
- **ego_pose.json**（自车位姿）：10Hz 采样，每条记录包含 `sample_idx`、`sample_timestamp`（纳秒）、`position`（xyz）、`orientation`（四元数）、`transform`（4×4 齐次变换矩阵）。覆盖 sample_idx 10-298。

### 数据关系

| 数据 | 文件名 | sample_idx | 时间戳 |
|------|--------|------------|--------|
| 标注帧 0015 PCD+BIN | `0015_1775728137200007680` | 15 | ~137.2s |
| 标注帧 0020 PCD+BIN | `0020_1775728137700007680` | 20 | ~137.7s |
| 无标注帧 0015 PCD | `0015_1775728143200007680` | 75 | ~143.2s |
| 无标注帧 0016 PCD | `0016_1775728143700007680` | 80 | ~143.7s |
| 无标注帧 0017 PCD | `0017_1775728144200007680` | 85 | ~144.2s |
| 无标注帧 0018 PCD | `0018_1775728144700007680` | 90 | ~144.7s |
| 无标注帧 0019 PCD | `0019_1775728145200007680` | 95 | ~145.2s |
| 无标注帧 0020 PCD | `0020_1775728145700007680` | 100 | ~145.7s |
| 无标注帧 0021 PCD | `0021_1775728146200007680` | 105 | ~146.2s |
| 无标注帧 0022 PCD | `0022_1775728146700007680` | 110 | ~146.7s |
| 无标注帧 0023 PCD | `0023_1775728147200007680` | 115 | ~147.2s |
| 无标注帧 0024 PCD | `0024_1775728147700007680` | 120 | ~147.7s |

- 文件名格式：`{编号}_{时间戳纳秒}.{pcd|bin}`
- 标注帧与无标注帧时间间隔约 6 秒，车辆有明显位移
- 无标注帧之间间隔 500ms（2Hz），共 10 帧
- ego_pose.json 中所有帧的时间戳都有对应条目

## 技术方案

### 核心算法：自车运动补偿 + 最近邻标签传播

对每个无标注帧，执行以下步骤：

1. 从 ego_pose.json 获取该帧的变换矩阵 `T_target`，以及两个标注帧的变换矩阵 `T_0015`、`T_0020`
2. 将标注帧 0015 的点云通过自车运动补偿对齐到目标帧坐标系：`pts = inv(T_target) @ T_0015 @ pts_0015`
3. 同理将标注帧 0020 的点云对齐到目标帧坐标系：`pts = inv(T_target) @ T_0020 @ pts_0020`
4. 合并两组变换后的带标签点云，构建 KD-Tree
5. 对目标帧每个点查最近邻，取对应标签
6. 距离超过阈值的点标记为默认类别（避免远距离错误传播）
7. 保存为 BIN 文件

### 关键设计决策

- 两个标注帧的点云合并后统一做最近邻，不做时间加权
- 标签为离散类别，最近邻直接取标签，不做线性插值
- 距离阈值初始设为 0.5m，可调参
- ego_pose 仅补偿自车运动；动态物体（车辆、行人等）可能存在偏差

## 项目结构

```
pointcloud/
├── claude.md                          # 项目文档（本文件）
├── .gitignore                         # 屏蔽 data/ 和 .github/skills/
├── data/                              # 数据目录（不入库）
│   ├── ego_pose.json                  # 10Hz 自车位姿
│   └── point_cloud_segmentation/      # PCD 和 BIN 文件
│       ├── 0015_...137200....pcd      # 标注帧 PCD
│       ├── 0015_...137200....bin      # 标注帧 BIN（标签）
│       ├── 0020_...137700....pcd      # 标注帧 PCD
│       ├── 0020_...137700....bin      # 标注帧 BIN（标签）
│       ├── 0015_...143200....pcd      # 无标注帧 PCD（待生成 BIN）
│       ├── 0016_...143700....pcd      # ...
│       └── ...
├── src/
│   ├── pcd_reader.py                  # PCD v0.7 binary_compressed 解析
│   ├── bin_io.py                      # BIN 文件读写
│   ├── transform.py                   # ego_pose 加载 + 自车运动补偿（帧间坐标变换）
│   ├── label_propagation.py           # 标签传播（KD-Tree 最近邻）
│   └── main.py                        # 主流程入口
```

## 开发阶段

### Phase 1: 基础模块

1. **`src/pcd_reader.py`** — 解析 PCD v0.7 binary_compressed，返回 (N, 5) 数组（x y z sensor_id intensity）
2. **`src/bin_io.py`** — BIN 读写（int32 数组）
3. **`src/transform.py`** — 加载 ego_pose.json，实现帧间坐标变换 `inv(T_dst) @ T_src @ [x,y,z,1]^T`

### Phase 2: 插值核心（依赖 Phase 1）

4. **`src/label_propagation.py`** — 加载标注帧 PCD+BIN → 自车运动补偿变换到目标帧坐标系 → 合并构建 cKDTree → 最近邻查询 → 距离阈值过滤 → 返回标签数组

### Phase 3: 主流程（依赖 Phase 2）

5. **`src/main.py`** — 解析数据目录、加载 ego_pose、遍历 10 个无标注帧逐帧调用标签传播、输出 BIN

## 依赖

- Python 3
- numpy
- scipy（scipy.spatial.cKDTree）

## 验证方法

1. 用标注帧 0020 推算标注帧 0015 的标签（反之亦然），对比真实标签计算准确率
2. 检查每个输出 BIN 的元素数是否等于对应 PCD 的点数
3. 统计输出 BIN 中各类别分布是否与源 BIN 大致一致

## 开发与部署

- **本地**：代码编辑、git 提交
- **远端服务器**（`ssh -p 31256 root@8.130.174.55`）：代码运行、测试
- **远端工作空间**：`/data/chenz/pointcloud`
- **远端 conda 环境**：`/data/chenz/conda_env/pointcloud`（Python 3.10 + numpy + scipy）
  - 激活：`conda activate /data/chenz/conda_env/pointcloud`
  - 单条命令运行：`conda run --prefix /data/chenz/conda_env/pointcloud python src/main.py`
- 代码同步：本地 `git push` → 远端 `git pull`
- 数据文件通过 `.gitignore` 屏蔽，不入仓库
