# YOLOv5-Face 可定制关键点检测

基于 [YOLOv5-Face](https://github.com/deepcam-cn/yolov5-face) 修改的**可定制关键点数量**的目标检测模型。

支持任意数量的关键点检测（3点、4点、5点、6点...），只需修改一个配置文件即可切换。
默认配置为 **RoboMaster 装甲板 4 关键点检测**，支持 36 类装甲板分类。

---

## 目录

- [效果展示](#效果展示)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据集准备](#数据集准备)
- [训练](#训练)
- [检测推理](#检测推理)
- [模型导出](#模型导出)
- [可用模型](#可用模型)
- [如何切换关键点数量](#如何切换关键点数量)
- [标签格式详解](#标签格式详解)
- [输出通道数说明](#输出通道数说明)
- [工具脚本](#工具脚本)
- [修改详情](#修改详情)
- [常见问题](#常见问题)
- [参考和致谢](#参考和致谢)

---

## 效果展示

模型能够检测目标并输出关键点坐标，同时进行分类。

**4 关键点模式（装甲板检测）**：
- 检测装甲板，输出 4 个角点坐标形成四边形轮廓
- 支持 36 类装甲板分类（蓝方/红方/中立/紫方 + 子类型）
- 绘制红/绿/蓝/黄色圆点 + 四边形连线

**5 关键点模式（人脸检测）**：
- 检测人脸，输出 5 个关键点（左眼、右眼、鼻子、左嘴角、右嘴角）
- 支持 WiderFace 等标准人脸数据集

---

## 核心特性

### 1. 关键点数量可定制

这是本项目最大的改进。原始 YOLOv5-Face 硬编码了 5 个关键点，本项目将其改为**参数化配置**：

```yaml
# data/widerface.yaml 中只需修改这一行
num_points: 4   # 改成你想要的关键点数量
```

| num_points | 典型应用 | 标签列数 | 输出通道(nc=1) |
|------------|---------|----------|---------------|
| 3 | 三角形目标 | 11 | 12 |
| 4 | 装甲板四角 | 13 | 14 |
| 5 | 人脸五点 | 15 | 16 |
| 6 | 自定义 | 17 | 18 |

### 2. 从人脸检测到通用检测

| 对比项 | 原始 (人脸) | 本项目 (装甲板) |
|--------|------------|----------------|
| 关键点数量 | 5 个 (左眼、右眼、鼻子、左嘴角、右嘴角) | 4 个 (装甲板四个角)，可自定义 |
| 关键点坐标数 | 10 (5×2) | 8 (4×2)，可自定义 |
| 每个anchor输出 | `nc + 5 + 10` | `nc + 5 + num_points*2` |
| 标签列数 | 15 (类别 + 框 + 10坐标) | `5 + num_points*2` |
| 类别数量 | 1 (人脸) | 36 (装甲板类型) |
| 检测源 | 人脸图片 | `./armor` 目录 |

### 3. 多种轻量级模型

提供了从 Nano 到 Large 的多种模型配置，以及 MobileNetV3 backbone，满足不同场景需求。

---

## 项目结构

```
yolov5-face2/
├── armor/                          # 测试用装甲板图片
├── data/
│   ├── widerface.yaml              # ★ 数据集配置 (关键点数量在这里设置)
│   ├── hyp.scratch.yaml            # 超参数配置
│   ├── retinaface2yolo.py          # RetinaFace标签转YOLO格式
│   ├── train2yolo.py               # 训练集标签转换脚本
│   └── test_data/                  # 训练/验证数据
│       ├── train/
│       │   ├── images/             # 训练图片 (.jpg/.png)
│       │   └── labels/             # 训练标签 (.txt)
│       └── val/
│           ├── images/             # 验证图片
│           └── labels/             # 验证标签
├── models/
│   ├── yolo.py                     # ★ 模型核心 (Detect类, parse_model)
│   ├── common.py                   # 通用模块 (Conv, C3, SPP, MobileNetV3等)
│   ├── experimental.py             # 实验性模块 (attempt_load等)
│   ├── yolov5n.yaml                # YOLOv5-Nano 配置 (~1.8M参数)
│   ├── yolov5n-0.5.yaml            # YOLOv5-Nano-0.5 配置 (~1M参数)
│   ├── yolov5s.yaml                # YOLOv5-Small 配置 (~7M参数)
│   ├── yolov5m.yaml                # YOLOv5-Medium 配置 (~21M参数)
│   ├── yolov5l.yaml                # YOLOv5-Large 配置 (~46M参数)
│   ├── yolov5s-mobilenet.yaml      # MobileNetV3 backbone 配置 (~3M参数)
│   ├── yolov5n6.yaml               # YOLOv5-Nano P6 配置
│   ├── yolov5s6.yaml               # YOLOv5-Small P6 配置
│   ├── yolov5m6.yaml               # YOLOv5-Medium P6 配置
│   ├── yolov5l6.yaml               # YOLOv5-Large P6 配置
│   ├── blazeface.yaml              # BlazeFace 配置
│   └── blazeface_fpn.yaml          # BlazeFace-FPN 配置
├── utils/
│   ├── loss.py                     # ★ 损失函数 (compute_loss, build_targets)
│   ├── general.py                  # ★ NMS等通用函数 (non_max_suppression_face)
│   ├── face_datasets.py            # ★ 数据加载 (LoadFaceImagesAndLabels)
│   ├── datasets.py                 # 通用数据加载 (未使用)
│   ├── autoanchor.py               # 自动anchor计算
│   ├── google_utils.py             # Google云工具
│   ├── metrics.py                  # 评估指标 (AP, mAP)
│   ├── plots.py                    # 绘图工具
│   ├── torch_utils.py              # PyTorch工具
│   ├── activations.py              # 激活函数
│   ├── augmentations.py            # 数据增强
│   ├── callbacks.py                # 回调函数
│   ├── downloads.py                # 下载工具
│   ├── general.py                  # 通用工具
│   ├── infer_utils.py              # 推理工具
│   └── loggers/                    # 日志工具
├── weights/                        # 预训练权重目录
├── runs/                           # 训练/检测结果目录
├── train.py                        # ★ 训练脚本
├── test.py                         # ★ 验证/测试脚本
├── detect_face.py                  # ★ 检测推理脚本
├── test_widerface.py               # WiderFace评估脚本
├── export.py                       # 模型导出脚本 (ONNX等)
├── hubconf.py                      # PyTorch Hub配置
├── torch2trt/                      # TensorRT转换工具
│   ├── main.py                     # TensorRT推理脚本
│   └── trt_model.py                # TensorRT模型加载
├── dataset_revise.py               # 数据集清理工具
├── transform_labels.py             # 标签格式转换工具
├── conver_pt1_2.py                 # 模型权重转换工具
├── 修改文档_关键点参数化.md          # 详细修改文档
└── README.md                       # 本文件
```

> 标注 ★ 的文件包含核心修改，是理解本项目的关键。

---

## 环境配置

### 系统要求

- **操作系统**：Windows 10/11 或 Linux (Ubuntu 18.04+)
- **Python**：3.8 或更高版本
- **GPU**：NVIDIA GPU（推荐）+ CUDA 11.0+
- **内存**：8GB 以上（推荐 16GB+）
- **硬盘**：至少 10GB 可用空间

### 依赖包

```
Python >= 3.8
PyTorch >= 1.7 (推荐 1.10+)
torchvision >= 0.8
opencv-python >= 4.1
numpy >= 1.18
matplotlib >= 3.2
tqdm >= 4.41
pyyaml >= 5.3
Pillow >= 7.1
scipy >= 1.4
```

### 安装步骤

**方式一：pip 安装**

```bash
# 创建虚拟环境（推荐）
conda create -n yolov5_face python=3.9
conda activate yolov5_face

# 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install opencv-python numpy matplotlib tqdm pyyaml Pillow scipy
```

**方式二：requirements.txt 安装**

```bash
pip install -r requirements.txt
```

### 验证安装

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出类似 `2.0.0+cu118` 和 `True`，说明安装成功。

---

## 数据集准备

### 1. 目录结构

```
data/test_data/
├── train/
│   ├── images/         # 训练图片
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   └── labels/         # 训练标签（与图片同名，扩展名为.txt）
│       ├── 0001.txt
│       ├── 0002.txt
│       └── ...
└── val/
    ├── images/         # 验证图片
    │   ├── 0001.jpg
    │   └── ...
    └── labels/         # 验证标签
        ├── 0001.txt
        └── ...
```

### 2. 标签格式

每个标签文件为 `.txt`，每行一个目标，格式如下：

```
类别ID  中心x  中心y  宽  高  x1  y1  x2  y2  x3  y3  x4  y4
```

**各字段说明**：

| 字段 | 说明 | 取值范围 |
|------|------|---------|
| 类别ID | 类别的数字编号 | 0 ~ nc-1 |
| 中心x | 边界框中心x坐标（归一化） | 0.0 ~ 1.0 |
| 中心y | 边界框中心y坐标（归一化） | 0.0 ~ 1.0 |
| 宽 | 边界框宽度（归一化） | 0.0 ~ 1.0 |
| 高 | 边界框高度（归一化） | 0.0 ~ 1.0 |
| x1, y1 | 第1个关键点坐标（归一化） | 0.0 ~ 1.0 或 -1（不可见） |
| x2, y2 | 第2个关键点坐标（归一化） | 0.0 ~ 1.0 或 -1（不可见） |
| x3, y3 | 第3个关键点坐标（归一化） | 0.0 ~ 1.0 或 -1（不可见） |
| x4, y4 | 第4个关键点坐标（归一化） | 0.0 ~ 1.0 或 -1（不可见） |

**总列数** = `5 + num_points * 2`
- 4个关键点时：5 + 8 = 13 列
- 5个关键点时：5 + 10 = 15 列

**示例**（4个关键点，类别0）：
```
0 0.500 0.500 0.100 0.100 0.450 0.450 0.550 0.450 0.550 0.550 0.450 0.550
```

**示例**（4个关键点，第3个不可见）：
```
0 0.500 0.500 0.100 0.100 0.450 0.450 0.550 0.450 -1.000 -1.000 0.450 0.550
```

**示例**（5个关键点，类别1）：
```
1 0.300 0.400 0.150 0.200 0.250 0.350 0.350 0.350 0.300 0.400 0.260 0.450 0.340 0.450
```

### 3. 类别定义（默认36类装甲板）

```
BG  - 蓝方守卫 (Blue Guard)
B1  - 蓝方1号 (Blue 1)
B2  - 蓝方2号 (Blue 2)
B3  - 蓝方3号 (Blue 3)
B4  - 蓝方4号 (Blue 4)
B5  - 蓝方5号 (Blue 5)
BO  - 蓝方前哨站 (Blue Outpost)
BBs - 蓝方小基地 (Blue Base small)
BBb - 蓝方大基地 (Blue Base big)

RG  - 红方守卫 (Red Guard)
R1  - 红方1号 (Red 1)
R2  - 红方2号 (Red 2)
R3  - 红方3号 (Red 3)
R4  - 红方4号 (Red 4)
R5  - 红方5号 (Red 5)
RO  - 红方前哨站 (Red Outpost)
RBs - 红方小基地 (Red Base small)
RBb - 红方大基地 (Red Base big)

NG  - 中立守卫 (Neutral Guard)
N1  - 中立1号 (Neutral 1)
N2  - 中立2号 (Neutral 2)
N3  - 中立3号 (Neutral 3)
N4  - 中立4号 (Neutral 4)
N5  - 中立5号 (Neutral 5)
NO  - 中立前哨站 (Neutral Outpost)
NBs - 中立小基地 (Neutral Base small)
NBb - 中立大基地 (Neutral Base big)

PG  - 紫方守卫 (Purple Guard)
P1  - 紫方1号 (Purple 1)
P2  - 紫方2号 (Purple 2)
P3  - 紫方3号 (Purple 3)
P4  - 紫方4号 (Purple 4)
P5  - 紫方5号 (Purple 5)
PO  - 紫方前哨站 (Purple Outpost)
PBs - 紫方小基地 (Purple Base small)
PBb - 紫方大基地 (Purple Base big)
```

### 4. 数据集配置文件

`data/widerface.yaml` 是数据集配置文件，内容如下：

```yaml
# 训练集路径
train: ./data/test_data/train

# 验证集路径
val: ./data/test_data/val

# 类别数量
nc: 36

# 关键点数量（在这里修改！）
num_points: 4

# 类别名称
names: ['BG','B1','B2','B3','B4','B5','BO','BBs','BBb',
        'RG','R1','R2','R3','R4','R5','RO','RBs','RBb',
        'NG','N1','N2','N3','N4','N5','NO','NBs','NBb',
        'PG','P1','P2','P3','P4','P5','PO','PBs','PBb']
```

**重要**：`num_points` 是关键点数量的唯一配置位置。修改这里就能切换关键点数量。

---

## 训练

### 1. 准备预训练权重

下载 YOLOv5 预训练权重并放入 `weights/` 目录：

```bash
mkdir weights
# 下载 yolov5n 权重（推荐，轻量级）
wget -P weights/ https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt
# 或下载 yolov5s 权重
wget -P weights/ https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
```

### 2. 训练命令

**使用 YOLOv5n（轻量级，推荐嵌入式/边缘设备）**：

```bash
python train.py \
    --data data/widerface.yaml \
    --cfg models/yolov5n.yaml \
    --weights weights/yolov5n.pt \
    --batch-size 32 \
    --epochs 300 \
    --img 640 \
    --device 0
```

**使用 YOLOv5s（通用场景）**：

```bash
python train.py \
    --data data/widerface.yaml \
    --cfg models/yolov5s.yaml \
    --weights weights/yolov5s.pt \
    --batch-size 16 \
    --epochs 300 \
    --img 640 \
    --device 0
```

**使用 YOLOv5m（高精度需求）**：

```bash
python train.py \
    --data data/widerface.yaml \
    --cfg models/yolov5m.yaml \
    --weights weights/yolov5m.pt \
    --batch-size 16 \
    --epochs 300 \
    --img 640 \
    --device 0
```

**使用 MobileNetV3 backbone（移动端部署）**：

```bash
python train.py \
    --data data/widerface.yaml \
    --cfg models/yolov5s-mobilenet.yaml \
    --weights '' \
    --batch-size 32 \
    --epochs 300 \
    --img 640 \
    --device 0
```

**从头训练（不使用预训练权重）**：

```bash
python train.py \
    --data data/widerface.yaml \
    --cfg models/yolov5s.yaml \
    --weights '' \
    --batch-size 16 \
    --epochs 300 \
    --img 640 \
    --device 0
```

### 3. 训练参数说明

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `--data` | 数据集配置文件路径 | `data/widerface.yaml` | - |
| `--cfg` | 模型结构配置文件 | `models/yolov5s.yaml` | 根据需求选择 |
| `--weights` | 预训练权重路径 | `weights/yolov5s.pt` | - |
| `--batch-size` | 批大小 | 16 | 16/32/64 |
| `--epochs` | 训练轮数 | 300 | 100~500 |
| `--img` | 输入图片尺寸 | 640 | 320/416/640 |
| `--device` | GPU设备 | `0` | `0` 或 `0,1` |
| `--workers` | 数据加载线程数 | 8 | 4~8 |
| `--cache-images` | 缓存图片到内存 | False | 小数据集可用 |
| `--rect` | 矩形训练 | False | True可加速 |
| `--single-cls` | 单类别模式 | False | - |
| `--adam` | 使用Adam优化器 | False | True可加速收敛 |
| `--sync-bn` | 同步BN（多GPU） | False | 多GPU时开启 |
| `--multi-scale` | 多尺度训练 | False | True可提升精度 |

### 4. 超参数配置

超参数在 `data/hyp.scratch.yaml` 中配置：

```yaml
lr0: 0.01           # 初始学习率
lrf: 0.2            # 最终学习率 = lr0 * lrf
momentum: 0.937     # SGD动量
weight_decay: 0.0005  # 权重衰减
warmup_epochs: 3.0  # 预热轮数
box: 0.05           # 边界框损失权重
cls: 0.5            # 分类损失权重
landmark: 0.005     # 关键点损失权重
obj: 1.0            # 置信度损失权重
iou_t: 0.20         # IoU匹配阈值
anchor_t: 4.0       # anchor匹配阈值
fl_gamma: 0.0       # focal loss gamma
mosaic: 0.5         # mosaic增强概率
fliplr: 0.5         # 左右翻转概率
```

### 5. 训练过程监控

训练过程中会输出以下信息：

```
     Epoch   gpu_mem   box   obj   cls   landmarks  total   targets  img_size
     0/299     2.5G   0.05  0.02  0.01    0.003    0.083      15       640
     1/299     2.5G   0.04  0.02  0.01    0.003    0.075      18       640
     ...
```

- **box**：边界框回归损失
- **obj**：置信度损失
- **cls**：分类损失
- **landmarks**：关键点损失
- **total**：总损失

训练结果保存在 `runs/train/exp*/` 目录：
- `weights/best.pt`：最佳权重
- `weights/last.pt`：最后一轮权重
- `results.txt`：训练日志
- `results.png`：训练曲线图

### 6. 断点续训

```bash
python train.py \
    --data data/widerface.yaml \
    --cfg models/yolov5s.yaml \
    --weights runs/train/exp5/weights/last.pt \
    --epochs 300 \
    --resume
```

---

## 检测推理

### 1. 检测命令

**检测单张图片**：

```bash
python detect_face.py \
    --source ./armor/0033.png \
    --weights runs/train/exp5/weights/best.pt \
    --img 640 \
    --conf-thres 0.5 \
    --device 0
```

**检测整个文件夹**：

```bash
python detect_face.py \
    --source ./armor/ \
    --weights runs/train/exp5/weights/best.pt \
    --img 640 \
    --conf-thres 0.5 \
    --device 0
```

**检测视频**：

```bash
python detect_face.py \
    --source video.mp4 \
    --weights runs/train/exp5/weights/best.pt \
    --img 640 \
    --conf-thres 0.5 \
    --device 0
```

**检测摄像头**：

```bash
python detect_face.py \
    --source 0 \
    --weights runs/train/exp5/weights/best.pt \
    --img 640 \
    --conf-thres 0.5 \
    --device 0
```

### 2. 检测参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source` | 检测源（图片/视频/摄像头/文件夹） | `./armor` |
| `--weights` | 模型权重路径 | `runs/train/exp6/weights/best.pt` |
| `--img` | 输入图片尺寸 | 640 |
| `--conf-thres` | 置信度阈值 | 0.6 |
| `--iou-thres` | NMS的IoU阈值 | 0.5 |
| `--device` | GPU设备 | 自动选择 |
| `--save-img` | 保存结果图片 | True |
| `--view-img` | 实时显示结果 | False |

### 3. 检测输出

检测结果保存在 `runs/detect/exp*/` 目录：

- 绘制边界框（绿色矩形）
- 绘制关键点（红/绿/蓝/黄色圆点）
- 绘制关键点连线（4个关键点时绘制四边形）
- 显示置信度和类别名称

### 4. 检测流程说明

```
输入图片
    ↓
预处理（letterbox缩放到640x640）
    ↓
模型推理（得到原始输出）
    ↓
NMS后处理（过滤重叠框）
    ↓
坐标缩放（从640x640缩放回原图尺寸）
    ↓
绘制结果（边界框 + 关键点 + 类别）
    ↓
保存/显示
```

---

## 模型导出

### 1. 导出 ONNX

```bash
python export.py \
    --weights runs/train/exp5/weights/best.pt \
    --include onnx \
    --img 640
```

导出的 ONNX 文件在权重同目录下。

### 2. TensorRT 转换

```bash
cd torch2trt

# 需要先将 ONNX 转换为 TensorRT 引擎
# 然后使用 main.py 进行推理
python main.py \
    --img_path sample.jpg \
    --trt_path best.trt \
    --output_shape [1,25200,14] \
    --num_points 4
```

**注意**：`output_shape` 的最后一维 = `5 + num_points * 2 + nc`
- 4个关键点，1个类别：5 + 8 + 1 = 14
- 4个关键点，36个类别：5 + 8 + 36 = 49

---

## 可用模型

### 1. 标准 YOLOv5 模型

| 模型 | 配置文件 | 参数量 | 适用场景 | 推荐batch-size |
|------|---------|--------|---------|---------------|
| YOLOv5n | `yolov5n.yaml` | ~1.8M | 嵌入式/边缘设备 | 32 |
| YOLOv5n-0.5 | `yolov5n-0.5.yaml` | ~1M | 极致轻量 | 32 |
| YOLOv5s | `yolov5s.yaml` | ~7M | 通用场景 | 16 |
| YOLOv5m | `yolov5m.yaml` | ~21M | 高精度需求 | 16 |
| YOLOv5l | `yolov5l.yaml` | ~46M | 最高精度 | 8 |

### 2. P6 模型（更高分辨率）

| 模型 | 配置文件 | 适用场景 |
|------|---------|---------|
| YOLOv5n6 | `yolov5n6.yaml` | 轻量级 + 高分辨率 |
| YOLOv5s6 | `yolov5s6.yaml` | 通用 + 高分辨率 |
| YOLOv5m6 | `yolov5m6.yaml` | 高精度 + 高分辨率 |
| YOLOv5l6 | `yolov5l6.yaml` | 最高精度 + 高分辨率 |

### 3. 自定义 Backbone

| 模型 | 配置文件 | 参数量 | 适用场景 |
|------|---------|--------|---------|
| MobileNetV3 | `yolov5s-mobilenet.yaml` | ~3M | 移动端部署 |
| BlazeFace | `blazeface.yaml` | ~1M | 超轻量 |
| BlazeFace-FPN | `blazeface_fpn.yaml` | ~1.5M | 轻量 + 多尺度 |

### 4. 如何选择模型

- **嵌入式/边缘设备**：YOLOv5n 或 MobileNetV3
- **通用场景**：YOLOv5s（推荐）
- **高精度需求**：YOLOv5m 或 YOLOv5l
- **极致轻量**：YOLOv5n-0.5 或 BlazeFace
- **高分辨率输入**：P6 系列（yolov5n6 等）

---

## 如何切换关键点数量

### 1. 修改配置

只需修改 `data/widerface.yaml` 中的 `num_points`：

```yaml
# 4个关键点（装甲板检测）
num_points: 4

# 5个关键点（人脸检测）
num_points: 5

# 3个关键点（自定义）
num_points: 3

# 6个关键点（自定义）
num_points: 6
```

### 2. 准备对应标签

标签列数必须与 `num_points` 匹配：
- 总列数 = `5 + num_points * 2`
- 4个关键点：13列
- 5个关键点：15列
- 3个关键点：11列

### 3. 重新训练

修改配置和标签后，需要重新训练模型：

```bash
python train.py \
    --data data/widerface.yaml \
    --cfg models/yolov5s.yaml \
    --weights weights/yolov5s.pt \
    --batch-size 16 \
    --epochs 300 \
    --img 640
```

### 4. 输出通道数

模型的输出通道数会自动调整：

```
输出通道数 no = nc + 5 + num_points * 2
```

| num_points | nc=1 时 | nc=36 时 | nc=80 时 |
|------------|---------|----------|----------|
| 3 | 12 | 47 | 91 |
| 4 | 14 | 49 | 93 |
| 5 | 16 | 51 | 95 |
| 6 | 18 | 53 | 97 |

---

## 标签格式详解

### 1. 标签文件格式

每个标签文件（.txt）的每一行代表一个目标：

```
类别ID  中心x  中心y  宽  高  x1  y1  x2  y2  ...  xN  yN
```

### 2. 坐标归一化

所有坐标都是**归一化值**（0~1之间）：

```
归一化x = 像素x / 图片宽度
归一化y = 像素y / 图片高度
```

例如，一张 640x480 的图片中，像素坐标 (320, 240) 归一化后为 (0.5, 0.5)。

### 3. 不可见关键点

如果某个关键点不可见（被遮挡或超出图片范围），将其坐标设为 `-1`：

```
0 0.500 0.500 0.100 0.100 0.450 0.450 0.550 0.450 -1.000 -1.000 0.450 0.550
```

上例中第3个关键点不可见。

### 4. 标签转换工具

**从 RetinaFace 格式转换**：

```bash
python data/retinaface2yolo.py --src_path path/to/retinaface/labels --dst_path path/to/yolo/labels
```

**从其他格式转换**：

```bash
python data/train2yolo.py
```

**标签格式检查**：

```bash
python dataset_revise.py
```

---

## 输出通道数说明

### 1. 检测头输出格式

每个 anchor（锚框）输出一个一维向量：

```
[ x, y, w, h, conf, x1, y1, x2, y2, ..., xN, yN, cls1, cls2, ..., clsM ]
|______box______|____|_______关键点坐标________|_______类别概率_______|
      4个值      1个值      N*2个值（N个关键点）        M个类别
```

### 2. 关键索引位置

```
索引 0~3           ：box (x, y, w, h)
索引 4             ：置信度 conf
索引 5 ~ 5+N*2-1  ：关键点坐标（N=num_points）
索引 5+N*2 ~ 末尾  ：类别概率
```

### 3. NMS 输出格式

NMS 后处理后的输出格式：

```
[ x1, y1, x2, y2, conf, x1, y1, x2, y2, ..., xN, yN, class_id ]
|______xyxy_____|____|_______关键点坐标________|____class____|
      4个值      1个值      N*2个值（N个关键点）     1个值
```

总列数 = `4 + 1 + num_points * 2 + 1 = 6 + num_points * 2`

---

## 工具脚本

### 1. 数据集清理

```bash
python dataset_revise.py
```

功能：检查数据集中图片和标签是否匹配，删除不匹配的文件。

### 2. 标签格式转换

```bash
python transform_labels.py
```

功能：将4点标签转换为纯边界框标签（去掉关键点坐标）。

### 3. RetinaFace 转 YOLO

```bash
python data/retinaface2yolo.py
```

功能：将 RetinaFace 格式的人脸标注转换为 YOLO 格式。

### 4. WiderFace 评估

```bash
python test_widerface.py \
    --weights runs/train/exp5/weights/best.pt \
    --dataset_folder ../WiderFace/val/images/ \
    --save_folder ./widerface_evaluate/widerface_txt/
```

功能：在 WiderFace 数据集上评估模型性能。

---

## 修改详情

本项目对原始 YOLOv5-Face 进行了以下核心修改：

### 1. 关键点数量参数化

将原本硬编码的关键点数量改为可配置参数，涉及以下文件：

| 文件 | 修改说明 |
|------|---------|
| `data/widerface.yaml` | 新增 `num_points` 配置项 |
| `models/yolo.py` | Detect 类接收 num_points，动态计算通道数 |
| `train.py` | 从数据配置读取 num_points，传给模型和数据加载器 |
| `test.py` | 从数据配置读取 num_points，传给 NMS 和数据加载器 |
| `utils/loss.py` | 从模型获取 num_points，动态计算损失 |
| `utils/general.py` | NMS 函数接收 num_points，动态计算索引 |
| `utils/face_datasets.py` | 数据加载器接收 num_points，动态处理标签 |
| `detect_face.py` | 从模型获取 num_points，动态处理检测结果 |
| `test_widerface.py` | 从模型获取 num_points，动态处理检测结果 |
| `torch2trt/main.py` | 新增 --num_points 参数 |

### 2. 核心公式

```
cls_all = 5 + num_points * 2
no = nc + cls_all
```

- `cls_all`：类别通道起始位置（即关键点坐标结束位置）
- `no`：每个 anchor 的总输出通道数
- `nc`：类别数

### 3. 修改标记

所有修改位置都以注释标记：
- `#修改`：原始5点→4点的修改（用户之前做的）
- `#修改2`：关键点数量参数化的修改（本次新增）

### 4. 详细修改文档

完整的修改说明请参阅：

**[修改文档_关键点参数化.md](修改文档_关键点参数化.md)**

该文档包含：
- 每条修改的详细说明（为什么改、改前改后对比、逐行解释）
- 58条修改的完整列表
- 修复的原代码 Bug 列表
- 参数传递链路图
- 使用说明

---

## 常见问题

### Q1：如何切换关键点数量？

**A**：只需修改 `data/widerface.yaml` 中的 `num_points`，然后重新训练。标签文件的列数也要对应调整。

### Q2：训练时报错 `labels require X columns each`

**A**：标签文件的列数与 `num_points` 不匹配。
- 4个关键点时应为13列
- 5个关键点时应为15列
- 计算公式：`5 + num_points * 2`

### Q3：训练时报错 `UnicodeDecodeError: 'gbk' codec can't decode`

**A**：Windows 系统编码问题。已修复，确保 `train.py` 和 `test.py` 中的 `open()` 调用都加了 `encoding='utf-8'`。

### Q4：如何使用自己的数据集？

**A**：
1. 准备图片和标签（按上述格式）
2. 放入 `data/test_data/train/` 和 `data/test_data/val/`
3. 修改 `data/widerface.yaml` 中的 `nc`（类别数）和 `names`（类别名）
4. 修改 `num_points`（关键点数量）
5. 开始训练

### Q5：训练时 GPU 内存不足怎么办？

**A**：
- 减小 `--batch-size`（如从16改为8或4）
- 减小 `--img`（如从640改为416或320）
- 使用更轻量的模型（如 YOLOv5n 或 MobileNetV3）

### Q6：如何提高检测精度？

**A**：
- 增加训练数据量
- 增加训练轮数（`--epochs`）
- 使用更大的模型（YOLOv5m 或 YOLOv5l）
- 调整超参数（特别是 `landmark` 权重）
- 使用数据增强（mosaic、mixup 等）

### Q7：如何部署到边缘设备？

**A**：
1. 使用轻量级模型（YOLOv5n 或 MobileNetV3）
2. 导出 ONNX：`python export.py --weights best.pt --include onnx`
3. 转换为 TensorRT：使用 `torch2trt/main.py`
4. 使用 `torch2trt/main.py` 进行推理

### Q8：检测结果中关键点顺序是什么？

**A**：关键点顺序与标签中的顺序一致。
- 4个关键点（装甲板）：左上、右上、右下、左下（按标签定义）
- 5个关键点（人脸）：左眼、右眼、鼻子、左嘴角、右嘴角

### Q9：如何修改类别数量？

**A**：
1. 修改 `data/widerface.yaml` 中的 `nc`（类别数）
2. 修改 `names`（类别名称列表）
3. 确保标签中的类别ID在 0 ~ nc-1 范围内
4. 重新训练

### Q10：如何可视化训练过程？

**A**：

```bash
# 启动 TensorBoard
tensorboard --logdir runs/train

# 浏览器打开 http://localhost:6006/
```

---

## 参考和致谢

### 原始项目

- [YOLOv5](https://github.com/ultralytics/yolov5) — 目标检测框架
- [YOLOv5-Face](https://github.com/deepcam-cn/yolov5-face) — 人脸关键点检测

### 参考资料

- 改造思路参考：[YOLOV5-face 改装甲板四点模型](https://blog.csdn.net/m0_58348465/article/details/121423964)
- MobileNetV3 backbone 参考：[TAber-W/RM_4-points_yolov5](https://github.com/TAber-W/RM_4-points_yolov5)

### 相关论文

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [RetinaFace](https://arxiv.org/abs/1905.00641) — 人脸检测
- [Wing Loss](https://arxiv.org/abs/1711.06753) — 关键点损失函数

---

## 许可证

本项目基于 GPL-3.0 许可证开源。

---

## 贡献

欢迎提交 Issue 和 Pull Request！

---

## 更新日志

### v2.0 (当前版本)
- ✅ 关键点数量参数化（支持任意数量关键点）
- ✅ 修复原代码 Bug（NMS 索引、文件编码等）
- ✅ 完善文档（详细修改文档 + README）
- ✅ 新增 test_widerface.py 和 torch2trt/main.py 支持

### v1.0
- ✅ 从人脸5点检测改为装甲板4点检测
- ✅ 支持36类装甲板分类
- ✅ 提供多种轻量级模型配置
