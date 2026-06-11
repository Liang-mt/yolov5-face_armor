# YOLOv5-Face 装甲板四点检测

基于 [YOLOv5-Face](https://github.com/deepcam-cn/yolov5-face) 修改的 **RoboMaster 装甲板四关键点检测模型**。

将原有的人脸检测 + 5关键点定位，改造为装甲板检测 + 4角点定位，支持 36 类装甲板分类。

---

## 效果展示

模型能够检测机器人装甲板，并输出 4 个角点坐标形成四边形轮廓，同时对装甲板进行分类（蓝方/红方/中立/紫方 + 子类型）。

---

## 核心改动概述

| 对比项 | 原始 (人脸) | 改造后 (装甲板) |
|--------|------------|----------------|
| 关键点数量 | 5 个 (左眼、右眼、鼻子、左嘴角、右嘴角) | 4 个 (装甲板四个角) |
| 关键点坐标数 | 10 (5×2) | 8 (4×2) |
| 每个anchor输出 | `nc + 5 + 10` | `nc + 5 + 8` |
| 标签列数 | 15 (类别 + 框 + 10坐标) | 13 (类别 + 框 + 8坐标) |
| 类别偏移量 | 15 | 13 |
| 类别数量 | 1 (人脸) | 36 (装甲板类型) |
| 检测源 | 人脸图片 | `./armor` 目录 |

---

## 项目结构

```
yolov5-face2/
├── armor/                      # 测试用装甲板图片
├── data/
│   ├── widerface.yaml          # 数据集配置 (36类装甲板)
│   ├── hyp.scratch.yaml        # 超参数配置
│   ├── train2yolo.py           # 训练集标签转换脚本
│   ├── val2yolo.py             # 验证集标签转换脚本
│   └── test_data/              # 训练/验证数据
├── models/
│   ├── yolo.py                 # ⭐ 模型核心 (已修改)
│   ├── common.py               # 通用模块 (含MobileNetV3)
│   ├── yolov5s.yaml            # YOLOv5-Small 模型配置
│   ├── yolov5n.yaml            # YOLOv5-Nano (轻量级)
│   ├── yolov5m.yaml            # YOLOv5-Medium
│   ├── yolov5l.yaml            # YOLOv5-Large
│   ├── yolov5s-mobilenet.yaml  # MobileNetV3 backbone
│   └── ...
├── utils/
│   ├── loss.py                 # ⭐ 损失函数 (已修改)
│   ├── general.py              # ⭐ NMS等通用函数 (已修改)
│   ├── face_datasets.py        # ⭐ 数据加载 (已修改)
│   └── ...
├── detect_face.py              # ⭐ 检测推理脚本 (已修改)
├── train.py                    # 训练脚本 (已修改)
├── test.py                     # 测试脚本 (已修改)
├── dataset_revise.py           # 数据集清理工具
└── transform_labels.py         # 标签格式转换工具
```

> 标注 ⭐ 的文件包含核心修改，详见 [MODIFICATION_GUIDE.md](./readme/MODIFICATION_GUIDE.md)。

---

## 环境配置

### 依赖

```
Python >= 3.8
PyTorch >= 1.7
torchvision
opencv-python
numpy
matplotlib
tqdm
pyyaml
```

### 安装

```bash
pip install torch torchvision opencv-python numpy matplotlib tqdm pyyaml
```

---

## 数据集准备

### 标签格式

每个标签文件为 `.txt`，每行一个目标，格式如下：

```
类别ID  cx  cy  w  h  x1  y1  x2  y2  x3  y3  x4  y4
```

- `cx, cy, w, h`：边界框中心点和宽高（归一化到 0~1）
- `x1~y4`：4个角点坐标（归一化到 0~1）
- 共 13 列：1(类别) + 4(框) + 8(4个角点×2坐标)

### 类别定义 (36类)

```
BG, B1, B2, B3, B4, B5, BO, BBs, BBb   # 蓝方 (Guard, 1-5, Outpost, Base小/大)
RG, R1, R2, R3, R4, R5, RO, RBs, RBb   # 红方
NG, N1, N2, N3, N4, N5, NO, NBs, NBb   # 中立(灰方)
PG, P1, P2, P3, P4, P5, PO, PBs, PBb   # 紫方
```

### 目录结构

```
data/test_data/
├── train/
│   ├── images/     # 训练图片 (.jpg/.png)
│   └── labels/     # 训练标签 (.txt)
└── val/
    ├── images/     # 验证图片
    └── labels/     # 验证标签
```

---

## 训练

```bash
# 使用 YOLOv5s 训练
python train.py --data data/widerface.yaml \
                --cfg models/yolov5s.yaml \
                --weights weights/yolov5s.pt \
                --batch-size 16 \
                --epochs 300 \
                --img 640 \
                --device 0

# 使用 YOLOv5n (轻量级)
python train.py --data data/widerface.yaml \
                --cfg models/yolov5n.yaml \
                --weights weights/yolov5n.pt \
                --batch-size 32 \
                --epochs 300 \
                --img 640

# 使用 MobileNetV3 backbone
python train.py --data data/widerface.yaml \
                --cfg models/yolov5s-mobilenet.yaml \
                --weights '' \
                --batch-size 32 \
                --epochs 300 \
                --img 640
```

### 关键训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 数据集配置 | `data/widerface.yaml` |
| `--cfg` | 模型结构配置 | `models/yolov5s.yaml` |
| `--weights` | 预训练权重 | `weights/yolov5s.pt` |
| `--batch-size` | 批大小 | 16 |
| `--epochs` | 训练轮数 | 300 |
| `--img` | 输入图片尺寸 | 640 |
| `--device` | GPU设备 | `0` |

---

## 检测推理

```bash
# 检测图片
python detect_face.py --source ./armor/0033.png \
                      --weights runs/train/exp/weights/best.pt \
                      --img 640 \
                      --conf-thres 0.5 \
                      --device 0

# 检测视频
python detect_face.py --source video.mp4 \
                      --weights best.pt \
                      --img 640

# 检测摄像头
python detect_face.py --source 0 \
                      --weights best.pt
```

### 推理输出

- 检测结果保存在 `runs/detect/` 目录
- 绘制 4 个角点（红/绿/蓝/黄色圆点）+ 四边形连线
- 显示置信度和类别名称

---

## 模型导出

```bash
# 导出 ONNX
python export.py --weights best.pt --include onnx --img 640

# TensorRT 转换
cd torch2trt
python main.py
```

---

## 可用模型

| 模型 | 配置文件 | 参数量 | 适用场景 |
|------|---------|--------|---------|
| YOLOv5n | `yolov5n.yaml` | ~1.8M | 嵌入式/边缘设备 |
| YOLOv5n-0.5 | `yolov5n-0.5.yaml` | ~1M | 极致轻量 |
| YOLOv5s | `yolov5s.yaml` | ~7M | 通用场景 |
| YOLOv5m | `yolov5m.yaml` | ~21M | 高精度需求 |
| YOLOv5l | `yolov5l.yaml` | ~46M | 最高精度 |
| MobileNetV3 | `yolov5s-mobilenet.yaml` | ~3M | 移动端部署 |

---

## 工具脚本

| 脚本 | 功能 |
|------|------|
| `dataset_revise.py` | 清理数据集中图片和标签不匹配的文件 |
| `transform_labels.py` | 将4点标签转换为纯边界框标签 |
| `data/train2yolo.py` | 将训练集标注转为YOLO格式 |
| `data/val2yolo.py` | 将验证集标注转为YOLO格式 |

---

## 修改详情

所有修改位置均以 `#修改` 注释标记。完整修改说明请参阅：

**[MODIFICATION_GUIDE.md](MODIFICATION_GUIDE.md)** — 详细讲解如何将 YOLOv5-face 从人脸5点检测改造为装甲板4点检测。

---

## 参考和致谢

- 改造思路参考了这篇 CSDN 博客：[YOLOV5-face 改装甲板四点模型](https://blog.csdn.net/m0_58348465/article/details/121423964?spm=1001.2014.3001.5506)
- MobileNetV3 backbone 的实现参考了 [TAber-W/RM_4-points_yolov5](https://github.com/TAber-W/RM_4-points_yolov5) 的代码（MobileNetV3 的接入方式有很多种写法，这里只是其中一种）

- [YOLOv5](https://github.com/ultralytics/yolov5) — 目标检测框架

- [YOLOv5-Face](https://github.com/deepcam-cn/yolov5-face) — 人脸关键点检测

  
