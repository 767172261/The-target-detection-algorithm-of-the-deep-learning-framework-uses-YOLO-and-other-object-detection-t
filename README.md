深度学习框架目标检测算法使用 YOLO 等对象检测技术对室内家庭火灾进行早期检测 室内火灾烟雾检测数据集的训练及应用
@[toc]
以下文字及代码仅供参考学习使用。
### 家庭火灾检测数据集信息表

#### 数据集概述

| 项目 | 内容 |
|------|------|
| **数据集名称** | 家庭火灾检测数据集（YOLO 格式） |
| **总图像数量** | 6,500 张 |
| **应用场景** | 室内火灾早期检测、烟雾与火焰识别、智能家居安防 |
| **目标类别** | 2 类：`flame`（火焰）、`smoke`（烟雾） |
| **标注格式** | `.txt`（YOLO 格式，归一化坐标） |
| **适用模型** | YOLO 系列（直接用于训练） |
| **使用限制** | 仅供非商业研究和学术使用 |

---

### 数据集划分及标注统计

| 数据集类型 | 图像数量 | 占比 | 备注 |
|------------|-----------|------|------|
| 训练集     | 3,900 张  | 60%  | 包含 `flame` 和 `smoke` 类别 |
| 验证集     | 1,300 张  | 20%  | 用于评估模型性能，调参 |
| 测试集     | 1,300 张  | 20%  | 用于最终模型评估，确保泛化能力 |
| **总计**   | **6,500 张** | **100%** | —— |

---

### 类别标签对照表

| 类别编号 | 英文名称 | 中文名称 | 描述 |
|----------|----------|----------|------|
| 0        | flame    | 火焰     | 火灾发生时产生的明亮区域 |
| 1        | smoke    | 烟雾     | 火灾初期或燃烧过程中产生的灰色或黑色气体 |

---

### 数据集特点总结

- ✅ **大规模数据集**：包含 6,500 张高清图像，适用于室内火灾早期检测任务。
- ✅ **双类别标注**：专注于火焰和烟雾两类目标，便于快速部署和应用。
- ✅ **标准格式**：提供 `.txt` 标注文件，兼容 YOLO 系列模型，可以直接用于训练。
- ✅ **科学划分**：按照 6:2:2 的比例进行训练集、验证集和测试集的划分，确保模型的有效性和泛化能力。

---

### 典型应用方向

- **智能家居安防**：实时监测室内环境中的火灾迹象，及时发出警报。
- **消防预警系统**：结合摄像头监控网络，实现自动化的火灾预警。
- **科研与教学**：适合用于目标检测算法的研究与开发，特别是在火灾检测领域。

---


```

文字描述：
火灾数据集
用于室内火灾早期检测的 YOLO 就绪数据集
家庭火灾检测数据集（YOLO 格式）
支持使用 YOLO 等对象检测技术对室内家庭火灾进行早期检测。
此数据集仅供非商业研究和学术使用。

包含 6500 张图像和相应的 YOLO 兼容格式的注释文件。数据集分为：.txt

train.zip：3,900 张图像
val.zip：1,300 张图像
test.zip：1,300 张图像
（类：flame/smoke）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1ec2e023823e4d6caf1516a3ee355222.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2bd83754c2ea4bc2ae8571b6e113fb57.jpeg)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4e02b10a64634fb3ac343f51df865c37.jpeg)
为了使用 **YOLOv8** 训练和推理一个用于室内火灾早期检测的数据集（6500张图像，包含火焰和烟雾两类标注），我们将从环境搭建、数据准备、模型训练、推理到评估提供完整的技术流程与代码。该任务旨在通过对象检测技术实现对室内家庭火灾的早期检测。

### 数据集信息

- **类别数量**: 2 (flame, smoke)
- **图像总数**: 6500 张
- **训练集**: 3900 张
- **验证集**: 1300 张
- **测试集**: 1300 张
- **标注格式**: YOLO 格式

### 目标
1. 搭建 YOLOv8 环境。
2. 配置数据集以适应 YOLOv8 训练。
3. 使用预训练模型进行迁移学习。
4. 完成模型训练、推理、评估全流程。
5. 导出模型用于部署。

---

## ✅ 一、环境配置

### 1. 安装 CUDA 驱动（GPU 加速）

确保你的 NVIDIA 显卡驱动已安装：

```bash
nvidia-smi
```

输出应显示 GPU 信息。推荐使用 **CUDA 11.8 或 12.1**。

---

### 2. 安装 Anaconda

下载并安装 [Anaconda](https://www.anaconda.com/products/distribution)（Python 3.9+ 推荐）。

---

### 3. 创建 Python 虚拟环境

```bash
conda create -n fire_detection_yolo python=3.9
conda activate fire_detection_yolo
```

---

### 4. 安装必要依赖

```bash
# 安装 PyTorch（以 CUDA 11.8 为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 YOLOv8 官方库
pip install ultralytics opencv-python numpy matplotlib tqdm scikit-learn pandas pillow

# 可选：TensorBoard 可视化
pip install tensorboard
```

验证 GPU 是否可用：

```python
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
```

---

## ✅ 二、数据集结构与 `data.yaml` 配置

假设你将使用 YOLO 格式的标注文件，并且数据集已经按照提供的比例划分（即训练集3900张，验证集1300张，测试集1300张）。

### 数据目录结构

```
fire_dataset/
├── images/
│   ├── train/      # 训练集图像
│   ├── val/        # 验证集图像
│   └── test/       # 测试集图像
├── labels/
│   ├── train/      # 对应的 .txt 文件 (YOLO 格式)
│   ├── val/
│   └── test/
└── data.yaml       # 数据配置文件
```

### 创建 `data.yaml`

在项目根目录创建 `data.yaml` 文件：

```yaml
# data.yaml
train: ./fire_dataset/images/train
val: ./fire_dataset/images/val
test: ./fire_dataset/images/test  # 支持独立测试集评估

# 类别数量
nc: 2

# 英文类别名称
names:
  - flame
  - smoke

# 中文映射（用于可视化）
names_zh:
  - 火焰
  - 烟雾
```

---

## ✅ 三、调用 YOLOv8 官方预训练模型进行训练

考虑到这是一个专门针对火灾检测的任务，建议使用 **YOLOv8m 或 YOLOv8l** 提升分类精度。

```python
from ultralytics import YOLO

# 加载官方预训练模型（推荐使用 yolov8m.pt）
model = YOLO('yolov8m.pt')  # 自动下载 ImageNet 预训练权重

# 开始训练
results = model.train(
    data='data.yaml',           # 数据配置文件
    epochs=150,                 # 训练轮数
    batch=32,                   # 根据显存调整（RTX 3090 可用 64）
    imgsz=640,                  # 输入尺寸（可尝试 800 提升小目标检测）
    optimizer='AdamW',          # 更稳定优化器
    lr0=0.001,                  # 初始学习率
    weight_decay=0.0005,
    momentum=0.937,

    # 数据增强（提升泛化能力）
    augment=True,
    hsv_h=0.015,                # 色调扰动
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,               # 随机旋转
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,                 # 左右翻转
    mosaic=1.0,                 # Mosaic 增强
    mixup=0.1,                  # MixUp
    copy_paste=0.1,             # Copy-Paste Augmentation

    # 正则化
    dropout=0.2,
    label_smoothing=0.1,        # 标签平滑，防止过拟合

    # 学习率调度
    cos_lr=True,                # 余弦退火学习率

    # 保存与日志
    project='runs/train',
    name='fire_detection_yolov8m',
    save=True,
    save_period=10,
    exist_ok=False,

    # 缓存（内存充足时开启）
    cache=True
)
```

> 注意：
> - 若显存不足，考虑改用 `yolov8s.pt` 或降低 `batch` 大小。
> - 尝试 `imgsz=800` 来提高对远处或较小火焰、烟雾的检测性能。

---

## ✅ 四、推理代码（单图 / 批量 / 视频）

### 1. 单张图像推理

```python
from ultralytics import YOLO
from PIL import Image

# 加载最佳模型
model = YOLO('runs/train/fire_detection_yolov8m/weights/best.pt')

# 推理
results = model('test_fire.jpg', conf=0.4)  # 设置置信度阈值

# 显示结果
for r in results:
    im_array = r.plot()  # 绘制边界框和标签
    im = Image.fromarray(im_array[..., ::-1])  # BGR to RGB
    im.show()
```

---

### 2. 批量图像推理

```python
results = model.predict(
    source='fire_dataset/images/test',
    save=True,
    project='runs/detect',
    name='fire_test_results',
    conf=0.4,
    imgsz=640
)
print("批量推理完成")
```

---

### 3. 视频检测（如监控视频）

```python
results = model.predict(
    source='fire_monitoring_video.mp4',
    save=True,
    project='runs/detect/video',
    name='fire_video_result',
    show=False,
    conf=0.4
)
print("视频检测完成")
```

---

### 4. 实时摄像头检测（OpenCV）

```python
import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)
    annotated_frame = results[0].plot()

    cv2.imshow('Fire Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## ✅ 五、模型评估（验证集 + 测试集）

### 1. 在验证集上评估

```python
metrics = model.val(
    data='data.yaml',
    split='val',
    batch=32,
    imgsz=640,
    save_json=False,
    project='runs/val',
    name='fire_eval'
)

print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall: {metrics.box.r:.4f}")

# 每类 AP
for i, name in enumerate(model.names):
    print(f"{name} (AP@0.5): {metrics.box.ap[i]:.4f}")
```

---

### 2. 在测试集上独立评估（最终性能）

```python
# 使用 test split 进行最终性能评估
test_metrics = model.val(
    data='data.yaml',
    split='test',
    save_json=True,
    name='test_final'
)

print(f"Test mAP@0.5: {test_metrics.box.map50:.4f}")
print(f"Test Recall: {test_metrics.box.r:.4f}")
```

---

## ✅ 六、可视化与分析

### 1. 使用中文标签显示

```python
# 修改模型类名显示为中文
model.names = ['火焰', '烟雾']

results = model('test.jpg')
results[0].plot()  # 显示中文标签
```

### 2. 查看训练曲线

```bash
tensorboard --logdir runs/train/fire_detection_yolov8m
```

---

## ✅ 七、模型导出（ONNX / TensorRT / TFLite）根据电脑系统匹配权重模型

```python
# 导出为 ONNX（通用部署）
model.export(format='onnx', dynamic=True, opset=13, imgsz=640)

# 导出为 TensorRT（NVIDIA GPU 加速）
model.export(format='engine', half=True, dynamic=True)

# 导出为 TFLite（移动端）
model.export(format='tflite', int8=True)

# 导出为 CoreML（iOS）
model.export(format='coreml')
```


