# Lazy Point

基于 YOLOv8 微调的航拍图像自动点标注工具。支持人群、羊群、集装箱等密集小目标的自动标注，输出 labelme 格式 JSON，可直接在 X-AnyLabeling 中打开审查和微调。

## 它能做什么

给定一批航拍/无人机图像，Lazy Point 自动识别图中的目标物体并标注点位（类似人群计数的效果）。你只需：

1. 把图片按类别分文件夹放好
2. 运行一行命令
3. 用 X-AnyLabeling 打开结果，人工检查和微调

## 支持的类别

当前预训练权重 `best.pt` 支持 3 类目标：

| 类别 | 标签名 | 典型场景 |
|------|--------|----------|
| 人群 | `people` | 马拉松、广场、集会等俯拍人群 |
| 羊群 | `sheep` | 牧场、草地上的羊群航拍 |
| 集装箱 | `container` | 港口、堆场的集装箱俯拍 |

如需新增类别（如太阳能板、向日葵等），参见下方"训练自己的模型"章节。

## 快速上手：直接用 best.pt 预测

如果你只想用现有权重对新图片做自动标注，不需要重新训练。

### 第一步：安装环境

```bash
# 克隆仓库
git clone https://github.com/luxiaogen/lazy-point.git
cd lazy-point

# 创建虚拟环境并安装依赖
bash setup.sh
source .venv/bin/activate
```

### 第二步：准备图片

将待标注的图片按类别放入文件夹：

```
你的数据/
├── label/          ← 留空或放入已有标注（可以为空文件夹）
│   ├── people/
│   ├── sheep/
│   └── container/
└── images/         ← 放入待预测的原始图片
    ├── people/     ← 人群图片放这里
    ├── sheep/      ← 羊群图片放这里
    └── container/  ← 集装箱图片放这里
```

每个类别文件夹里放 `.jpg` / `.png` 图片即可，无需任何标注文件。

### 第三步：修改路径

打开 `run_pipeline.py`，修改顶部 3 行路径：

```python
LABEL_DIR = "/你的数据路径/label"    # 可以为空，但目录需要存在
RAW_DIR   = "/你的数据路径/images"   # 待预测图片目录
WORK_DIR  = "/代码所在路径"          # run_pipeline.py 所在目录
```

### 第四步：运行预测

```bash
# 直接用 best.pt 预测，跳过训练
python run_pipeline.py predict --model best.pt --device 0
```

如果没有 GPU，把 `--device 0` 改成 `--device cpu`（速度会慢一些）。

### 第五步：查看结果

预测完成后，结果保存在 `predictions/` 目录：

```
predictions/
├── container/
│   ├── image001.jpg      ← 原始图片（自动复制过来的）
│   ├── image001.json     ← 对应的点标注文件
│   └── ...
├── people/
└── sheep/
```

每个 `.json` 文件包含该图片上所有检测到的点位，格式兼容 labelme/X-AnyLabeling。

## 用 X-AnyLabeling 审查和微调

预测结果需要人工检查。X-AnyLabeling 是一个可视化标注工具，可以直观地看到每个点，支持增删改操作。

### 安装 X-AnyLabeling

```bash
# 在上面的虚拟环境中直接安装
pip install anylabeling
```

### 打开预测结果

```bash
# 打开某个类别的预测结果
anylabeling predictions/sheep
anylabeling predictions/people
anylabeling predictions/container
```

### 审查操作指南

打开后你会看到图片上标有彩色圆点，每个点对应一个检测到的目标。

**浏览图片：** 用键盘 `←` `→` 方向键切换上一张/下一张，或在左侧文件列表中点击跳转。

**检查漏标：** 如果发现图片中有目标没被标上，点击工具栏的 **Point** 按钮，然后在漏标的位置点击添加一个新点。弹出的对话框中输入类别名（如 `sheep`）。

**删除误标：** 点击选中错误的点（会高亮显示），按键盘 `Delete` 键删除。

**调整位置：** 如果点的位置不够准确，拖拽点到正确位置即可。

**修改类别：** 双击标注点，可以修改标签名称。

**保存：** 标注会自动保存。也可以按 `Ctrl + S`（macOS: `Cmd + S`）手动保存。

### 审查建议

- 优先检查密集区域和模糊区域，这些地方最容易漏标或误标
- 如果某张图片特别大或特别模糊，建议先在图像处理软件中裁剪出局部区域再标注
- 审查修正后的数据可以作为新的训练数据，重新训练以提高模型精度

## 训练自己的模型

如果你的场景不在现有 3 个类别中，或者想提高特定场景的精度，可以训练自己的模型。

### 标注种子数据

1. 用 X-AnyLabeling 打开原始图片文件夹：
   ```bash
   anylabeling /你的数据/images/sheep
   ```

2. 使用 Point 工具在每个目标上点一个点，输入类别标签

3. 每个类别标注 20~30 张图即可开始训练（越多越准）

4. 标注完成后，将图片和 JSON 文件复制到 `label/` 对应类别文件夹下

### 开始训练

```bash
# 一键全流程：数据准备 → 训练 → 预测
python run_pipeline.py all --device 0 --epochs 80 --batch 16

# 或分步执行
python run_pipeline.py prepare          # 转换标注格式
python run_pipeline.py train            # 训练模型（约 5~15 分钟）
python run_pipeline.py predict          # 用新模型预测
```

训练完成后，新权重自动保存为 `best.pt`，覆盖原有文件。

### 新增类别

在 `label/` 和 `images/` 下新建类别文件夹（文件夹名即为类别名），放入标注数据和图片，重新训练即可。脚本会自动识别所有类别。

### 训练参数调整

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 80 | 训练轮数，数据少可降到 50，数据多可到 100+ |
| `--batch` | 16 | 批大小，24GB 显存建议 16，12GB 建议 8 |
| `--img-size` | 640 | 输入图像尺寸 |
| `--conf` | 0.25 | 预测置信度阈值，调低可检出更多目标，调高减少误检 |
| `--device` | 0 | GPU 设备号，`cpu` 为 CPU 模式 |
| `--label-dir` | 脚本内配置 | 标注数据路径 |
| `--raw-dir` | 脚本内配置 | 原始图片路径 |
| `--model` | best.pt | 预测时使用的模型路径 |

## 迭代优化

一个好的工作流是反复迭代的：

```
标注 20~30 张 → 训练 → 预测剩余图片 → X-AnyLabeling 审查修正
    → 修正后的数据合并回训练集 → 重新训练 → 再预测 → 再审查
    → ... 直到精度满意
```

每一轮迭代都会让模型更准，人工审查的工作量也会越来越少。

## 目录结构

```
lazy-point/
├── run_pipeline.py                      # 主脚本（数据准备 + 训练 + 预测）
├── best.pt                              # 微调后的权重（container/people/sheep）
├── yolov8n.pt                           # YOLOv8n 预训练权重（训练起点）
├── yolo26n.pt                           # YOLO26n 预训练权重（可选替代）
├── requirements.txt                     # Python 依赖
├── setup.sh                             # 一键安装脚本
├── README.md                            # 本文档
└── X-AnyLabeling安装与使用指南.md        # X-AnyLabeling 详细使用文档
```

## 权重文件说明

| 文件 | 用途 | 来源 |
|------|------|------|
| `best.pt` | 微调后的检测权重，可直接预测 container/people/sheep | 基于 250 张标注图训练 |
| `yolov8n.pt` | YOLOv8n 官方预训练权重，作为微调的起点 | Ultralytics 官方 |
| `yolo26n.pt` | YOLO26n 预训练权重，更新的网络架构 | Ultralytics 官方 |

如需使用 YOLO26n 作为训练起点，修改 `run_pipeline.py` 中的 `MODEL_NAME = "yolo26n"`。

## JSON 标注格式

输出和输入的 JSON 均为 labelme 格式：

```json
{
  "version": "0.4.36",
  "shapes": [
    {
      "label": "sheep",
      "points": [[320.5, 150.3]],
      "shape_type": "point"
    }
  ],
  "imagePath": "photo.jpg",
  "imageWidth": 1920,
  "imageHeight": 1080
}
```

## 常见问题

**Q: 没有 GPU 能用吗？**
可以，把 `--device 0` 改成 `--device cpu`，训练和预测都会慢一些但能正常运行。

**Q: 训练时显存溢出 (CUDA OOM)？**
减小 batch size：`--batch 8` 或 `--batch 4`。

**Q: 预测结果漏了很多目标？**
降低置信度阈值：`--conf 0.15`，或者增加训练数据重新训练。

**Q: 预测结果太多误检？**
提高置信度阈值：`--conf 0.35`。

**Q: 国内服务器下载依赖很慢？**
预训练权重已包含在仓库中，无需额外下载。Python 依赖可以配置国内镜像源。

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — 目标检测框架
- [X-AnyLabeling](https://github.com/vietanhdev/anylabeling) — 标注工具
