# Lazy Point

基于 YOLOv8 微调的自动点标注工具。用于航拍/无人机图像中的密集小目标（人群、羊群、集装箱等）自动点标注，输出 labelme 格式 JSON，可直接在 X-AnyLabeling 中打开并人工微调。

## 工作原理

1. 将人工标注的 labelme 点标注 JSON 转换为 YOLO 格式（点扩展为小框）
2. 在标注数据上微调 YOLOv8n
3. 对未标注图片自动预测，框中心点转回点标注
4. 输出 labelme 格式 JSON，兼容 X-AnyLabeling 审查

## 目录结构

```
lazy-point/
├── run_pipeline.py      # 主脚本（准备数据 + 训练 + 预测）
├── best.pt              # 预训练权重（container/people/sheep 三类别）
├── requirements.txt     # Python 依赖
├── setup.sh             # 一键安装脚本
└── README.md
```

## 环境要求

- Python >= 3.10
- NVIDIA GPU（推荐 3090/4090，24GB VRAM）
- CUDA >= 11.8

## 快速开始

### 1. 安装环境

```bash
bash setup.sh
source .venv/bin/activate
```

### 2. 准备数据

按以下结构组织数据：

```
数据根目录/
├── label/           ← 已标注数据（图片 + labelme JSON 混放）
│   ├── people/
│   ├── sheep/
│   └── container/
└── lys/             ← 原始图片（待预测）
    ├── people/
    ├── sheep/
    ├── container/
    ├── solar/
    └── sunflower/
```

JSON 标注格式（labelme / X-AnyLabeling 导出）：

```json
{
  "shapes": [
    {
      "label": "sheep",
      "points": [[320.5, 150.3]],
      "shape_type": "point"
    }
  ],
  "imagePath": "xxx.jpg",
  "imageWidth": 1920,
  "imageHeight": 1080
}
```

### 3. 修改配置

打开 `run_pipeline.py`，修改顶部的路径配置：

```python
LABEL_DIR = "/你的路径/label"    # 标注数据目录
RAW_DIR   = "/你的路径/lys"      # 原始图片目录
WORK_DIR  = "/你的路径/code"     # 代码目录
```

### 4. 运行

```bash
# 一键全流程
python run_pipeline.py all --device 0 --epochs 80 --batch 16

# 或分步执行
python run_pipeline.py prepare          # 数据转换
python run_pipeline.py train            # 训练模型
python run_pipeline.py predict          # 预测输出
```

### 5. 使用已有权重预测

如果只想用 `best.pt` 直接预测（不重新训练）：

```bash
python run_pipeline.py predict --model best.pt --device 0
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 80 | 训练轮数 |
| `--batch` | 16 | 批大小（24GB 显存建议 16） |
| `--img-size` | 640 | 输入图像尺寸 |
| `--conf` | 0.25 | 预测置信度阈值 |
| `--device` | 0 | GPU 设备号，`cpu` 为 CPU |
| `--label-dir` | 脚本内配置 | 标注数据路径 |
| `--raw-dir` | 脚本内配置 | 原始图片路径 |
| `--model` | best.pt | 预测时的模型路径 |

## 输出

预测结果在 `predictions/` 目录，按类别子文件夹，每个图片配一个 JSON：

```
predictions/
├── container/
│   ├── image1.jpg
│   ├── image1.json
│   └── ...
├── people/
└── sheep/
```

用 X-AnyLabeling 打开审查：

```bash
x-anylabeling predictions/sheep
```

## 支持的目标类别

当前 `best.pt` 支持 3 类：`container`、`people`、`sheep`。

如需新增类别，在 `label/` 下新建文件夹放入标注数据，重跑 `python run_pipeline.py all` 即可。

## 注意事项

- 训练时自动关闭 mosaic/mixup 增强，避免密集小目标导致显存溢出
- 点标注转换为 24px 小框进行训练，预测时取框中心还原为点
- 4090/3090 训练约 5-15 分钟
- 国内服务器下载预训练权重可能很慢，可手动下载 `yolov8n.pt` 放到代码目录
