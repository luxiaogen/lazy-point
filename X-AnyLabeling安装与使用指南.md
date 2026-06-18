# X-AnyLabeling 安装与使用指南

本文档介绍如何安装和使用 X-AnyLabeling 标注工具，配合 Lazy Point 工作流完成点标注任务。

---

## 1. 安装

### 1.1 环境要求

- Python >= 3.10
- 操作系统：macOS / Windows / Linux
- 推荐在虚拟环境中安装，避免依赖冲突

### 1.2 安装步骤

```bash
# 创建并激活虚拟环境
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 安装 X-AnyLabeling
pip install anylabeling
```

如果需要 PyQt6 支持（部分系统需要）：

```bash
pip install PyQt6
```

### 1.3 验证安装

```bash
anylabeling --version
```

---

## 2. 启动 X-AnyLabeling

### 2.1 基本启动

```bash
# 打开当前目录
anylabeling

# 打开指定文件夹
anylabeling /path/to/your/images

# 示例：打开 sheep 标注目录
anylabeling /Users/luxiaogen/Downloads/predictions/sheep
```

### 2.2 常用启动参数

```bash
# 指定标签列表
anylabeling --labels person,car,dog /path/to/images

# 指定输出目录（标注文件保存到指定位置）
anylabeling --output /path/to/output /path/to/images
```

---

## 3. 点标注操作流程

### 3.1 新建点标注

1. 打开 X-AnyLabeling，加载图片文件夹
2. 点击左侧工具栏的 **Point** 按钮（或使用快捷键）
3. 在图片上点击目标位置，创建一个点标注
4. 在弹出的对话框中输入标签名称（如 `sheep`、`person`、`container`）
5. 重复以上步骤，标注所有目标

### 3.2 编辑标注

| 操作 | 方法 |
|------|------|
| 移动点 | 选中点 → 拖拽到新位置 |
| 删除点 | 选中点 → 按 `Delete` 键 |
| 修改标签 | 双击标注 → 修改标签名 |
| 撤销 | `Ctrl + Z`（macOS: `Cmd + Z`） |
| 切换图片 | `←` / `→` 方向键，或左侧文件列表点击 |

### 3.3 保存标注

- 标注会自动保存为与图片同名的 `.json` 文件
- 也可通过 `Ctrl + S` 手动保存
- 标注文件与图片放在同一目录下

---

## 4. JSON 标注格式

X-AnyLabeling 输出的 JSON 格式兼容 Labelme，结构如下：

```json
{
  "version": "0.4.36",
  "flags": {},
  "shapes": [
    {
      "label": "sheep",
      "text": "",
      "points": [[320.5, 150.3]],
      "group_id": null,
      "shape_type": "point",
      "flags": {}
    }
  ],
  "imagePath": "sheep_001.jpg",
  "imageData": null,
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

关键字段说明：

- `shapes[].label` — 类别名称
- `shapes[].points` — 点的坐标 `[[x, y]]`
- `shapes[].shape_type` — 固定为 `"point"`
- `imagePath` — 对应图片文件名
- `imageWidth` / `imageHeight` — 图片尺寸

---

## 5. 与 Lazy Point Pipeline 配合使用

### 5.1 人工标注种子数据

```bash
# 打开原始图片文件夹进行标注
anylabeling /path/to/lys/sheep
```

每个类别标注 20~30 张即可用于训练。

### 5.2 运行自动标注 Pipeline

```bash
# 一键完成：数据准备 → 训练 → 预测
python run_pipeline.py all --device 0 --epochs 80 --batch 16
```

### 5.3 审查和微调预测结果

```bash
# 打开预测结果目录进行人工审查
anylabeling /path/to/predictions/sheep
```

在 X-AnyLabeling 中：
- 检查是否有漏标的目标 → 补充点标注
- 检查是否有误标的目标 → 删除错误点
- 检查点的位置是否准确 → 拖拽调整

### 5.4 迭代优化（可选）

审查修正后的数据可以合并回训练集，重新训练以提升模型精度：

```bash
# 将审查后的数据复制回标注目录，重新训练
cp -r predictions/sheep/* label/sheep/
python run_pipeline.py all --device 0 --epochs 100
```

---

## 6. 常见问题

### Q: 启动后界面空白？

检查图片路径是否包含中文或特殊字符，尽量使用英文路径。

### Q: 保存的 JSON 文件在哪里？

默认与图片保存在同一目录下，文件名为 `图片名.json`。

### Q: 如何批量修改标签名？

X-AnyLabeling 本身不支持批量修改，可以用脚本处理 JSON 文件中的 `label` 字段。

### Q: 图片太大/太小怎么办？

- 大图（>4K）：建议先裁剪或缩放再标注
- 小图：直接标注即可，模型会自动适配

### Q: GPU 内存不足？

减小 batch size：
```bash
python run_pipeline.py train --batch 8
```

---

## 7. 参考链接

- X-AnyLabeling 官方仓库：https://github.com/vietanhdev/anylabeling
- Lazy Point 项目：https://github.com/luxiaogen/lazy-point
- YOLOv8 文档：https://docs.ultralytics.com/
