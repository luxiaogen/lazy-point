#!/usr/bin/env python3
"""
点标注 → YOLOv8 微调 → 自动点标注 Pipeline
==========================================
用法:
    # 1. 准备数据 (labelme JSON → YOLO 格式)
    python run_pipeline.py prepare

    # 2. 训练模型
    python run_pipeline.py train

    # 3. 预测未标注图片，输出 labelme JSON
    python run_pipeline.py predict

    # 4. 一键全流程
    python run_pipeline.py all

参数可在下方 CONFIG 区域修改，也可命令行覆盖:
    python run_pipeline.py train --epochs 100 --batch 32
"""

import argparse
import glob
import json
import os
import random
import shutil
import sys
from pathlib import Path

# ============================================================================
# CONFIG — 根据实际情况修改
# ============================================================================

LABEL_DIR = "/root/ds/label"                             # 标注数据根目录 (子文件夹=类别)
RAW_DIR   = "/root/ds/lys"                               # 原始图片根目录 (子文件夹=类别)
WORK_DIR  = "/root/code"                                 # 代码目录

YOLO_DATASET_DIR = os.path.join(WORK_DIR, "yolo_dataset")
OUTPUT_DIR       = os.path.join(WORK_DIR, "predictions")
MODEL_PATH       = os.path.join(WORK_DIR, "best.pt")     # 训完的模型路径

# YOLO 训练参数
EPOCHS     = 80
BATCH_SIZE = 16          # 4090/3090 用 16 避免 OOM
IMG_SIZE   = 640
MODEL_NAME = "yolov8n"   # nano，轻量快速

# 点 → 小框的像素大小 (以点为中心扩展成正方形)
POINT_BOX_SIZE = 24      # 太小模型学不到，建议 20~32

# 预测参数
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45

# ============================================================================
# STEP 1: labelme JSON → YOLO 格式
# ============================================================================

def discover_categories():
    """自动发现 LABEL_DIR 下的类别文件夹"""
    cats = []
    for name in sorted(os.listdir(LABEL_DIR)):
        d = os.path.join(LABEL_DIR, name)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.json")):
            cats.append(name)
    return cats


def labelme_to_yolo(categories):
    """
    把 labelme 点标注 JSON 转成 YOLO 格式:
      images/train/xxx.jpg
      images/val/xxx.jpg
      labels/train/xxx.txt   (class_id cx cy w h，归一化)
      labels/val/xxx.txt
    同时生成 dataset.yaml。
    """
    print("\n" + "=" * 60)
    print("STEP 1: labelme → YOLO 格式")
    print("=" * 60)

    # 类别 → id 映射
    class_map = {name: i for i, name in enumerate(categories)}
    print(f"类别映射: {class_map}")

    # 清理并创建目录
    for split in ["train", "val"]:
        os.makedirs(os.path.join(YOLO_DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATASET_DIR, "labels", split), exist_ok=True)

    all_pairs = []  # (image_path, json_path, category)

    for cat in categories:
        cat_dir = os.path.join(LABEL_DIR, cat)
        jsons = glob.glob(os.path.join(cat_dir, "*.json"))

        for jf in jsons:
            base = os.path.splitext(os.path.basename(jf))[0]
            # 找对应图片
            with open(jf) as f:
                d = json.load(f)
            img_name = d.get("imagePath", base + ".jpg")
            img_path = os.path.join(cat_dir, img_name)
            if not os.path.exists(img_path):
                for ext in [".jpg", ".jpeg", ".png"]:
                    candidate = os.path.join(cat_dir, base + ext)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
            if os.path.exists(img_path):
                all_pairs.append((img_path, jf, cat))

    print(f"共发现 {len(all_pairs)} 对 (图片+标注)")

    # 按 8:2 划分 train/val
    random.seed(42)
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * 0.8)
    splits = {
        "train": all_pairs[:split_idx],
        "val":   all_pairs[split_idx:],
    }

    for split_name, pairs in splits.items():
        img_dir = os.path.join(YOLO_DATASET_DIR, "images", split_name)
        lbl_dir = os.path.join(YOLO_DATASET_DIR, "labels", split_name)
        print(f"\n  {split_name}: {len(pairs)} 张")

        for img_path, json_path, cat in pairs:
            # 读取 JSON
            with open(json_path) as f:
                d = json.load(f)

            img_w = d.get("imageWidth") or d.get("imageSize", [0, 0])[0]
            img_h = d.get("imageHeight") or d.get("imageSize", [0, 0])[1]

            # 如果 JSON 中没有宽高，从图片读取
            if not img_w or not img_h:
                from PIL import Image
                with Image.open(img_path) as im:
                    img_w, img_h = im.size

            # 复制图片 (用类别+原名防止重名)
            safe_name = f"{cat}_{os.path.basename(img_path)}"
            dst_img = os.path.join(img_dir, safe_name)
            shutil.copy2(img_path, dst_img)

            # 生成 YOLO label
            safe_label = os.path.splitext(safe_name)[0] + ".txt"
            dst_lbl = os.path.join(lbl_dir, safe_label)
            class_id = class_map[cat]

            lines = []
            for shape in d.get("shapes", []):
                if shape.get("shape_type") != "point":
                    continue
                px, py = shape["points"][0]
                # 点 → 小框
                half = POINT_BOX_SIZE / 2
                x1 = max(0, px - half)
                y1 = max(0, py - half)
                x2 = min(img_w, px + half)
                y2 = min(img_h, py + half)
                # 归一化 YOLO 格式: cx cy w h
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines))

            print(f"    {safe_name}: {len(lines)} objects")

    # 写 dataset.yaml
    yaml_path = os.path.join(YOLO_DATASET_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {YOLO_DATASET_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"\nnc: {len(categories)}\n")
        f.write(f"names: {categories}\n")

    print(f"\n  dataset.yaml → {yaml_path}")
    return yaml_path, class_map


# ============================================================================
# STEP 2: 训练 YOLOv8
# ============================================================================

def train_model(yaml_path, device="0"):
    print("\n" + "=" * 60)
    print("STEP 2: 训练 YOLOv8")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(f"{MODEL_NAME}.pt")  # 加载预权重

    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        name="point_detector",
        project=WORK_DIR,
        exist_ok=True,
        patience=20,           # 早停耐心值
        save=True,
        device=device,         # "0" = GPU 0, "cpu" = CPU
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        augment=True,
        mosaic=False,          # 关闭，密集小目标 mosaic 会爆显存
        mixup=False,           # 关闭，同上
        copy_paste=0.3,        # 用 copy-paste 增强代替
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    # 最佳模型路径
    best = os.path.join(WORK_DIR, "point_detector", "weights", "best.pt")
    if os.path.exists(best):
        shutil.copy2(best, MODEL_PATH)
        print(f"\n  最佳模型 → {MODEL_PATH}")
    else:
        print("\n  [WARNING] 未找到 best.pt，请检查训练输出")

    return MODEL_PATH


# ============================================================================
# STEP 3: 预测 → 输出 labelme JSON
# ============================================================================

def predict_images(model_path, class_map, device="0"):
    print("\n" + "=" * 60)
    print("STEP 3: 预测未标注图片 → labelme JSON")
    print("=" * 60)

    from ultralytics import YOLO
    from PIL import Image

    model = YOLO(model_path)
    id_to_class = {v: k for k, v in class_map.items()}

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 收集所有原始图片 (排除已标注的)
    labeled_images = set()
    for cat in class_map:
        cat_dir = os.path.join(LABEL_DIR, cat)
        if os.path.isdir(cat_dir):
            for jf in glob.glob(os.path.join(cat_dir, "*.json")):
                with open(jf) as f:
                    d = json.load(f)
                labeled_images.add(d.get("imagePath", ""))

    total_predicted = 0

    for cat in class_map:
        raw_cat_dir = os.path.join(RAW_DIR, cat)
        if not os.path.isdir(raw_cat_dir):
            print(f"\n  [SKIP] 原始目录不存在: {raw_cat_dir}")
            continue

        out_cat_dir = os.path.join(OUTPUT_DIR, cat)
        os.makedirs(out_cat_dir, exist_ok=True)

        images = glob.glob(os.path.join(raw_cat_dir, "*.jpg")) + \
                 glob.glob(os.path.join(raw_cat_dir, "*.jpeg")) + \
                 glob.glob(os.path.join(raw_cat_dir, "*.png"))

        # 过滤已标注的
        unlabeled = []
        for img in images:
            name = os.path.basename(img)
            if name not in labeled_images:
                unlabeled.append(img)

        print(f"\n  {cat}: {len(unlabeled)} 张待预测 (共 {len(images)} 张, {len(images)-len(unlabeled)} 已标注)")

        if not unlabeled:
            continue

        # 批量预测
        results = model.predict(
            source=unlabeled,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            device=device,
            verbose=False,
        )

        for result, img_path in zip(results, unlabeled):
            img_name = os.path.basename(img_path)
            img_h, img_w = result.orig_shape  # YOLO 返回 (h, w)

            # 构建 labelme JSON
            shapes = []
            if result.boxes is not None:
                for box in result.boxes:
                    # 取框中心点
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    cls_id = int(box.cls[0])
                    label = id_to_class.get(cls_id, cat)

                    shapes.append({
                        "label": label,
                        "text": "",
                        "points": [[round(cx, 2), round(cy, 2)]],
                        "group_id": None,
                        "shape_type": "point",
                        "flags": {},
                    })

            labelme_json = {
                "version": "0.4.36",
                "flags": {},
                "shapes": shapes,
                "imagePath": img_name,
                "imageData": None,
                "imageHeight": img_h,
                "imageWidth": img_w,
            }

            # 保存 JSON
            json_name = os.path.splitext(img_name)[0] + ".json"
            json_path = os.path.join(out_cat_dir, json_name)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(labelme_json, f, indent=2, ensure_ascii=False)

            # 复制原图到输出目录 (方便 X-AnyLabeling 一起打开)
            shutil.copy2(img_path, os.path.join(out_cat_dir, img_name))

            total_predicted += 1
            print(f"    {img_name}: {len(shapes)} 个点")

    print(f"\n  共预测 {total_predicted} 张图片")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"\n  用 X-AnyLabeling 打开:")
    for cat in class_map:
        out_cat_dir = os.path.join(OUTPUT_DIR, cat)
        if os.path.isdir(out_cat_dir):
            print(f"    x-anylabeling {out_cat_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global EPOCHS, BATCH_SIZE, IMG_SIZE, CONF_THRESHOLD, LABEL_DIR, RAW_DIR

    parser = argparse.ArgumentParser(description="点标注 YOLOv8 Pipeline")
    parser.add_argument("step", choices=["prepare", "train", "predict", "all"],
                        help="执行步骤: prepare/train/predict/all")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--device", type=str, default="0",
                        help="设备: '0' 为 GPU 0, 'cpu' 为 CPU")
    parser.add_argument("--label-dir", type=str, default=LABEL_DIR)
    parser.add_argument("--raw-dir", type=str, default=RAW_DIR)
    parser.add_argument("--model", type=str, default=None,
                        help="predict 时指定模型路径 (默认用 best.pt)")
    args = parser.parse_args()

    # 覆盖全局配置
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    IMG_SIZE = args.img_size
    CONF_THRESHOLD = args.conf
    LABEL_DIR = args.label_dir
    RAW_DIR = args.raw_dir

    categories = discover_categories()
    if not categories:
        print(f"[ERROR] 在 {LABEL_DIR} 下未发现任何带 JSON 标注的类别文件夹")
        sys.exit(1)
    print(f"发现类别: {categories}")

    class_map = {name: i for i, name in enumerate(categories)}

    if args.step == "prepare":
        yaml_path, class_map = labelme_to_yolo(categories)
        print("\n数据准备完成。接下来: python run_pipeline.py train")

    elif args.step == "train":
        yaml_path = os.path.join(YOLO_DATASET_DIR, "dataset.yaml")
        if not os.path.exists(yaml_path):
            print("未找到 YOLO 数据集，先执行 prepare...")
            yaml_path, class_map = labelme_to_yolo(categories)
        train_model(yaml_path, device=args.device)
        print("\n训练完成。接下来: python run_pipeline.py predict")

    elif args.step == "predict":
        model_path = args.model or MODEL_PATH
        if not os.path.exists(model_path):
            print(f"[ERROR] 模型不存在: {model_path}")
            print("请先训练: python run_pipeline.py train")
            sys.exit(1)
        predict_images(model_path, class_map, device=args.device)
        print("\n预测完成。用 X-AnyLabeling 打开输出目录审查。")

    elif args.step == "all":
        yaml_path, class_map = labelme_to_yolo(categories)
        train_model(yaml_path, device=args.device)
        predict_images(MODEL_PATH, class_map, device=args.device)
        print("\n全流程完成！")


if __name__ == "__main__":
    main()
