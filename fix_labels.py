#!/usr/bin/env python3
"""
批量修正预测 JSON 中的标签名。

因为每个类别文件夹下的图片只包含该类别的目标，
所以将所有 shape 的 label 统一改为文件夹名。

用法:
    # 修正单个类别
    python fix_labels.py predictions/sheep

    # 修正所有类别
    python fix_labels.py predictions/
"""

import argparse
import json
import os
import glob


def fix_labels_in_dir(dir_path):
    """将目录下所有 JSON 中的 label 改为文件夹名"""
    category = os.path.basename(os.path.normpath(dir_path))
    json_files = glob.glob(os.path.join(dir_path, "*.json"))

    if not json_files:
        print(f"  [{category}] 无 JSON 文件，跳过")
        return 0

    fixed_count = 0
    total_shapes = 0

    for jf in sorted(json_files):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        changed = False

        for shape in shapes:
            if shape.get("label") != category:
                shape["label"] = category
                changed = True
            total_shapes += 1

        if changed:
            with open(jf, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            fixed_count += 1

    print(f"  [{category}] {len(json_files)} 个 JSON, "
          f"{total_shapes} 个点, {fixed_count} 个文件已修正")
    return fixed_count


def main():
    parser = argparse.ArgumentParser(description="批量修正预测 JSON 标签名")
    parser.add_argument("path", help="预测目录路径，如 predictions/ 或 predictions/sheep")
    args = parser.parse_args()

    target = os.path.abspath(args.path)

    if not os.path.isdir(target):
        print(f"[ERROR] 目录不存在: {target}")
        return

    # 判断是类别目录还是父目录
    has_json = bool(glob.glob(os.path.join(target, "*.json")))

    if has_json:
        # 直接是类别目录
        print(f"修正目录: {target}")
        fix_labels_in_dir(target)
    else:
        # 父目录，遍历子文件夹
        print(f"修正所有子目录: {target}")
        subdirs = sorted([
            d for d in os.listdir(target)
            if os.path.isdir(os.path.join(target, d))
        ])
        total_fixed = 0
        for sub in subdirs:
            total_fixed += fix_labels_in_dir(os.path.join(target, sub))
        print(f"\n共修正 {total_fixed} 个文件")

    print("\n完成！")


if __name__ == "__main__":
    main()
