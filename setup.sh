#!/bin/bash
# 一键安装环境 (在服务器上运行)
# 用法: bash setup.sh

set -e

echo "=== 创建 Python 3.12 虚拟环境 ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== 安装依赖 ==="
pip install --upgrade pip
pip install ultralytics Pillow numpy

echo ""
echo "环境就绪。使用方式:"
echo "  source .venv/bin/activate"
echo "  python run_pipeline.py all"
