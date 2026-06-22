#!/bin/bash
# 一键安装环境 (macOS / Linux / WSL)
# 用法: bash setup.sh

set -e

# 修复 Windows CRLF 换行问题
if command -v dos2unix &>/dev/null; then
    dos2unix "$0" 2>/dev/null || true
fi

# 检测操作系统
OS="$(uname -s)"
echo "=== 检测系统: $OS ==="

# 查找可用的 Python
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERROR] 未找到 Python 3.10+，请先安装。"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "  Python: $PYTHON ($PY_VERSION)"

# Debian/Ubuntu 下确保 python3-venv 可用
if [ "$OS" = "Linux" ]; then
    if ! $PYTHON -m venv --help &>/dev/null; then
        echo "=== 安装 python3-venv ==="
        sudo apt update && sudo apt install -y python3-venv python3-pip
    fi
fi

echo "=== 创建虚拟环境 ==="
$PYTHON -m venv .venv
source .venv/bin/activate

echo "=== 安装依赖 ==="
pip install --upgrade pip
pip install ultralytics Pillow numpy

echo ""
echo "环境就绪！使用方式:"
echo "  source .venv/bin/activate"
echo "  python run_pipeline.py all --device 0"
