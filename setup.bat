@echo off
chcp 65001 >nul 2>&1
REM 一键安装环境 (Windows CMD)
REM 用法: setup.bat

echo === 检测 Python ===
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未找到 Python，请先从 https://python.org 下载安装。
    echo 安装时请勾选 "Add Python to PATH"。
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo   Python: %PY_VER%

echo === 创建虚拟环境 ===
python -m venv .venv
call .venv\Scripts\activate.bat

echo === 安装依赖 ===
pip install --upgrade pip
pip install ultralytics Pillow numpy

echo.
echo 环境就绪！使用方式:
echo   .venv\Scripts\activate
echo   python run_pipeline.py all --device 0
pause
