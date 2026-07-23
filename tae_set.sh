#!/bin/bash
set -e  # 遇到错误立即退出

# ============================================
# 代理设置
# ============================================


# ============================================
# 1. 普通依赖安装
# ============================================
echo ">>> 安装普通依赖..."
pip install \
    matplotlib \
    scikit-image \
    diffusers==0.25.0 \
    loralib \
    paddlepaddle-gpu \
    paddleocr==2.7.3 \
    pyiqa

# ============================================
# 2. basicsr 特殊安装（先无依赖安装，再手动补依赖）
# ============================================
echo ">>> 安装 basicsr（无依赖模式）..."
pip install basicsr --no-deps

echo ">>> 安装 basicsr 的依赖..."
pip install \
    addict \
    future \
    lmdb \
    numpy \
    opencv-python \
    Pillow \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    torchvision \
    yapf

# ============================================
# 3. 修复 basicsr 与新版 torchvision 的兼容性问题
# ============================================
echo ">>> 修复 basicsr 兼容性..."

# 自动检测 Python 环境路径（支持 conda/virtualenv 等多种环境）
PYTHON_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

DEGRADATIONS_FILE="${PYTHON_SITE_PACKAGES}/basicsr/data/degradations.py"

if [ -f "$DEGRADATIONS_FILE" ]; then
    # 备份原文件
    cp "$DEGRADATIONS_FILE" "${DEGRADATIONS_FILE}.bak"
    
    # 执行替换
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$DEGRADATIONS_FILE"
    
    echo ">>> 已修复: $DEGRADATIONS_FILE"
    echo ">>> 原文件备份: ${DEGRADATIONS_FILE}.bak"
else
    echo ">>> 警告: 未找到 $DEGRADATIONS_FILE，请检查 basicsr 是否安装成功"
    exit 1
fi

# ============================================
# 4. 验证安装
# ============================================
echo ""
echo ">>> 验证关键包安装..."
python -c "import basicsr; print(f'basicsr 版本: {basicsr.__version__}')" || echo "basicsr 导入失败"
python -c "import torchvision; print(f'torchvision 版本: {torchvision.__version__}')" || echo "torchvision 导入失败"

echo ""
echo ">>> 安装完成！"
