
#!/bin/bash
set -e

# ============================================================
# PyTracking - Kaggle setup
# Python 3.9.12 + Conda
# ============================================================

ENV_NAME="pytracking"
CONDA_DIR="/kaggle/working/miniconda3"

echo "============================================================"
echo " PyTracking Kaggle Installation"
echo " Python 3.9.12 + Conda"
echo "============================================================"


# ============================================================
# 1. Install Miniconda
# ============================================================

echo ""
echo "[1/9] Installing Miniconda..."

if [ ! -d "$CONDA_DIR" ]; then

    wget -q \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /kaggle/working/miniconda.sh

    bash /kaggle/working/miniconda.sh \
        -b \
        -p "$CONDA_DIR"

    rm /kaggle/working/miniconda.sh

else

    echo "Miniconda already exists."

fi

## ============================================================
# 2. Initialize Conda
# ============================================================

echo ""
echo "[2/9] Initializing Conda..."

source "$CONDA_DIR/etc/profile.d/conda.sh"

conda config --set always_yes yes
conda config --set changeps1 no

echo "Accepting Anaconda Terms of Service..."

conda tos accept \
    --override-channels \
    --channel https://repo.anaconda.com/pkgs/main

conda tos accept \
    --override-channels \
    --channel https://repo.anaconda.com/pkgs/r 
# 3. Create Python 3.9.12 environment
# ============================================================

echo ""
echo "[3/9] Creating Conda environment..."

if conda env list | grep -q "^${ENV_NAME} "; then

    echo "Environment '$ENV_NAME' already exists."

else

    conda create \
        -n "$ENV_NAME" \
        python=3.9.12

fi


# ============================================================
# 4. Activate environment
# ============================================================

echo ""
echo "[4/9] Activating environment..."

conda activate "$ENV_NAME"

echo ""
echo "Python version:"
python --version

echo ""
echo "Python location:"
which python


# ============================================================
# 5. Install PyTorch
# ============================================================

echo ""
echo "[5/9] Installing PyTorch..."

# PyTorch 1.13.1 supports Python 3.9 and is much closer
# to the era/API expected by older PyTracking code.
#
# CUDA 11.7 is used instead of the original CUDA 10.0.

pip install \
    torch==1.13.1 \
    torchvision==0.14.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117


# ============================================================
# 6. Install Python dependencies
# ============================================================

echo ""
echo "[6/9] Installing Python dependencies..."

python -m pip install \
    numpy==1.23.5 \
    matplotlib==3.7.5 \
    pandas \
    tqdm \
    opencv-python-headless \
    tb-nightly \
    scikit-image \
    tikzplotlib \
    cython \
    pycocotools \
    lvis \
    ninja \
    timm==0.6.13
# ============================================================
# 7. Install tracker-specific dependencies
# ============================================================

echo ""
echo "[7/9] Installing tracker dependencies..."

pip install spatial-correlation-sampler


echo ""
echo "Installing jpeg4py..."

pip install jpeg4py || \
    echo "WARNING: jpeg4py installation failed. Continuing."

# ============================================================
# 9. Create PyTracking/LTR environment files
# ============================================================

echo ""
echo "[9/9] Creating PyTracking environment files..."


# ============================================================
# Verification
# ============================================================

echo ""
echo "============================================================"
echo " VERIFICATION"
echo "============================================================"

echo ""
echo "Python:"
python --version

echo ""
echo "Python executable:"
which python

echo ""
echo "PyTorch:"
python - <<'PY'

import torch

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
print("CUDA version    :", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))
    print("GPU count       :", torch.cuda.device_count())

PY


echo ""
echo "DiMP50:"

if [ -f "pytracking/networks/dimp50.pth" ]; then
    ls -lh pytracking/networks/dimp50.pth
    echo "OK"
else
    echo "WARNING: dimp50.pth not found"
fi


echo ""
echo "============================================================"
echo " INSTALLATION COMPLETE"
echo "============================================================"

echo ""
echo "Environment:"
echo "    $ENV_NAME"

echo ""
echo "Python:"
python --version

echo ""
echo "IMPORTANT:"
echo "The Conda environment is active only inside this script."
echo "For a Kaggle notebook cell, run:"
echo ""
echo "source $CONDA_DIR/etc/profile.d/conda.sh"
echo "conda activate $ENV_NAME"
echo ""
