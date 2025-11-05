#!/bin/bash
set -e

# 1. Conda 가상환경 생성
ENV_NAME=hugging
PYTHON_VERSION=3.10

echo "[1/3] Conda 가상환경 생성: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# 2. 가상환경 활성화
echo "[2/3] 가상환경 활성화"
# conda activate 는 bash 안에서 직접 안 먹히므로 다음과 같이 처리
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 3. 필요한 패키지 설치
echo "[3/3] 패키지 설치"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate safetensors
pip install datasets
pip install matplotlib
pip install bert-score
pip install rouge-score
pip install wandb