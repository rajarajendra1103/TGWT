#!/bin/bash
# ==============================================================================
# Google Colab Setup & Execution Script for TGWT (NVIDIA Tesla T4 GPU)
# Repository: https://github.com/rajarajendra1103/TGWT.git
# ==============================================================================

set -e

echo "=== 1. Checking GPU Environment ==="
nvidia-smi

echo "=== 2. Cloning TGWT Repository ==="
if [ ! -d "TGWT" ]; then
    git clone https://github.com/rajarajendra1103/TGWT.git
fi
cd TGWT

echo "=== 3. Installing Dependencies ==="
pip install -r requirements.txt

echo "=== 4. Downloading & Preparing WikiText-103 ==="
python download_data.py --cache_dir ./data_cache --seq_len 1024

echo "=== 5. Training TGWT Model on NVIDIA Tesla T4 (FP16 Mixed Precision) ==="
python train.py \
    --config configs/tgwt_colab_t4.yaml \
    --model_type tgwt \
    --max_steps 5000 \
    --batch_size 4 \
    --seq_len 1024 \
    --output_dir ./checkpoints_tgwt

echo "=== 6. Training Parameter-Matched Baseline Transformer ==="
python train.py \
    --config configs/baseline_colab_t4.yaml \
    --model_type baseline \
    --max_steps 5000 \
    --batch_size 4 \
    --seq_len 1024 \
    --output_dir ./checkpoints_baseline

echo "=== 7. Running 5-Metric Head-to-Head Comparison Benchmark ==="
python evaluate.py --seq_len 1024 --batch_size 4 --test_scaling

echo "=== ALL RUNS COMPLETED SUCCESSFULLY ==="
