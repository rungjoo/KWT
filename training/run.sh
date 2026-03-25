#!/bin/bash
# =============================================================================
# Training Scripts - All Training Commands
# =============================================================================
# Usage: Uncomment the commands you want to run

# Base model path
BASE_MODEL="../../model/Llama-3.2-3B"
# BASE_MODEL="../../model/Qwen2.5-3B"

# =============================================================================
# 1. Basic SFT Training (train_sft.py)
# =============================================================================
# Standard supervised fine-tuning

# HaluEval
# python train_sft.py \
#     --model_path $BASE_MODEL \
#     --dataname halueval \
#     --output_dir halueval/llama-3.2-3b/sft

# MedQA
# python train_sft.py \
#     --model_path $BASE_MODEL \
#     --dataname medqa \
#     --output_dir medqa/llama-3.2-3b/sft

# SciQ
# python train_sft.py \
#     --model_path $BASE_MODEL \
#     --dataname sciq \
#     --output_dir sciq/llama-3.2-3b/sft

# =============================================================================
# 2. Weighted Training with IDK (train_weighted.py)
# =============================================================================
# Sample-weighted training with IDK token

# HaluEval
# python train_weighted.py \
#     --model_path $BASE_MODEL \
#     --dataname halueval \
#     --sft_idk_weight 0.16 \
#     --save_run_name sample_weight_reverse_smooth

# MedQA
# python train_weighted.py \
#     --model_path $BASE_MODEL \
#     --dataname medqa \
#     --sft_idk_weight 0.16 \
#     --save_run_name sample_weight_reverse_smooth

# SciQ
# python train_weighted.py \
#     --model_path $BASE_MODEL \
#     --dataname sciq \
#     --sft_idk_weight 0.16 \
#     --save_run_name sample_weight_reverse_smooth

# =============================================================================
# 3. Weighted Training without IDK (train_weighted_noidk.py)
# =============================================================================
# Sample-weighted training without IDK token

# Weight strategies: reverse_smooth, smooth, uniform

# HaluEval
# python train_weighted_noidk.py \
#     --model_path $BASE_MODEL \
#     --dataname halueval \
#     --weight_strategy reverse_smooth \
#     --save_run_name sample_weight_reverse_smooth_noidk

# MedQA
# python train_weighted_noidk.py \
#     --model_path $BASE_MODEL \
#     --dataname medqa \
#     --weight_strategy reverse_smooth \
#     --save_run_name sample_weight_reverse_smooth_noidk

# SciQ
# python train_weighted_noidk.py \
#     --model_path $BASE_MODEL \
#     --dataname sciq \
#     --weight_strategy reverse_smooth \
#     --save_run_name sample_weight_reverse_smooth_noidk

# =============================================================================
# 4. SEAL Training (train_seal.py)
# =============================================================================
# Self-Aware Learning method

# HaluEval
# python train_seal.py \
#     --model_path $BASE_MODEL \
#     --dataname halueval \
#     --save_run_name seal

# MedQA
# python train_seal.py \
#     --model_path $BASE_MODEL \
#     --dataname medqa \
#     --save_run_name seal

# SciQ
# python train_seal.py \
#     --model_path $BASE_MODEL \
#     --dataname sciq \
#     --save_run_name seal
