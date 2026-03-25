#!/bin/bash
# =============================================================================
# Inference Scripts - All Inference Commands
# =============================================================================
# Usage: Uncomment the commands you want to run

# Model paths
BASE_MODEL="../../model/Llama-3.2-3B"
TRAINED_MODEL="../training/halueval/llama-3.2-3b/sample_weight_reverse_smooth_llm_idk0.16"
JUDGE_MODEL="../../model/Llama-3.2-3B-Instruct"

# =============================================================================
# 1. Model Inference (run_inference.py)
# =============================================================================
# Run inference on trained model

# HaluEval
# python run_inference.py \
#     --model_path $TRAINED_MODEL \
#     --dataname halueval \
#     --split test \
#     --fewshot 3 \
#     --num_samples 5 \
#     --temperature 0.7

# MedQA
# python run_inference.py \
#     --model_path $TRAINED_MODEL \
#     --dataname medqa \
#     --split test \
#     --fewshot 3 \
#     --num_samples 5

# SciQ
# python run_inference.py \
#     --model_path $TRAINED_MODEL \
#     --dataname sciq \
#     --split test \
#     --fewshot 3 \
#     --num_samples 5

# =============================================================================
# 2. Answer Checking (check_answers.py)
# =============================================================================
# LLM-based answer evaluation

# python check_answers.py \
#     --input_file halueval/llama-3.2-3b/results.jsonl \
#     --model_path $JUDGE_MODEL

# python check_answers.py \
#     --input_file medqa/llama-3.2-3b/results.jsonl \
#     --model_path $JUDGE_MODEL

# python check_answers.py \
#     --input_file sciq/llama-3.2-3b/results.jsonl \
#     --model_path $JUDGE_MODEL

# =============================================================================
# 3. Compute Statistics (compute_stats.py)
# =============================================================================
# Compute evaluation statistics

# python compute_stats.py --input_file halueval/llama-3.2-3b/results_evaluated.json
# python compute_stats.py --input_file medqa/llama-3.2-3b/results_evaluated.json
# python compute_stats.py --input_file sciq/llama-3.2-3b/results_evaluated.json

# =============================================================================
# 4. Extract Diverse Samples (extract_samples.py)
# =============================================================================
# Extract samples with diverse score ranges for annotation

# python extract_samples.py \
#     --rouge_file halueval/llama-3.2-3b/rouge_results.json \
#     --llm_file halueval/llama-3.2-3b/llm_results.json \
#     --output_file halueval/llama-3.2-3b/diverse_samples.json \
#     --num_items 100

# =============================================================================
# 5. Compare with Human Annotation (compare_human_annotation.py)
# =============================================================================
# Compare LLM evaluation with human annotation

# python compare_human_annotation.py \
#     --annotation_file halueval/llama-3.2-3b/human_annotation.json \
#     --llm_file halueval/llama-3.2-3b/llm_results.json

# =============================================================================
# 6. Compare LLM vs ROUGE (compare_llm_rouge.py)
# =============================================================================
# Compare LLM evaluation with ROUGE scores

# python compare_llm_rouge.py \
#     --llm_file halueval/llama-3.2-3b/llm_results.json \
#     --rouge_file halueval/llama-3.2-3b/rouge_results.json
