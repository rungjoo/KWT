#!/bin/bash
# =============================================================================
# Analysis Scripts - All Evaluation Commands
# =============================================================================
# Usage: Uncomment the commands you want to run

# Base model path
BASE_MODEL="../../model/Llama-3.2-3B"
# BASE_MODEL="../../model/Qwen2.5-3B"

# =============================================================================
# 1. Basic Evaluation (evaluate_results.py)
# =============================================================================
# Evaluates trained model on test set

# HaluEval
# python evaluate_results.py --dataname halueval --save_run_name sample_weight_reverse_smooth --sft_idk_weight 0.16 --data_eval_method llm --base_model $BASE_MODEL

# MedQA
# python evaluate_results.py --dataname medqa --save_run_name sample_weight_reverse_smooth --sft_idk_weight 0.16 --data_eval_method llm --base_model $BASE_MODEL

# SciQ
# python evaluate_results.py --dataname sciq --save_run_name sample_weight_reverse_smooth --sft_idk_weight 0.16 --data_eval_method llm --base_model $BASE_MODEL

# =============================================================================
# 2. Answer Checking (check_answers.py)
# =============================================================================
# LLM-based answer correctness evaluation

# python check_answers.py --input_file halueval/llama-3.2-3b/sample_weight_reverse_smooth_llm_idk0.16.jsonl --eval_method llm
# python check_answers.py --input_file medqa/llama-3.2-3b/sample_weight_reverse_smooth_llm_idk0.16.jsonl --eval_method llm
# python check_answers.py --input_file sciq/llama-3.2-3b/sample_weight_reverse_smooth_llm_idk0.16.jsonl --eval_method llm

# =============================================================================
# 3. Advanced Evaluation (evaluate_advanced.py)
# =============================================================================
# Sampling-based evaluation with multiple runs

# python evaluate_advanced.py --dataname halueval --save_run_name sample_weight_reverse_smooth --sft_idk_weight 0.16 --base_model $BASE_MODEL
# python evaluate_advanced.py --dataname medqa --save_run_name sample_weight_reverse_smooth --sft_idk_weight 0.16 --base_model $BASE_MODEL
# python evaluate_advanced.py --dataname sciq --save_run_name sample_weight_reverse_smooth --sft_idk_weight 0.16 --base_model $BASE_MODEL

# =============================================================================
# 4. IDK Evaluation (evaluate_idk.py)
# =============================================================================
# Evaluate IDK response quality on NEC/RefuNQ datasets

# python evaluate_idk.py --model_path /path/to/model --dataset nec
# python evaluate_idk.py --model_path /path/to/model --dataset refunq

# =============================================================================
# 5. OOD Evaluation (evaluate_ood.py)
# =============================================================================
# Evaluate on out-of-distribution datasets

# python evaluate_ood.py --model_path /path/to/model --dataset refunq
# python evaluate_ood.py --model_path /path/to/model --dataset selfaware

# =============================================================================
# 6. TAUC Computation (compute_tauc.py)
# =============================================================================
# Compute Truthful AUC metric

# python compute_tauc.py --input_file halueval/llama-3.2-3b/results_evaluated.json --alpha_max 1.0
# python compute_tauc.py --input_file medqa/llama-3.2-3b/results_evaluated.json --alpha_max 1.0
# python compute_tauc.py --input_file sciq/llama-3.2-3b/results_evaluated.json --alpha_max 1.0

# =============================================================================
# 7. KL Divergence Analysis (compute_kl_divergence.py)
# =============================================================================
# Analyze KL divergence between base and trained model

# python compute_kl_divergence.py --dataname halueval --our_model_name sample_weight_reverse_smooth_llm_idk0.16 --base_model $BASE_MODEL
# python compute_kl_divergence.py --dataname medqa --our_model_name sample_weight_reverse_smooth_llm_idk0.16 --base_model $BASE_MODEL
# python compute_kl_divergence.py --dataname sciq --our_model_name sample_weight_reverse_smooth_llm_idk0.16 --base_model $BASE_MODEL

# =============================================================================
# 8. Model Comparison (compare_models.py)
# =============================================================================
# Compare KWT vs SFT vs Base model performance

# python compare_models.py \
#     --kwt_file halueval/llama-3.2-3b/kwt_results.jsonl \
#     --sft_file halueval/llama-3.2-3b/sft_results.jsonl \
#     --base_file halueval/llama-3.2-3b/base_results.jsonl

# =============================================================================
# 9. IDK Probability Visualization (visualize_idk_prob.py)
# =============================================================================
# Visualize IDK probability sequences

# python visualize_idk_prob.py --input_file results_with_prob.jsonl --output_dir figures/

# =============================================================================
# 10. Extract IDK Correct Cases (extract_idk_correct.py)
# =============================================================================
# Extract cases correctly handled by IDK response

# python extract_idk_correct.py --input_file results.jsonl --output_file idk_correct_cases.jsonl
