# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hallucination detection training and evaluation framework for Large Language Models (LLMs). The project focuses on training SFT (Supervised Fine-Tuning) models and evaluating their ability to provide accurate answers without hallucinating, using QA datasets like HaluEval, MedQA, and SciQ.

## Project Structure

```
hall/
├── dataset/           # Dataset preparation and raw data
├── training/          # Model training scripts
│   ├── train_sft.py           # Standard SFT training
│   ├── train_weighted.py      # Weighted sample training with IDK
│   ├── train_weighted_noidk.py # Weighted training without IDK
│   └── train_seal.py          # SEAL method training
├── inference/         # Model inference and evaluation
│   ├── run_inference.py       # Run model inference
│   ├── check_answers.py       # LLM-based answer checking
│   ├── compute_stats.py       # Compute statistics
│   └── compare_*.py           # Comparison scripts
├── analysis/          # Result analysis and visualization
│   ├── evaluate_results.py    # Evaluate model results
│   ├── evaluate_advanced.py   # Advanced evaluation metrics
│   ├── evaluate_idk.py        # IDK response evaluation
│   ├── compute_tauc.py        # Compute TAUC metric
│   └── compute_kl_divergence.py # KL divergence analysis
└── CLAUDE.md
```

## Key Commands

### Model Training
```bash
# Train weighted sample model
python training/train_weighted.py --model_path MODEL --dataset halueval

# Train SFT model
python training/train_sft.py --model_path MODEL
```

### Model Inference
```bash
# Run inference
python inference/run_inference.py --model_path MODEL --data_path DATA
```

### Evaluation
```bash
# Evaluate model answers
python analysis/evaluate_results.py --input_file RESULTS.jsonl

# Evaluate IDK responses
python analysis/evaluate_idk.py --input_file RESULTS.jsonl
```

## Key Components

- **training/**: Model training scripts
  - `train_weighted.py`: Main training script with sample weighting
  - `train_sft.py`: Standard supervised fine-tuning

- **inference/**: Inference and answer evaluation
  - `run_inference.py`: Generate model answers
  - `check_answers.py`: LLM-based answer verification

- **analysis/**: Result analysis
  - `evaluate_results.py`: Main evaluation script
  - `compute_tauc.py`: TAUC (Truthful AUC) computation

## Evaluation Strategy

The project uses a two-stage evaluation:
1. Direct text matching (normalized)
2. LLM-based semantic equivalence checking for non-exact matches

This approach helps identify when models produce correct answers in different phrasings, reducing false negatives in accuracy measurements.
