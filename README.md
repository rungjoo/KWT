# Hallucination-Aware LLM Training

A framework for training LLMs to detect hallucinations and learn to respond with "I don't know" (IDK) when uncertain.

## Overview

This project trains LLMs to recognize when they don't know the answer to a question and respond with "I don't know" instead of hallucinating. It uses sample-weighted training based on model accuracy to apply differential learning.

## Project Structure

```
hall/
├── dataset/           # Datasets (HaluEval, MedQA, SciQ)
├── training/          # Model training scripts
├── inference/         # Inference and answer evaluation
├── analysis/          # Result analysis and visualization
├── README.md          # Project description (this file)
└── CLAUDE.md          # Claude Code guide
```

## Quick Start

### 1. Environment Setup

```bash
pip install torch transformers datasets wandb tqdm
```

### 2. Full Pipeline

```
[Data Preparation] → [Model Training] → [Inference] → [Evaluation] → [Analysis]
```

## Execution Order

### Step 1: Check Data
```bash
ls dataset/halueval/
# train.jsonl, val.jsonl, test.jsonl
```

### Step 2: Train Model
```bash
cd training

# Basic SFT training
python train_sft.py --model_path /path/to/llama --dataname halueval

# Weighted sample training (recommended)
python train_weighted.py \
    --model_path /path/to/llama \
    --dataname halueval \
    --sft_idk_weight 0.16
```

### Step 3: Run Inference
```bash
cd inference

python run_inference.py \
    --model_path /path/to/trained_model \
    --dataname halueval \
    --split test
```

### Step 4: Evaluate Answers
```bash
python check_answers.py \
    --input_file halueval/model_results.jsonl \
    --model_path /path/to/judge_model
```

### Step 5: Analyze Results
```bash
cd analysis

# Basic evaluation
python evaluate_results.py \
    --model_path /path/to/model \
    --dataname halueval

# Compute TAUC metric
python compute_tauc.py --input_file results_evaluated.json
```

## Key Features

### Sample-Weighted Training
- Apply different weights to samples based on model accuracy
- Higher weights for difficult questions (lower accuracy)
- Train `<IDK>` token for "I don't know" responses

### Evaluation Metrics
| Metric | Description |
|--------|-------------|
| Accuracy | Correctness rate |
| IDK Ratio | Ratio of IDK responses |
| TAUC | Accuracy-IDK trade-off metric |

### Supported Datasets
| Dataset | Domain |
|---------|--------|
| HaluEval | General QA |
| MedQA | Medical |
| SciQ | Science |

## Folder Details

See `README.md` in each folder for details:
- [training/README.md](training/README.md) - Training scripts
- [inference/README.md](inference/README.md) - Inference scripts
- [analysis/README.md](analysis/README.md) - Analysis scripts
- [dataset/README.md](dataset/README.md) - Datasets

## Training Strategies

### 1. Basic SFT
Train with uniform weights for all samples

### 2. Weighted Training (Recommended)
```
High accuracy → Low weight (easy questions)
Low accuracy  → High weight (hard questions)
Zero accuracy → Learn IDK response
```

### 3. SEAL
Self-Aware Learning method

## Example Results

```
Model: sample_weight_reverse_smooth_llm_idk0.16
Dataset: HaluEval

Accuracy: 72.5%
IDK Ratio: 15.3%
TAUC: 0.847
```

## Notes

- Model checkpoints are excluded from Git via `.gitignore`
- Result files (`.jsonl`, `_evaluated.json`) are also excluded
- Wandb logs are saved in `training/wandb/`
