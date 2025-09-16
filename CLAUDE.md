# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hallucination detection training and evaluation framework for Large Language Models (LLMs). The project focuses on training SFT (Supervised Fine-Tuning) models and evaluating their ability to provide accurate answers without hallucinating, using QA datasets like SQuAD and TriviaQA.

## Key Commands

### Data Preparation
```bash
# Download SQuAD and TriviaQA datasets
python dataset/download_datasets.py

# Convert datasets to JSONL format
python dataset/convert_to_json.py

# Prepare merged training data from both datasets
python dataset/prepare_training_data.py
```

### Model Training
```bash
# Train Llama SFT model on prepared data
python model/train_llama_sft.py
```

### Model Inference and Evaluation
```bash
# Run inference on models (base, instruct, SFT)
python inference_models.py --model_type [base|instruct|sft] --model_path PATH --data_path PATH

# Evaluate model answers using LLM judge
python answer_check.py --input_file dataset_llama_3.2-3b/MODEL_results.jsonl --model_path PATH_TO_JUDGE_MODEL

# Merge and analyze evaluation results from multiple models
python merge_evaluated_results.py
```

## High-Level Architecture

### Data Pipeline
1. **Dataset Download**: Downloads SQuAD and TriviaQA datasets using HuggingFace datasets library
2. **Format Conversion**: Converts datasets to JSONL format with question-answer pairs
3. **Data Merging**: Combines both datasets into unified training/validation sets

### Model Training Flow
1. **Base Model**: Starting point (e.g., Llama-3.2-3B)
2. **SFT Training**: Fine-tunes base model on merged QA data using supervised learning
3. **Model Checkpointing**: Saves trained models with tokenizers and configs

### Evaluation Pipeline
1. **Inference**: Generates answers from different model variants (base, instruct, SFT)
2. **Answer Checking**: Uses an LLM judge to evaluate semantic equivalence between model answers and ground truth
3. **Results Merging**: Aggregates and compares performance across models

### Key Components

- **dataset/**: Contains data preparation scripts and downloaded datasets
  - Raw datasets stored in subdirectories (squad/, trivia_qa/)
  - Merged training data (merged_train.jsonl, merged_val.jsonl)

- **model/**: Training scripts and saved model checkpoints
  - Llama-3.2-3B-SFT/: Fine-tuned model checkpoints

- **dataset_llama_3.2-3b/**: Inference results and evaluations
  - base_model_results.jsonl: Base model outputs
  - instruct_model_results.jsonl: Instruction-tuned model outputs
  - self_sft_model_results.jsonl: SFT model outputs
  - *_evaluated.json: LLM-judged evaluation results

### Evaluation Strategy

The project uses a two-stage evaluation:
1. Direct text matching (normalized)
2. LLM-based semantic equivalence checking for non-exact matches

This approach helps identify when models produce correct answers in different phrasings, reducing false negatives in accuracy measurements.