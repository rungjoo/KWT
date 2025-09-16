# Hallucination Detection Model Training

This directory contains a script for training a hallucination detection model using mixed SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) losses in a single training run.

## Training Strategy

The training uses evaluation results from base_model and self_sft_model to apply different loss functions based on model performance:

1. **Both Correct** → SFT loss with right_answer
2. **Both Wrong** → SFT loss with "I don't know"
3. **Only Base Correct** → DPO loss (right_answer > self_sft wrong answer)
4. **Only SFT Correct** → DPO loss (right_answer > base wrong answer)

## Quick Start

```bash
python train_mixed_hallucination.py
```

This will:
- Load the merged evaluation results from `../dataset_llama_3.2-3b/merged/merged_evaluated.json`
- Apply SFT loss for both-correct and both-wrong cases
- Apply DPO loss for single-correct cases
- Train a single model with mixed losses

## Files

- `train_mixed_hallucination.py`: Main training script with mixed loss implementation
- `README.md`: This documentation

## Output Model

- `Llama-3.2-3B-Hallucination-Mixed/`: Trained model with mixed losses

## Requirements

Install required packages:
```bash
pip install torch transformers datasets trl tensorboard
```

## View Training Statistics

```bash
python train_hallucination_model.py --stage stats
```

## Monitor Training

View tensorboard logs:
```bash
# For SFT training
tensorboard --logdir ./Llama-3.2-3B-Hallucination-SFT/logs

# For DPO training
tensorboard --logdir ./Llama-3.2-3B-Hallucination-DPO/logs
```