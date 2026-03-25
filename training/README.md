# Training

Scripts for model training.

## Files

| File | Description |
|------|-------------|
| `train_sft.py` | Basic SFT (Supervised Fine-Tuning) training |
| `train_weighted.py` | Sample-weighted training with IDK token |
| `train_weighted_noidk.py` | Sample-weighted training without IDK token |
| `train_seal.py` | SEAL method training |
| `data_cur.yaml` | Data curriculum configuration |

## Training Strategies

### 1. Basic SFT (`train_sft.py`)
- Standard Supervised Fine-Tuning
- Uniform weights for all samples

### 2. Weighted Training (`train_weighted.py`)
- Apply different weights based on model accuracy
- Higher weights for difficult questions (lower accuracy)
- Train `<IDK>` token for "I don't know" responses

### 3. Weighted Training - No IDK (`train_weighted_noidk.py`)
- Same weighting strategy but without IDK token
- Supports multiple weighting strategies:
  - `reverse_smooth`: Inverse weight + smoothing
  - `smooth`: Forward weight + smoothing
  - `uniform`: Uniform weights

### 4. SEAL Training (`train_seal.py`)
- SEAL (Self-Aware Learning) method
- Self-awareness based training

## Usage

```bash
# Basic SFT training
python train_sft.py \
    --model_path /path/to/base_model \
    --data_path ../dataset/halueval/train.jsonl

# Weighted training with IDK
python train_weighted.py \
    --model_path /path/to/base_model \
    --dataname halueval \
    --sft_idk_weight 0.16

# Weighted training without IDK
python train_weighted_noidk.py \
    --model_path /path/to/base_model \
    --dataname halueval \
    --weight_strategy reverse_smooth
```

## Output

Trained models are saved under each dataset folder:
- `halueval/{model_name}/`
- `medqa/{model_name}/`
- `sciq/{model_name}/`

## Dependencies

```bash
pip install torch transformers datasets wandb
```
