# Training

Fine-tunes the base model on the judged knowledge files from `inference/`. `run.sh` trains every model in the paper.

| File | Description |
|---|---|
| `train_weighted.py` | KWT and most baselines, selected by `--save_run_name` (see table below) |
| `train_weighted_noidk.py` | KWT (non-IDK): the same weighting without `<IDK>` supervision (Table 11) |
| `train_seal.py` | SEAL baseline: token-level reallocation of target probability to `<IDK>` |

## `--save_run_name` of `train_weighted.py`

| Run name | Paper | Weight of known (KS>0) | Unknown (KS=0) |
|---|---|---|---|
| `sample_weight_reverse_smooth` | **KWT** (F) | (S·KS+1)/(S+1) | 1/(S+1), target `answer <IDK>` |
| `sample_weight_smooth` | KWT-RF | 1 − S·KS/(S+1) | 1, target `answer <IDK>` |
| `sample_uniform` | KWT-U | 1 | `--sft_idk_weight`, target `answer <IDK>` |
| `sample_weighted_reverse_ridk` | KWT (prepend-IDK) | as KWT | target `<IDK> answer` |
| `sample_weighted_reverse_idkonly` | KWT (only-IDK) | as KWT | target `<IDK>` |
| `sft` | SFT | 1 | 1, target `answer` |
| `popular` | FT-TOP | 1 | excluded |
| `rtuning_r` | R-Tuning | 1 | `--sft_idk_weight`, target `answer <IDK>` (greedy + EM knowledge) |

Other options:
- `--eval_method {llm,rouge,em}` selects the matching function used for the knowledge score.
- `--threshold` sets the ROUGE threshold. The default is 0.35 for HaluEval and 0.6 for MedQA / SciQ.

For every model with `<IDK>`, the `<IDK>` token is added to the vocabulary and initialized with the mean embedding of "I don't know". SEAL is the exception.

## Usage

```bash
cd training
python train_weighted.py --dataname halueval --model_path meta-llama/Llama-3.2-3B \
    --save_run_name sample_weight_reverse_smooth --eval_method llm --sft_idk_weight 0.16
python train_weighted_noidk.py --dataname halueval --save_run_name sample_weight_reverse_smooth
python train_seal.py --dataname halueval
```

Default hyper-parameters (all methods): 3 epochs, lr 2e-5, batch size 4 × grad-accum 8, max length 1024, bf16. Training logs go to Weights & Biases (`--wandb_project`, default `kwt`).

## Output

Checkpoints are saved under `checkpoints/` (change with `--ckpt_root`):

```
checkpoints/<dataset>/<model>/<run_name>/sw_<eval_method><threshold>_idk<sft_idk_weight>/   # KWT & variants
checkpoints/<dataset>/<model>/sft/sft                                                     # SFT
checkpoints/<dataset>/<model>/seal/seal                                                   # SEAL
checkpoints/<dataset>/<model>/<run_name>_noidk/sw_...                                     # train_weighted_noidk.py
```

The evaluation scripts in `analysis/` resolve the same paths from the same options.
