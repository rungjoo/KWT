#!/bin/bash
# =============================================================================
# Step 2. Fine-tuning (paper Sec. 3.2 / 4.4, Appendix C-D)
# =============================================================================
# Requires the judged knowledge files from inference/run.sh.
# Checkpoints are written to checkpoints/<dataset>/<model>/<run_name>/... (see paths.py);
# pass --ckpt_root to store them elsewhere.
#
# Usage:  bash training/run.sh
#         BASE_MODEL=Qwen/Qwen2.5-3B bash training/run.sh
# =============================================================================
set -e
cd "$(dirname "$0")"

BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.2-3B}

for DS in halueval medqa sciq; do
    COMMON="--dataname $DS --model_path $BASE_MODEL"

    # --- KWT (familiarity weighting + append-IDK) with each matching function (Tables 5-7)
    python train_weighted.py $COMMON --save_run_name sample_weight_reverse_smooth --eval_method llm   --sft_idk_weight 0.16
    python train_weighted.py $COMMON --save_run_name sample_weight_reverse_smooth --eval_method rouge --sft_idk_weight 0.16
    python train_weighted.py $COMMON --save_run_name sample_weight_reverse_smooth --eval_method em    --sft_idk_weight 0.16

    # --- Baselines (Sec. 4.4)
    python train_weighted.py $COMMON --save_run_name sft       --sft_idk_weight 0.0   # SFT
    python train_weighted.py $COMMON --save_run_name popular   --sft_idk_weight 0.0   # FT-TOP
    python train_weighted.py $COMMON --save_run_name rtuning_r --sft_idk_weight 1.0   # R-Tuning (greedy + EM)
    python train_seal.py     $COMMON                                                  # SEAL

    # --- Weighting strategies (Sec. 4.8, Table 9)
    python train_weighted.py $COMMON --save_run_name sample_weight_smooth --eval_method llm --sft_idk_weight 1.0   # KWT-RF
    python train_weighted.py $COMMON --save_run_name sample_uniform       --eval_method llm --sft_idk_weight 1.0   # KWT-U

    # --- Position of <IDK> (Sec. 5.4, Appendix D, Table 16)
    python train_weighted.py $COMMON --save_run_name sample_weighted_reverse_ridk    --eval_method llm --sft_idk_weight 0.16  # prepend-IDK
    python train_weighted.py $COMMON --save_run_name sample_weighted_reverse_idkonly --eval_method llm --sft_idk_weight 0.16  # only-IDK

    # --- KWT without <IDK> supervision (Sec. 5.3, Table 11)
    python train_weighted_noidk.py $COMMON --save_run_name sample_weight_reverse_smooth --eval_method llm --sft_idk_weight 0.16
done
