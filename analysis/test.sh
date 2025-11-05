#!/bin/bash

# python3 01_eval.py --dataname halueval --save_run_name sft_dpo3 --dpo_weight 1.0 --sft_idk_weight 1.0
# python3 01_eval.py --dataname medqa --save_run_name sft_dpo3 --dpo_weight 1.0 --sft_idk_weight 1.0
# python3 01_eval.py --dataname sciq --save_run_name sft_dpo3 --dpo_weight 1.0 --sft_idk_weight 1.0
# python3 02_halueval_answer_check.py --input_file halueval/llama_3.2-3b/sft_dpo3_dpo1.0_sft_idk1.0.jsonl
# python3 02_halueval_answer_check.py --input_file medqa/llama_3.2-3b/sft_dpo3_dpo1.0_sft_idk1.0.jsonl
# python3 02_halueval_answer_check.py --input_file sciq/llama_3.2-3b/sft_dpo3_dpo1.0_sft_idk1.0.jsonl

# weight 쌍 하드코딩
weight_pairs=(
    # "1.0 1.0 1.5"
    "1.0 1.0 2.0"
)

SAVE_RUN_NAME="f_sft1"
MODEL_DIR="llama_3.2-3b"
DATASETS=("halueval" "medqa" "sciq")

echo "=== Starting batch evaluations ==="

for pair in "${weight_pairs[@]}"; do
    # 문자열을 공백 기준으로 분리 → DPO_WEIGHT, SFT_IDK_WEIGHT 추출
    read -r DPO_WEIGHT SFT_IDK_WEIGHT SFT_HARDER_WEIGHT<<< "$pair"
    echo ">>> Running for DPO_WEIGHT=${DPO_WEIGHT}, SFT_IDK_WEIGHT=${SFT_IDK_WEIGHT}, SFT_HARDER_WEIGHT=${SFT_HARDER_WEIGHT}"

    for DATASET in "${DATASETS[@]}"; do
        echo "--- Evaluating dataset: ${DATASET} ---"
        
        # 1. Run evaluation
        python3 01_eval2.py \
            --dataname "${DATASET}" \
            --save_run_name "${SAVE_RUN_NAME}" \
            --dpo_weight "${DPO_WEIGHT}" \
            --sft_idk_weight "${SFT_IDK_WEIGHT}" \
            --sft_harder_weight "${SFT_HARDER_WEIGHT}"

        # 2. Run answer check
        INPUT_FILE="${DATASET}/${MODEL_DIR}/${SAVE_RUN_NAME}_dpo${DPO_WEIGHT}_sft_idk${SFT_IDK_WEIGHT}_sft_harder${SFT_HARDER_WEIGHT}.jsonl"
        echo "Checking answers for ${INPUT_FILE}"
        python3 02_halueval_answer_check.py --input_file "${INPUT_FILE}"

        echo "--- Done ${DATASET} ---"
        echo
    done

    echo ">>> Finished combination: DPO=${DPO_WEIGHT}, SFT_IDK=${SFT_IDK_WEIGHT}, SFT_HARDER_WEIGHT=${SFT_HARDER_WEIGHT}"
    echo "=================================================="
done

echo "✅ All evaluations finished!"

python3 03_merge_test_results.py --dataname halueval/llama_3.2-3b
python3 03_merge_test_results.py --dataname medqa/llama_3.2-3b
python3 03_merge_test_results.py --dataname sciq/llama_3.2-3b