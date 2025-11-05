#!/bin/bash

# IDK evaluation script for unanswerable questions
# Usage: bash false_test.sh

# Hardcoded parameters
# DATANAMES=("halueval" "medqa" "sciq")
DATANAMES=("medqa")
SAVE_RUN_NAME="sft_dpo7"
DPO_WEIGHT="1.0"
SFT_IDK_WEIGHT="1.0"
SFT_HARDER_WEIGHT="2.0"

DATA_DIR="../dataset_type"
OUTPUT_DIR="./idk_evaluation"

# Build model path
for DATANAME in "${DATANAMES[@]}"; do
    # MODEL_PATH="/mnt/frdata/rungjoo/hall/halu_model/${DATANAME}/${SAVE_RUN_NAME}/dpo${DPO_WEIGHT}_sft_idk${SFT_IDK_WEIGHT}"
    # MODEL_PATH="/mnt/frdata/rungjoo/hall/halu_model/${DATANAME}/${SAVE_RUN_NAME}/dpo${DPO_WEIGHT}_sft_idk${SFT_IDK_WEIGHT}_sft_harder${SFT_HARDER_WEIGHT}"
    MODEL_PATH="../halu_model/${DATANAME}/${SAVE_RUN_NAME}/dpo${DPO_WEIGHT}_sft_idk${SFT_IDK_WEIGHT}_sft_harder${SFT_HARDER_WEIGHT}"

    echo "----------------------------------------------"
    echo "Evaluating dataset: ${DATANAME}"
    echo "Model Path: ${MODEL_PATH}"
    echo "----------------------------------------------"

    python3 05_evaluate_idk.py \
        --model_path "${MODEL_PATH}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/${DATANAME}" \
        --train_data_name "${DATANAME}" \
        --dataset all \
        --device cuda

    echo ""
    echo "✅ Finished evaluation for ${DATANAME}"
    echo ""
done

echo ""
echo "=============================================="
echo "Evaluation completed! Check results in ${OUTPUT_DIR}"
echo "=============================================="
