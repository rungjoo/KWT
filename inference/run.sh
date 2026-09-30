#!/bin/bash
# =============================================================================
# Step 1. Knowledge estimation (paper Sec. 3.1, Tables 2-3)
# =============================================================================
# Multi-sampled 3-shot inference with the base model, then judge every sampled
# response with LLM-as-a-judge / ROUGE-L / EM. The judged files are the training
# data of every method in training/.
#
# Usage:  bash inference/run.sh
#         BASE_MODEL=Qwen/Qwen2.5-3B bash inference/run.sh
# =============================================================================
set -e
cd "$(dirname "$0")"

BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.2-3B}
JUDGE_MODEL=${JUDGE_MODEL:-google/gemma-3-12b-it}
MODEL_NAME=$(basename "$BASE_MODEL" | tr '[:upper:]' '[:lower:]')
declare -A ROUGE_TH=([halueval]=0.35 [medqa]=0.6 [sciq]=0.6)

for DS in halueval medqa sciq; do
    OUT=$DS/$MODEL_NAME
    SAMPLED=$OUT/base_model_temp0.7_samples5_fewshot3.jsonl

    # 1) S=5 sampled responses per training question (temperature 0.7, resampled 3-shot demos)
    python run_inference.py --dataname $DS --base_model $BASE_MODEL --split train

    # 2) Knowledge scores with each matching function
    python check_answers.py --input_file $SAMPLED --eval_method llm --model_path $JUDGE_MODEL
    python check_answers.py --input_file $SAMPLED --eval_method rouge --threshold ${ROUGE_TH[$DS]}
    python check_answers.py --input_file $SAMPLED --eval_method em

    # 3) R-Tuning baseline: one greedy response judged by EM (known / unknown)
    python run_inference.py --dataname $DS --base_model $BASE_MODEL --greedy
    python check_answers.py --input_file $OUT/base_model_greedy_samples_fewshot3.jsonl --eval_method em

    # 4) Knowledge scores on the test split (only for the analysis in Figure 1)
    python run_inference.py --dataname $DS --base_model $BASE_MODEL --split test
    python check_answers.py --input_file $OUT/test/base_model_temp0.7_samples5_fewshot3.jsonl \
        --eval_method llm --model_path $JUDGE_MODEL

    # Table 3: distribution of knowledge scores
    for M in em rouge${ROUGE_TH[$DS]} llm; do
        python compute_stats.py --input_file $OUT/base_model_temp0.7_samples5_fewshot3_evaluated_$M.json
    done
done

# Table 2: agreement of EM / ROUGE / LLM-as-a-judge with human annotation
#   (annotated files are provided in dataset/human_annotation/;
#    extract_samples.py re-creates the un-annotated sampling if needed)
python compare_human_annotation.py --input_file ../dataset/human_annotation/halueval.json
python compare_human_annotation.py --input_file ../dataset/human_annotation/medqa.json
