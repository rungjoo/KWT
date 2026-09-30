#!/bin/bash
# =============================================================================
# Step 3. Evaluation and analysis (paper Sec. 4-5)
# =============================================================================
# Requires the checkpoints from training/run.sh.
#
# Usage:  bash analysis/run.sh
#         BASE_MODEL=Qwen/Qwen2.5-3B bash analysis/run.sh
# =============================================================================
set -e
cd "$(dirname "$0")"

BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.2-3B}
JUDGE_MODEL=${JUDGE_MODEL:-google/gemma-3-12b-it}
MODEL_NAME=$(basename "$BASE_MODEL" | tr '[:upper:]' '[:lower:]')

# <save_run_name> <data_eval_method> <sft_idk_weight>  (must match training/run.sh)
RUNS=(
    "sample_weight_reverse_smooth llm 0.16"        # KWT (LLM)
    "sample_weight_reverse_smooth rouge 0.16"      # KWT (Rouge)
    "sample_weight_reverse_smooth em 0.16"         # KWT (EM)
    "sft llm 0.0"                                  # SFT
    "popular llm 0.0"                              # FT-TOP
    "rtuning_r em 1.0"                             # R-Tuning
    "seal llm 0.16"                                # SEAL
    "sample_weight_smooth llm 1.0"                 # KWT-RF
    "sample_uniform llm 1.0"                       # KWT-U
    "sample_weighted_reverse_ridk llm 0.16"        # KWT (prepend-IDK)
    "sample_weighted_reverse_idkonly llm 0.16"     # KWT (only-IDK)
    "sample_weight_reverse_smooth_noidk llm 0.16"  # KWT (non-IDK)
)

# -----------------------------------------------------------------------------
# In-domain: generate -> judge -> metrics (Tables 5, 6, 8, 9, 11, 14, 15, 16)
# -----------------------------------------------------------------------------
for DS in halueval medqa sciq; do
    for R in "${RUNS[@]}"; do
        read -r NAME METHOD W <<< "$R"
        python evaluate_results.py --dataname $DS --base_model $BASE_MODEL \
            --save_run_name $NAME --data_eval_method $METHOD --sft_idk_weight $W
    done

    # Table 10: external knowledge in the prompt (datasets that provide it)
    if [[ $DS != medqa ]]; then
        python evaluate_results.py --dataname $DS --base_model $BASE_MODEL --method knowledge
    fi

    # Judge every generation that has not been judged yet
    for F in $DS/$MODEL_NAME/*.jsonl; do
        [[ $F == *_prob.jsonl ]] && continue
        [[ -f ${F%.jsonl}_evaluated_llm.json ]] || python check_answers.py --input_file $F --model_path $JUDGE_MODEL
    done

    echo "=== $DS ==="
    python compute_metrics.py $DS/$MODEL_NAME/*_evaluated_llm.json
    # SEAL / only-IDK count every <IDK> response as incorrect (Tables 8, 16)
    python compute_metrics.py --idk_as_incorrect \
        $DS/$MODEL_NAME/seal_evaluated_llm.json \
        $DS/$MODEL_NAME/sample_weighted_reverse_idkonly_llm_idk0.16_evaluated_llm.json
done

# -----------------------------------------------------------------------------
# Out-of-domain: NEC / RefuNQ / SelfAware with HaluEval-trained models (Tables 7, 8, 13)
# -----------------------------------------------------------------------------
for R in "sample_weight_reverse_smooth llm 0.16" "sample_weight_reverse_smooth rouge 0.16" \
         "sample_weight_reverse_smooth em 0.16" "rtuning_r em 1.0" "seal llm 0.16"; do
    read -r NAME METHOD W <<< "$R"
    python evaluate_ood.py --dataname halueval --base_model $BASE_MODEL --judge_model_path $JUDGE_MODEL \
        --save_run_name $NAME --data_eval_method $METHOD --sft_idk_weight $W
done
for OOD in RefuNQ selfAware NEC; do
    echo "=== $OOD ==="
    python compute_metrics.py --ood_dataset $OOD ood_evaluation/*_halueval_idk_results_llm.jsonl
    python compute_metrics.py --ood_dataset $OOD --idk_as_incorrect ood_evaluation/seal_halueval_idk_results_llm.jsonl
done

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
for DS in halueval medqa sciq; do
    # Figure 1: <IDK> rate by the base model's knowledge score on the test set
    python analyze_by_knowledge.py --dataset $DS --model_name $MODEL_NAME

    # Table 12: token-level KL divergence to the base model (SFT vs. KWT, SFT vs. SEAL)
    python compute_kl_divergence.py --dataname $DS --base_model $BASE_MODEL
    python compute_kl_divergence.py --dataname $DS --base_model $BASE_MODEL --save_run_name seal

    # Figures 2-3: <IDK> probability over response positions (append- vs. prepend-IDK)
    python run_idk_prob.py --dataname $DS --base_model $BASE_MODEL --save_run_name sample_weight_reverse_smooth
    python run_idk_prob.py --dataname $DS --base_model $BASE_MODEL --save_run_name sample_weighted_reverse_ridk
done
python visualize_idk_prob.py --model_name $MODEL_NAME --run_stem sample_weight_reverse_smooth_llm_idk0.16
python visualize_idk_prob.py --model_name $MODEL_NAME --run_stem sample_weighted_reverse_ridk_llm_idk0.16
