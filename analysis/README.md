# Analysis

Scripts for model evaluation and result analysis.

## Files

### Evaluation Scripts
| File | Description |
|------|-------------|
| `evaluate_results.py` | Basic model evaluation (accuracy, IDK ratio) |
| `evaluate_advanced.py` | Advanced evaluation (sampling-based analysis) |
| `evaluate_idk.py` | IDK response evaluation (NEC, RefuNQ, etc.) |
| `evaluate_ood.py` | OOD dataset evaluation (RefuNQ, SelfAware) |
| `check_answers.py` | LLM-based answer checking |

### Analysis Scripts
| File | Description |
|------|-------------|
| `compute_tauc.py` | Compute TAUC (Truthful AUC) metric |
| `compute_kl_divergence.py` | KL divergence analysis with base model |
| `compare_models.py` | Compare KWT/SFT/Base model performance |
| `visualize_idk_prob.py` | Visualize IDK probability sequences |
| `extract_idk_correct.py` | Extract cases correctly handled by IDK |

## Execution Order

### Basic Evaluation Flow
```
1. evaluate_results.py    → Basic accuracy evaluation
2. compute_tauc.py        → Compute TAUC metric
3. compare_models.py      → Model comparison (optional)
```

### IDK Analysis Flow
```
1. evaluate_idk.py        → Evaluate IDK responses
2. compute_kl_divergence.py → KL divergence analysis
3. visualize_idk_prob.py  → Visualization (optional)
```

## Usage

### 1. Basic Evaluation
```bash
python evaluate_results.py \
    --model_path /path/to/model \
    --dataname halueval \
    --result_file results.jsonl
```

### 2. Compute TAUC
```bash
python compute_tauc.py \
    --input_file evaluated_results.json \
    --alpha_max 1.0
```

**TAUC**: Metric measuring the trade-off between accuracy and IDK ratio

### 3. KL Divergence Analysis
```bash
python compute_kl_divergence.py \
    --dataname halueval \
    --our_model_name sample_weight_reverse_smooth_llm_idk0.16
```

Analyzes distribution difference with base model during IDK responses.

### 4. Model Comparison
```bash
python compare_models.py \
    --kwt_file kwt_results.jsonl \
    --sft_file sft_results.jsonl \
    --base_file base_results.jsonl
```

### 5. OOD Evaluation
```bash
python evaluate_ood.py \
    --model_path /path/to/model \
    --dataset refunq
```

Supported datasets: RefuNQ, SelfAware

### 6. IDK Probability Visualization
```bash
python visualize_idk_prob.py \
    --input_file results_with_prob.jsonl \
    --output_dir figures/
```

## Output

### Evaluation Results
- `{dataname}/{model}/results_evaluated.json`: Evaluated results
- `TAUC/{dataname}_tauc.json`: TAUC scores
- `kl_divergence/{dataname}/kl_analysis_{model}.json`: KL analysis results

### Key Metrics
| Metric | Description |
|--------|-------------|
| Accuracy | Correctness rate |
| IDK Ratio | Ratio of IDK responses |
| TAUC | Truthful AUC (accuracy-IDK trade-off) |
| KL Divergence | Distribution difference from base model |
