# Analysis: Evaluation and Analyses

`run.sh` runs every evaluation in the paper.

| File | Description |
|---|---|
| `evaluate_results.py` | Generates answers on the in-domain test set with a fine-tuned checkpoint (`--method knowledge` for Table 10) |
| `check_answers.py` | Judges the generations (LLM-as-a-judge after removing `<IDK>`) and records `<IDK>` usage |
| `compute_metrics.py` | Accuracy, nAUPC, A-FPR, IDK Precision, IDK Recall (in-domain), and OOD IDK rates / IDK Score |
| `evaluate_ood.py` | Inference and judging on NEC / RefuNQ / SelfAware (Tables 7, 8, 13) |
| `compute_kl_divergence.py` | Token-level KL(base ‖ fine-tuned) on gold responses (Table 12) |
| `analyze_by_knowledge.py` | SFT vs. KWT and `<IDK>` rate grouped by the base model's knowledge score (Figure 1) |
| `run_idk_prob.py` | Test-set inference that stores the `<IDK>` probability at every decoding step |
| `visualize_idk_prob.py` | Plots `<IDK>` probability over response positions (Figures 2, 3) |

## In-domain evaluation

Use the same `--save_run_name / --data_eval_method / --sft_idk_weight` as in training. The checkpoint path is resolved automatically, or can be given with `--model_path`.

```bash
cd analysis
python evaluate_results.py --dataname halueval --save_run_name sample_weight_reverse_smooth --data_eval_method llm --sft_idk_weight 0.16
#   -> halueval/llama-3.2-3b/sample_weight_reverse_smooth_llm_idk0.16.jsonl
python check_answers.py --input_file halueval/llama-3.2-3b/sample_weight_reverse_smooth_llm_idk0.16.jsonl
#   -> ..._evaluated_llm.json
python compute_metrics.py halueval/llama-3.2-3b/*_evaluated_llm.json
python compute_metrics.py --idk_as_incorrect halueval/llama-3.2-3b/seal_evaluated_llm.json   # SEAL / only-IDK
```

Result file names are `<run_name>_<eval_method><threshold>_idk<w>.jsonl`, except `sft_llm_idk0.0.jsonl` and `seal.jsonl`.

## Out-of-domain evaluation

```bash
python evaluate_ood.py --dataname halueval --save_run_name sample_weight_reverse_smooth --data_eval_method llm
#   -> ood_evaluation/sample_weight_reverse_smooth_sw_llm_idk0.16_halueval_idk_results_llm.jsonl
python compute_metrics.py --ood_dataset RefuNQ ood_evaluation/*_halueval_idk_results_llm.jsonl
```

With `--ood_dataset`:
- nAUPC / A-FPR / IDK Precision are computed on the answerable questions of that dataset (Table 7).
- `IR_ans`, `IR_unans` and the IDK Score (Table 13) use both the answerable and the unanswerable questions.

## Analyses

```bash
python analyze_by_knowledge.py --dataset halueval       # needs inference/<ds>/<model>/test/..._evaluated_llm.json
python compute_kl_divergence.py --dataname halueval                           # Base vs SFT, Base vs KWT
python compute_kl_divergence.py --dataname halueval --save_run_name seal      # Base vs SFT, Base vs SEAL
python run_idk_prob.py --dataname halueval --save_run_name sample_weighted_reverse_ridk
python visualize_idk_prob.py --run_stem sample_weighted_reverse_ridk_llm_idk0.16
```

## Metrics

|  | Correct | Incorrect |
|---|---|---|
| Response w/ `<IDK>` | A | B |
| Response w/o `<IDK>` | C | D |

- Accuracy = (A+C)/(A+B+C+D)
- UA-Acc(α) = (αB+C)/(A+B+C+D)
- CA-Acc(α) = (C−αA)/(A+B+C+D)
- nAUPC = (1/α_max) ∫₀^α_max UA-Acc(α)·CA-Acc(α) dα, with α_max = 1 and percentage-scaled values
- A-FPR = A/(A+C)
- IDK Precision = B/(A+B)
- IDK Recall = B/(B+D)
