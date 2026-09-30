# Inference: Knowledge Estimation

Estimates the base model's knowledge of every question (paper Sec. 3.1). `run.sh` runs the whole step.

| File | Description |
|---|---|
| `run_inference.py` | Multi-sampled 3-shot inference with the base model (S=5, T=0.7; `--greedy` for R-Tuning) |
| `check_answers.py` | Judges each sampled response with `llm` (LLM-as-a-judge), `rouge` (ROUGE-L ≥ threshold) or `em` |
| `compute_stats.py` | Distribution of knowledge scores (Table 3) |
| `extract_samples.py` | Samples responses with diverse scores for human annotation |
| `compare_human_annotation.py` | Agreement of EM / ROUGE / LLM with human labels, and the best ROUGE threshold (Table 2) |

## Usage

```bash
cd inference

# 1) 5 sampled responses per training question
python run_inference.py --dataname halueval --base_model meta-llama/Llama-3.2-3B --split train
#    -> halueval/llama-3.2-3b/base_model_temp0.7_samples5_fewshot3.jsonl

# 2) knowledge scores (samples_correct / samples_total per question)
F=halueval/llama-3.2-3b/base_model_temp0.7_samples5_fewshot3.jsonl
python check_answers.py --input_file $F --eval_method llm --model_path google/gemma-3-12b-it   # -> ..._evaluated_llm.json
python check_answers.py --input_file $F --eval_method rouge --threshold 0.35                    # -> ..._evaluated_rouge0.35.json
python check_answers.py --input_file $F --eval_method em                                        # -> ..._evaluated_em.json

# 3) R-Tuning uses a single greedy response judged by EM
python run_inference.py --dataname halueval --greedy
python check_answers.py --input_file halueval/llama-3.2-3b/base_model_greedy_samples_fewshot3.jsonl --eval_method em

# Table 3 / Table 2
python compute_stats.py --input_file halueval/llama-3.2-3b/base_model_temp0.7_samples5_fewshot3_evaluated_llm.json
python compare_human_annotation.py --input_file ../dataset/human_annotation/halueval.json
```

ROUGE-L thresholds: 0.35 for HaluEval and 0.6 for MedQA / SciQ. They were selected by agreement with human annotation.

`--split test` writes to `<dataset>/<model>/test/`. These test-set knowledge scores are only used for the analysis in Figure 1 (`analysis/analyze_by_knowledge.py`).

## Output format

`*_evaluated_<method>.json`:
```json
{
  "summary": {"eval_method": "llm", "sample_accuracy": 31.2, "...": "..."},
  "results": [
    {"question": "...", "right_answer": "...", "knowledge": "...",
     "sample_evaluations": [{"model_answer": "...", "match": true, "score": 1.0}, "..."],
     "samples_correct": 3, "samples_total": 5}
  ]
}
```
