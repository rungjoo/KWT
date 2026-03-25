# Inference

Scripts for model inference and answer evaluation.

## Files

| File | Description |
|------|-------------|
| `run_inference.py` | Run model inference |
| `check_answers.py` | LLM-based answer accuracy evaluation |
| `compute_stats.py` | Compute evaluation statistics |
| `extract_samples.py` | Extract diverse score range samples |
| `compare_human_annotation.py` | Compare LLM vs human evaluation |
| `compare_llm_rouge.py` | Compare LLM vs ROUGE scores |

## Execution Order

```
1. run_inference.py      → Generate model answers
2. check_answers.py      → Evaluate generated answers
3. compute_stats.py      → Compute statistics (optional)
```

## Usage

### 1. Model Inference
```bash
python run_inference.py \
    --model_path /path/to/model \
    --dataname halueval \
    --split test \
    --fewshot 3 \
    --num_samples 5
```

**Main Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Model path | Required |
| `--dataname` | Dataset (halueval, medqa, sciq) | halueval |
| `--split` | Data split (train, val, test) | test |
| `--fewshot` | Number of few-shot examples | 3 |
| `--num_samples` | Number of sampling runs | 5 |
| `--temperature` | Sampling temperature | 0.7 |

### 2. Answer Evaluation
```bash
python check_answers.py \
    --input_file results.jsonl \
    --model_path /path/to/judge_model
```

Uses LLM Judge to evaluate model answer accuracy.

### 3. Compute Statistics
```bash
python compute_stats.py \
    --input_file results_evaluated.json
```

### 4. Extract Samples (Optional)
```bash
python extract_samples.py \
    --rouge_file rouge_results.json \
    --llm_file llm_results.json \
    --output_file diverse_samples.json
```

### 5. Compare Evaluation Methods (Optional)
```bash
# Compare with human evaluation
python compare_human_annotation.py \
    --annotation_file human_annotation.json \
    --llm_file llm_results.json

# Compare with ROUGE
python compare_llm_rouge.py \
    --llm_file llm_results.json \
    --rouge_file rouge_results.json
```

## Output Format

### Inference Results (`*.jsonl`)
```json
{
  "question": "Question content",
  "right_answer": "Correct answer",
  "model_answer": "Model response",
  "samples_correct": 3,
  "samples_total": 5
}
```

### Evaluation Results (`*_evaluated.json`)
```json
{
  "question": "Question content",
  "is_correct": true,
  "llm_judgment": "correct"
}
```
