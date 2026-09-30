# Dataset

```
dataset/
├── halueval/            halueval_train.jsonl (8,000) / halueval_test.jsonl (2,000)   8:2 split of HaluEval QA
├── medqa/               train.jsonl (10,178) / dev.jsonl (1,272) / test.jsonl (1,273)
├── sciq/                train.jsonl (11,679) / validation.jsonl (1,000) / test.jsonl (1,000)
├── ood/                 out-of-domain evaluation (analysis/evaluate_ood.py)
│   ├── NEC/             NEC_answerable.json / NEC_unanswerable.json (+ entity lists)
│   ├── RefuNQ/          RefuNQ_answerable.json / RefuNQ_unanswerable.json
│   └── selfAware/       train.json
└── human_annotation/    human correctness labels for 100 questions (500 responses) per dataset (Table 2)
```

Only the `train` and `test` splits are used.

## Fields

| Dataset | Question | Gold answer | Knowledge paragraph |
|---|---|---|---|
| HaluEval | `question` | `right_answer` | `knowledge` |
| MedQA | `question` | `answer` | – |
| SciQ | `question` | `correct_answer` | `support` |

The scripts normalize these to `question` / `right_answer` / `knowledge`. The knowledge paragraph is used only
- by the LLM judge (when available), and
- for knowledge-provided prompting (`evaluate_results.py --method knowledge`, Table 10).

OOD files are JSON lines with `prompt` and `label` (NEC, RefuNQ). SelfAware is a JSON array with `question`, `answer` and `answerable`.

## Human annotation

`human_annotation/<dataset>.json` is the output of `inference/extract_samples.py` (sampled responses of Llama-3.2-3B with their ROUGE and LLM-judge scores), with the `human` field of each sampled response filled by an annotator. `inference/compare_human_annotation.py` uses these files to reproduce Table 2 and to select the ROUGE-L threshold.
