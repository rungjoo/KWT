# KWT: Knowledge-Weighted Fine-Tuning

Code for the paper **"What Models Know, How Well They Know It: Knowledge-Weighted Fine-Tuning for Learning When to Say "I Don't Know""** ([arXiv:2604.05779](https://arxiv.org/abs/2604.05779)).

KWT estimates, for every fine-tuning instance, how well the pre-trained model already knows the answer. It then uses that knowledge score to
1. weight each instance's loss by the model's knowledge (familiarity weighting), and
2. append a special `<IDK>` token to the target of instances the model does not know at all (knowledge score = 0).

The knowledge score is estimated with multi-sampled 3-shot inference (S = 5 responses per question). Each response is judged against the gold answer with EM, ROUGE-L or LLM-as-a-judge.

## Repository structure

```
KWT/
├── paths.py        # shared path conventions (datasets, knowledge files, checkpoints, results)
├── dataset/        # HaluEval / MedQA / SciQ splits, OOD sets, human annotation
├── inference/      # Step 1. knowledge estimation with the base model
├── training/       # Step 2. KWT and baselines
├── analysis/       # Step 3. evaluation, metrics and analyses
└── checkpoints/    # created by training (git-ignored)
```

All paths are resolved relative to the repository root (`paths.py`), so the scripts can be launched from any directory.

## Setup

```bash
pip install -r requirements.txt
```

The experiments use `meta-llama/Llama-3.2-3B` (and `Qwen/Qwen2.5-3B`, Appendix C) as base models and `google/gemma-3-12b-it` as the LLM judge. Every script accepts a Hugging Face id or a local path (`--base_model` / `--model_path` / `--judge_model_path`).

## Reproducing the paper

```bash
bash inference/run.sh   # 1. knowledge estimation   -> inference/<dataset>/<model>/*_evaluated_{llm,rouge*,em}.json
bash training/run.sh    # 2. fine-tuning            -> checkpoints/<dataset>/<model>/<run_name>/...
bash analysis/run.sh    # 3. evaluation & analysis  -> analysis/<dataset>/<model>/...
```

For Qwen, prefix each command with `BASE_MODEL=Qwen/Qwen2.5-3B`. Each `run.sh` lists the individual commands, which can also be run one by one. See the README in each folder for details.

### Methods and run names

Run names (`--save_run_name`) are kept identical to the ones used for the experiments, so checkpoint and result file names match.

| Paper | `--save_run_name` | Script | Knowledge / notes |
|---|---|---|---|
| **KWT** (F weighting, append-IDK) | `sample_weight_reverse_smooth` | `train_weighted.py` | `--eval_method {llm,rouge,em}` |
| KWT-RF | `sample_weight_smooth` | `train_weighted.py` | `--sft_idk_weight 1.0` |
| KWT-U | `sample_uniform` | `train_weighted.py` | `--sft_idk_weight 1.0` |
| KWT (prepend-IDK) | `sample_weighted_reverse_ridk` | `train_weighted.py` | |
| KWT (only-IDK) | `sample_weighted_reverse_idkonly` | `train_weighted.py` | |
| KWT (non-IDK) | `sample_weight_reverse_smooth` | `train_weighted_noidk.py` | saved as `..._noidk` |
| SFT | `sft` | `train_weighted.py` | |
| FT-TOP | `popular` | `train_weighted.py` | trains only on KS > 0 |
| R-Tuning | `rtuning_r` | `train_weighted.py` | greedy response + EM, `--sft_idk_weight 1.0` |
| SEAL | `seal` | `train_seal.py` | |

Sample weights (S = 5, KS = knowledge score):
- F: `w = (S·KS + 1) / (S + 1)`
- RF: `w = 1 − S·KS / (S + 1)`
- U: `w = 1`

`--sft_idk_weight` only affects `sample_uniform` and `rtuning_r`. For the other strategies it only appears in the output name (`..._idk0.16`).

### Where each result comes from

| Paper | Script |
|---|---|
| Table 2 (agreement with humans) | `inference/compare_human_annotation.py` |
| Table 3 (knowledge-score distribution) | `inference/compute_stats.py` |
| Tables 5, 6, 9, 11, 14, 15, 16 | `analysis/evaluate_results.py` → `analysis/check_answers.py` → `analysis/compute_metrics.py` |
| Table 8 (SEAL), Table 16 (only-IDK) | `compute_metrics.py --idk_as_incorrect` |
| Table 10 (knowledge in prompt) | `evaluate_results.py --method knowledge` |
| Tables 7, 13 (OOD) | `analysis/evaluate_ood.py` → `compute_metrics.py --ood_dataset {RefuNQ,selfAware,NEC}` |
| Table 12 (KL divergence) | `analysis/compute_kl_divergence.py` |
| Figure 1 (IDK rate vs. knowledge score) | `analysis/analyze_by_knowledge.py` |
| Figures 2, 3 (IDK probability by position) | `analysis/run_idk_prob.py` → `analysis/visualize_idk_prob.py` |

### Metrics

With the confusion matrix `A` (correct, with `<IDK>`), `B` (incorrect, with `<IDK>`), `C` (correct, without `<IDK>`) and `D` (incorrect, without `<IDK>`):

- **Accuracy** = (A+C) / (A+B+C+D)
- **nAUPC**: normalized area under UA-Acc(α) · CA-Acc(α) over α ∈ [0, 1], where UA-Acc(α) = (αB + C) / N and CA-Acc(α) = (C − αA) / N
- **A-FPR** = A / (A+C)
- **IDK Precision** = B / (A+B)
- **IDK Recall** = B / (B+D)

## Datasets

See [dataset/README.md](dataset/README.md). HaluEval ([Li et al., 2023](https://github.com/RUCAIBox/HaluEval)) is split 8:2 into train/test. The other datasets are MedQA ([Jin et al., 2021](https://github.com/jind11/MedQA)), SciQ ([Welbl et al., 2017](https://allenai.org/data/sciq)), NEC and RefuNQ ([Liu et al., 2024](https://github.com/genglinliu/UnknownBench)), and SelfAware ([Yin et al., 2023](https://github.com/yinzhangyue/SelfAware)). Please follow the original licenses of each dataset.

## Citation

```bibtex
@article{lee2026kwt,
  title   = {What Models Know, How Well They Know It: Knowledge-Weighted Fine-Tuning for Learning When to Say "I Don't Know"},
  author  = {Lee, Joosung and Jo, Hwiyeol and Ko, Donghyeon and Chae, Kyubyung and Park, Cheonbok and Kim, Jeonghoon},
  journal = {arXiv preprint arXiv:2604.05779},
  year    = {2026}
}
```
