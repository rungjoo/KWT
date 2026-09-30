#!/usr/bin/env python3
"""
Group test questions by the base model's knowledge score (samples_correct out of 5)
and compare SFT and KWT per group, including the KWT <IDK> rate per group
(paper Sec. 5.2, Figure 1).

Requires
  - the base-model knowledge scores on the test split:
      inference/<dataset>/<model>/test/base_model_temp0.7_samples5_fewshot3_evaluated_llm.json
      (run_inference.py --split test, then check_answers.py)
  - judged test-set results of KWT and SFT in analysis/<dataset>/<model>/
"""

import sys
import json
import argparse
import os
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RESULT_DIR, knowledge_file

def parse_args():
    parser = argparse.ArgumentParser(description='Compare SFT and KWT by base-model knowledge score')
    parser.add_argument('--dataset', type=str, default='halueval', choices=['halueval', 'medqa', 'sciq'])
    parser.add_argument('--model_name', type=str, default='llama-3.2-3b')
    parser.add_argument('--kwt_stem', type=str, default='sample_weight_reverse_smooth_llm_idk0.16',
                        help='Result stem of the KWT model')
    parser.add_argument('--sft_stem', type=str, default='sft_llm_idk0.0', help='Result stem of the SFT model')
    parser.add_argument('--output_dir', type=str, default=str(RESULT_DIR / 'idk_test_case'))
    return parser.parse_args()

args = parse_args()
dataset = args.dataset
result_dir = RESULT_DIR / dataset / args.model_name

# Load trained model results (KWT)
trained_path = result_dir / f"{args.kwt_stem}_evaluated_llm.json"
with open(trained_path, 'r') as f:
    trained_data = json.load(f)

# Load SFT model results
sft_path = result_dir / f"{args.sft_stem}_evaluated_llm.json"
with open(sft_path, 'r') as f:
    sft_data = json.load(f)

# Load base model knowledge scores on the test split
base_path = knowledge_file(dataset, args.model_name, "llm", split="test")
with open(base_path, 'r') as f:
    base_data = json.load(f)

# Create mapping from question to trained model result
trained_map = {}
for item in trained_data['results']:
    trained_map[item['question']] = item

# Create mapping from question to SFT model result
sft_map = {}
for item in sft_data['results']:
    sft_map[item['question']] = item

# Analyze by base model's samples_correct (0~5)
results_by_base_correct = defaultdict(lambda: {
    'total': 0,
    'trained_correct': 0,
    'trained_incorrect': 0,
    'trained_idk': 0,
    'correct_with_idk': 0,
    'correct_without_idk': 0,
    'incorrect_with_idk': 0,
    'incorrect_without_idk': 0,
    # SFT results
    'sft_correct': 0,
    'sft_incorrect': 0
})

for base_item in base_data['results']:
    question = base_item['question']
    samples_correct = base_item['samples_correct']

    if question in trained_map:
        trained_item = trained_map[question]
        eval_result = trained_item['evaluation']
        had_idk = trained_item.get('had_idk_token', False)

        results_by_base_correct[samples_correct]['total'] += 1
        if eval_result == 'correct':
            results_by_base_correct[samples_correct]['trained_correct'] += 1
            if had_idk:
                results_by_base_correct[samples_correct]['correct_with_idk'] += 1
            else:
                results_by_base_correct[samples_correct]['correct_without_idk'] += 1
        else:
            results_by_base_correct[samples_correct]['trained_incorrect'] += 1
            if had_idk:
                results_by_base_correct[samples_correct]['incorrect_with_idk'] += 1
            else:
                results_by_base_correct[samples_correct]['incorrect_without_idk'] += 1
        if had_idk:
            results_by_base_correct[samples_correct]['trained_idk'] += 1

    # SFT analysis
    if question in sft_map:
        sft_item = sft_map[question]
        sft_eval = sft_item['evaluation']
        if sft_eval == 'correct':
            results_by_base_correct[samples_correct]['sft_correct'] += 1
        else:
            results_by_base_correct[samples_correct]['sft_incorrect'] += 1

# Prepare output
output_lines = []

def log(text=""):
    print(text)
    output_lines.append(text)

# Print results
log("# Base Model Samples Correct vs Model Performance Comparison")
log(f"\n**Dataset: {dataset}**\n")

log("## SFT vs KWT Performance by Base Model Correctness")
log()
log("| Base Correct | Total | SFT Correct | SFT Acc (%) | KWT Correct | KWT IDK | KWT Acc (%) | Diff (KWT-SFT) |")
log("|--------------|-------|-------------|-------------|-------------|---------|-------------|----------------|")

total_all = 0
kwt_correct_all = 0
sft_correct_all = 0

for i in range(6):
    stats = results_by_base_correct[i]
    total = stats['total']
    kwt_correct = stats['trained_correct']
    kwt_idk = stats['trained_idk']
    sft_correct = stats['sft_correct']

    kwt_acc = (kwt_correct / total * 100) if total > 0 else 0
    sft_acc = (sft_correct / total * 100) if total > 0 else 0
    diff = kwt_acc - sft_acc

    total_all += total
    kwt_correct_all += kwt_correct
    sft_correct_all += sft_correct

    log(f"| {i}/5 | {total} | {sft_correct} | {sft_acc:.1f} | {kwt_correct} | {kwt_idk} | {kwt_acc:.1f} | {diff:+.1f} |")

overall_kwt_acc = (kwt_correct_all / total_all * 100) if total_all > 0 else 0
overall_sft_acc = (sft_correct_all / total_all * 100) if total_all > 0 else 0
overall_diff = overall_kwt_acc - overall_sft_acc
log(f"| **Total** | {total_all} | {sft_correct_all} | {overall_sft_acc:.1f} | {kwt_correct_all} | - | {overall_kwt_acc:.1f} | {overall_diff:+.1f} |")

log()
log("## Detailed Analysis: IDK Token Impact")
log()
log("| Base Correct | IDK Rate (%) | Correct w/ IDK | Correct w/o IDK | Incorrect w/ IDK | Incorrect w/o IDK |")
log("|--------------|--------------|----------------|-----------------|------------------|-------------------|")

for i in range(6):
    stats = results_by_base_correct[i]
    total = stats['total']
    idk = stats['trained_idk']
    c_w_idk = stats['correct_with_idk']
    c_wo_idk = stats['correct_without_idk']
    i_w_idk = stats['incorrect_with_idk']
    i_wo_idk = stats['incorrect_without_idk']

    idk_rate = (idk / total * 100) if total > 0 else 0

    log(f"| {i}/5 | {idk_rate:.1f} | {c_w_idk} | {c_wo_idk} | {i_w_idk} | {i_wo_idk} |")

log()
log("## Summary Interpretation")
log()
log("- **Base 0/5**: Questions that base model NEVER answered correctly (hardest)")
log("- **Base 5/5**: Questions that base model ALWAYS answered correctly (easiest)")
log()
log("### Observations")

# Calculate key metrics for KWT
hard_total = results_by_base_correct[0]['total'] + results_by_base_correct[1]['total']
hard_kwt_correct = results_by_base_correct[0]['trained_correct'] + results_by_base_correct[1]['trained_correct']
hard_idk = results_by_base_correct[0]['trained_idk'] + results_by_base_correct[1]['trained_idk']

easy_total = results_by_base_correct[4]['total'] + results_by_base_correct[5]['total']
easy_kwt_correct = results_by_base_correct[4]['trained_correct'] + results_by_base_correct[5]['trained_correct']
easy_idk = results_by_base_correct[4]['trained_idk'] + results_by_base_correct[5]['trained_idk']

# Calculate key metrics for SFT
hard_sft_correct = results_by_base_correct[0]['sft_correct'] + results_by_base_correct[1]['sft_correct']
easy_sft_correct = results_by_base_correct[4]['sft_correct'] + results_by_base_correct[5]['sft_correct']

log()
log("**KWT Performance:**")
log(f"- Hard questions (base 0-1/5): {hard_kwt_correct}/{hard_total} correct ({hard_kwt_correct/hard_total*100:.1f}%), IDK rate: {hard_idk/hard_total*100:.1f}%")
log(f"- Easy questions (base 4-5/5): {easy_kwt_correct}/{easy_total} correct ({easy_kwt_correct/easy_total*100:.1f}%), IDK rate: {easy_idk/easy_total*100:.1f}%")
log()
log("**SFT Performance:**")
log(f"- Hard questions (base 0-1/5): {hard_sft_correct}/{hard_total} correct ({hard_sft_correct/hard_total*100:.1f}%)")
log(f"- Easy questions (base 4-5/5): {easy_sft_correct}/{easy_total} correct ({easy_sft_correct/easy_total*100:.1f}%)")
log()
log("**SFT vs KWT Comparison:**")
hard_diff = hard_kwt_correct/hard_total*100 - hard_sft_correct/hard_total*100
easy_diff = easy_kwt_correct/easy_total*100 - easy_sft_correct/easy_total*100
log(f"- Hard questions: KWT {'+' if hard_diff >= 0 else ''}{hard_diff:.1f}% vs SFT")
log(f"- Easy questions: KWT {'+' if easy_diff >= 0 else ''}{easy_diff:.1f}% vs SFT")

# Hallucination avoidance analysis
log()
log("## Hallucination Avoidance Analysis")
log()
log("For hard questions (base 0-1/5), high IDK rate suggests model is avoiding hallucination.")
log("For easy questions (base 4-5/5), low IDK rate and high accuracy is ideal.")
log()

# Calculate avoided hallucination for each category
log("| Base Correct | Would-be Halluc. (Incorrect) | Avoided (IDK on Incorrect) | Not Avoided (Answer on Incorrect) |")
log("|--------------|------------------------------|----------------------------|-----------------------------------|")

for i in range(6):
    stats = results_by_base_correct[i]
    incorrect_total = stats['trained_incorrect']
    avoided = stats['incorrect_with_idk']  # Wrong answers but said IDK
    not_avoided = stats['incorrect_without_idk']  # Wrong answers without IDK

    avoided_rate = (avoided / incorrect_total * 100) if incorrect_total > 0 else 0
    not_avoided_rate = (not_avoided / incorrect_total * 100) if incorrect_total > 0 else 0

    log(f"| {i}/5 | {incorrect_total} | {avoided} ({avoided_rate:.1f}%) | {not_avoided} ({not_avoided_rate:.1f}%) |")

# Save results to file
os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, f"{dataset}_comparison.md")
with open(output_path, 'w') as f:
    f.write('\n'.join(output_lines))
log()
log(f"Results saved to: {output_path}")
