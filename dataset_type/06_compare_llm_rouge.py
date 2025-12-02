#!/usr/bin/env python3
"""
Compare LLM evaluation (as ground truth) with ROUGE threshold at question level.
Evaluate how well different ROUGE thresholds match LLM evaluation.
"""

import json
import re
from typing import Dict, List, Tuple
from collections import defaultdict


def load_evaluation_files(llm_file: str, rouge_file: str) -> Tuple[List[Dict], List[Dict]]:
    """Load LLM and ROUGE evaluation files."""
    with open(llm_file, 'r') as f:
        llm_data = json.load(f)

    with open(rouge_file, 'r') as f:
        rouge_data = json.load(f)

    return llm_data['results'], rouge_data['results']


def calculate_metrics(predictions: List[bool], ground_truth: List[bool]) -> Dict[str, float]:
    """
    Calculate precision, recall, F1, and accuracy.

    Args:
        predictions: List of predicted match/no-match (True/False)
        ground_truth: List of ground truth match/no-match (True/False)

    Returns:
        Dictionary with metrics
    """
    # Calculate confusion matrix values
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p == True and g == True)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if p == False and g == False)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p == True and g == False)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if p == False and g == True)

    # Calculate metrics
    total = len(predictions)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total': total
    }


def get_question_level_data(llm_results: List[Dict], rouge_results: List[Dict]) -> Tuple[List[bool], List[List[float]]]:
    """
    Get question-level LLM labels and ROUGE scores.
    A question is correct if at least one of its samples is correct.

    Args:
        llm_results: LLM evaluation results
        rouge_results: ROUGE evaluation results

    Returns:
        Tuple of (question_llm_labels, question_rouge_scores)
    """
    question_llm_labels = []
    question_rouge_scores = []

    for llm_result, rouge_result in zip(llm_results, rouge_results):
        # Verify questions match
        assert llm_result['question'] == rouge_result['question'], "Questions don't match!"

        # Question-level LLM label: at least one sample is correct
        llm_label = any(sample['match'] for sample in llm_result['sample_evaluations'])
        question_llm_labels.append(llm_label)

        # Collect all ROUGE scores for this question (handle None values)
        scores = []
        for sample in rouge_result['sample_evaluations']:
            score = sample.get('score', 0.0)
            scores.append(score if score is not None else 0.0)
        question_rouge_scores.append(scores)

    return question_llm_labels, question_rouge_scores


def evaluate_rouge_threshold_question_level(question_rouge_scores: List[List[float]], threshold: float) -> List[bool]:
    """
    Evaluate questions based on ROUGE threshold at question level.
    A question is correct if at least one of its samples has rouge_score >= threshold.

    Args:
        question_rouge_scores: List of lists of ROUGE scores (one list per question)
        threshold: ROUGE score threshold

    Returns:
        List of predictions (True/False) for each question
    """
    return [any(score >= threshold for score in scores) for scores in question_rouge_scores]


def find_optimal_rouge_threshold_question_level(question_rouge_scores: List[List[float]],
                                                  question_llm_labels: List[bool],
                                                  thresholds: List[float] = None) -> Tuple[float, List[Dict]]:
    """
    Find the optimal ROUGE threshold at question level.

    Args:
        question_rouge_scores: List of lists of ROUGE scores
        question_llm_labels: List of LLM labels for questions (ground truth)
        thresholds: List of thresholds to test

    Returns:
        Tuple of (best_threshold, metrics_list)
    """
    if thresholds is None:
        thresholds = [i * 0.05 for i in range(21)]  # 0.0 to 1.0 with 0.05 step

    best_threshold = None
    best_f1 = -1
    all_results = []

    for threshold in thresholds:
        rouge_predictions = evaluate_rouge_threshold_question_level(question_rouge_scores, threshold)
        metrics = calculate_metrics(rouge_predictions, question_llm_labels)

        all_results.append({
            'threshold': threshold,
            **metrics
        })

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold

    return best_threshold, all_results


def main():
    # python3 06_compare_llm_rouge.py \
    # --llm_file halueval/base_model_temp0.7_samples5_fewshot3_evaluated_llm.json \
    # --rouge_file halueval/base_model_temp0.7_samples5_fewshot3_evaluated_rouge0.6.json \
    # --output_file halueval/llm_rouge_comparison_results.json
    import argparse

    parser = argparse.ArgumentParser(description='Compare LLM evaluation with ROUGE threshold at question level')
    parser.add_argument('--llm_file', type=str, required=True,
                        help='Path to LLM evaluation file (ground truth)')
    parser.add_argument('--rouge_file', type=str, required=True,
                        help='Path to ROUGE evaluation file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output comparison results')

    args = parser.parse_args()

    # Load data
    print(f"Loading LLM evaluation from: {args.llm_file}")
    print(f"Loading ROUGE evaluation from: {args.rouge_file}")
    llm_results, rouge_results = load_evaluation_files(args.llm_file, args.rouge_file)

    print(f"\nTotal questions: {len(llm_results)}")

    # ==================== QUESTION-LEVEL ANALYSIS ====================
    print("\n" + "="*80)
    print("QUESTION-LEVEL ANALYSIS")
    print("="*80)
    print("(A question is correct if at least one of its samples is correct)")
    print("="*80)

    # Get question-level data
    question_llm_labels, question_rouge_scores = get_question_level_data(llm_results, rouge_results)

    print(f"\nTotal questions: {len(question_llm_labels)}")
    print(f"LLM labels - True: {sum(question_llm_labels)}, False: {len(question_llm_labels) - sum(question_llm_labels)}")
    print(f"LLM accuracy (question-level): {sum(question_llm_labels) / len(question_llm_labels) * 100:.2f}%")

    # Find optimal ROUGE threshold at question level
    print("\n" + "-"*80)
    print("Finding optimal ROUGE threshold (Question-Level)...")
    print("-"*80)

    thresholds = [i * 0.05 for i in range(21)]  # 0.0 to 1.0 with 0.05 step
    question_best_threshold, question_rouge_results_list = find_optimal_rouge_threshold_question_level(
        question_rouge_scores, question_llm_labels, thresholds)

    print(f"\n{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6}")
    print("-" * 80)

    for result in question_rouge_results_list:
        print(f"{result['threshold']:<10.2f} "
              f"{result['accuracy']:<10.4f} "
              f"{result['precision']:<10.4f} "
              f"{result['recall']:<10.4f} "
              f"{result['f1']:<10.4f} "
              f"{result['tp']:<6d} "
              f"{result['tn']:<6d} "
              f"{result['fp']:<6d} "
              f"{result['fn']:<6d}")

    question_best_metrics = [r for r in question_rouge_results_list if r['threshold'] == question_best_threshold][0]

    print("\n" + "-"*80)
    print(f"OPTIMAL ROUGE THRESHOLD (Question-Level): {question_best_threshold:.2f}")
    print(f"  Accuracy:  {question_best_metrics['accuracy']:.4f}")
    print(f"  Precision: {question_best_metrics['precision']:.4f}")
    print(f"  Recall:    {question_best_metrics['recall']:.4f}")
    print(f"  F1 Score:  {question_best_metrics['f1']:.4f}")
    print("-"*80)

    # Analysis for current threshold (0.6) at question level
    print("\n" + "-"*80)
    print("Current ROUGE Threshold (0.6) Analysis (Question-Level):")
    print("-"*80)
    current_threshold = 0.6
    question_current_metrics = [r for r in question_rouge_results_list if abs(r['threshold'] - current_threshold) < 0.01]
    if question_current_metrics:
        question_current_metrics = question_current_metrics[0]
        print(f"  Accuracy:  {question_current_metrics['accuracy']:.4f}")
        print(f"  Precision: {question_current_metrics['precision']:.4f}")
        print(f"  Recall:    {question_current_metrics['recall']:.4f}")
        print(f"  F1 Score:  {question_current_metrics['f1']:.4f}")

        if question_best_threshold != current_threshold:
            improvement = question_best_metrics['f1'] - question_current_metrics['f1']
            print(f"\n  → Changing from 0.6 to {question_best_threshold:.2f} would improve F1 by {improvement:.4f}")
        else:
            print(f"\n  → Current threshold (0.6) is already optimal!")

    # Save results
    output_data = {
        'question_level': {
            'summary': {
                'total_questions': len(question_llm_labels),
                'llm_true_count': sum(question_llm_labels),
                'llm_false_count': len(question_llm_labels) - sum(question_llm_labels),
                'llm_accuracy': sum(question_llm_labels) / len(question_llm_labels) if question_llm_labels else 0,
                'optimal_rouge_threshold': question_best_threshold,
                'best_rouge_metrics': question_best_metrics,
                'current_threshold_0.6_metrics': question_current_metrics if question_current_metrics else None
            },
            'rouge_threshold_analysis': question_rouge_results_list
        }
    }

    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {args.output_file}")


if __name__ == '__main__':
    main()
