#!/usr/bin/env python3
"""
Compare human annotation with EM, ROUGE and LLM-as-a-judge (paper Table 2), and
find the ROUGE-L threshold that maximizes agreement with human annotation.
"""

import json
import re
from typing import Dict, List, Tuple
from collections import defaultdict
from pathlib import Path


def load_human_annotation(file_path: str) -> List[Dict]:
    """Load human annotation data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['results']


def normalize_text(text: str) -> str:
    """Simple text normalization: lowercase, strip whitespace, collapse multiple spaces"""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


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


def get_annotated_samples(results: List[Dict]) -> Tuple[List[Dict], List[int]]:
    """
    Filter samples that have human annotation.

    Args:
        results: List of result dictionaries

    Returns:
        Tuple of (annotated_samples, indices)
    """
    annotated = []
    for result in results:
        annotated_evals = []
        for sample_eval in result['sample_evaluations']:
            if 'human' in sample_eval and sample_eval['human'] is not None:
                annotated_evals.append(sample_eval)

        if annotated_evals:
            result_copy = result.copy()
            result_copy['sample_evaluations'] = annotated_evals
            annotated.append(result_copy)

    return annotated


def evaluate_rouge_threshold(results: List[Dict], threshold: float) -> List[bool]:
    """
    Evaluate samples based on ROUGE threshold.

    Args:
        results: List of result dictionaries
        threshold: ROUGE score threshold

    Returns:
        List of predictions (True/False) for each sample
    """
    predictions = []
    for result in results:
        for sample_eval in result['sample_evaluations']:
            rouge_score = sample_eval.get('rouge_score', 0.0)
            predictions.append(rouge_score >= threshold)
    return predictions


def get_exact_match_predictions(results: List[Dict]) -> List[bool]:
    """
    Get exact match predictions (case-insensitive, normalized).

    Args:
        results: List of result dictionaries

    Returns:
        List of exact match predictions (True/False) for each sample
    """
    predictions = []
    for result in results:
        right_answer = normalize_text(result.get('right_answer', ''))
        for sample_eval in result['sample_evaluations']:
            model_answer = normalize_text(sample_eval.get('filtered_model_answer', ''))
            predictions.append(model_answer == right_answer)
    return predictions


def get_llm_predictions(results: List[Dict]) -> List[bool]:
    """
    Get LLM predictions.

    Args:
        results: List of result dictionaries

    Returns:
        List of LLM predictions (True/False) for each sample
    """
    predictions = []
    for result in results:
        for sample_eval in result['sample_evaluations']:
            llm_match = sample_eval.get('llm_match', False)
            predictions.append(llm_match)
    return predictions


def get_human_labels(results: List[Dict]) -> List[bool]:
    """
    Get human annotation labels.

    Args:
        results: List of result dictionaries

    Returns:
        List of human labels (True/False) for each sample
    """
    labels = []
    for result in results:
        for sample_eval in result['sample_evaluations']:
            human_label = sample_eval.get('human', None)
            labels.append(human_label)
    return labels


def evaluate_question_level_rouge(results: List[Dict], threshold: float, only_annotated: bool = False) -> List[bool]:
    """
    Evaluate questions based on ROUGE threshold at question level.
    A question is correct if at least one of its samples has rouge_score >= threshold.

    Args:
        results: List of result dictionaries
        threshold: ROUGE score threshold
        only_annotated: If True, only consider annotated samples

    Returns:
        List of predictions (True/False) for each question
    """
    predictions = []
    for result in results:
        samples = result['sample_evaluations']
        if only_annotated:
            samples = [s for s in samples if 'human' in s and s['human'] is not None]

        # Check if at least one sample has rouge_score >= threshold
        has_correct_sample = any(
            sample_eval.get('rouge_score', 0.0) >= threshold
            for sample_eval in samples
        )
        predictions.append(has_correct_sample)
    return predictions


def get_question_level_human_labels(results: List[Dict]) -> List[bool]:
    """
    Get human annotation labels at question level.
    A question is correct if at least one of its annotated samples is labeled as True by human.
    Questions with no annotated samples are labeled as None.

    Args:
        results: List of result dictionaries

    Returns:
        List of human labels (True/False/None) for each question
    """
    labels = []
    for result in results:
        # Only check annotated samples
        annotated_samples = [s for s in result['sample_evaluations']
                            if 'human' in s and s['human'] is not None]

        if not annotated_samples:
            # No annotated samples for this question
            labels.append(None)
        else:
            # Check if at least one annotated sample is labeled as True by human
            has_correct_sample = any(
                sample_eval.get('human', None) == True
                for sample_eval in annotated_samples
            )
            labels.append(has_correct_sample)
    return labels


def get_question_level_exact_match(results: List[Dict], only_annotated: bool = False) -> List[bool]:
    """
    Get exact match predictions at question level.
    A question is correct if at least one of its samples matches exactly.

    Args:
        results: List of result dictionaries
        only_annotated: If True, only consider annotated samples

    Returns:
        List of exact match predictions (True/False) for each question
    """
    predictions = []
    for result in results:
        samples = result['sample_evaluations']
        if only_annotated:
            samples = [s for s in samples if 'human' in s and s['human'] is not None]

        right_answer = normalize_text(result.get('right_answer', ''))
        # Check if at least one sample matches exactly
        has_exact_match = any(
            normalize_text(sample_eval.get('filtered_model_answer', '')) == right_answer
            for sample_eval in samples
        )
        predictions.append(has_exact_match)
    return predictions


def get_question_level_llm_predictions(results: List[Dict], only_annotated: bool = False) -> List[bool]:
    """
    Get LLM predictions at question level.
    A question is correct if at least one of its samples is predicted as correct by LLM.

    Args:
        results: List of result dictionaries
        only_annotated: If True, only consider annotated samples

    Returns:
        List of LLM predictions (True/False) for each question
    """
    predictions = []
    for result in results:
        samples = result['sample_evaluations']
        if only_annotated:
            samples = [s for s in samples if 'human' in s and s['human'] is not None]

        # Check if at least one sample is predicted as correct by LLM
        has_llm_match = any(
            sample_eval.get('llm_match', False)
            for sample_eval in samples
        )
        predictions.append(has_llm_match)
    return predictions


def find_optimal_rouge_threshold(results: List[Dict],
                                  thresholds: List[float] = None) -> Tuple[float, Dict]:
    """
    Find the optimal ROUGE threshold that maximizes F1 score with human annotation.

    Args:
        results: List of result dictionaries
        thresholds: List of thresholds to test (default: 0.0 to 1.0 with 0.05 step)

    Returns:
        Tuple of (best_threshold, metrics_dict)
    """
    if thresholds is None:
        thresholds = [i * 0.05 for i in range(21)]  # 0.0 to 1.0 with 0.05 step

    human_labels = get_human_labels(results)

    best_threshold = None
    best_f1 = -1
    all_results = []

    for threshold in thresholds:
        rouge_predictions = evaluate_rouge_threshold(results, threshold)
        metrics = calculate_metrics(rouge_predictions, human_labels)

        all_results.append({
            'threshold': threshold,
            **metrics
        })

        if metrics['accuracy'] > best_f1:
            best_f1 = metrics['accuracy']
            best_threshold = threshold

    return best_threshold, all_results


def main():
    import argparse
    # python3 compare_human_annotation.py --input_file ../dataset/human_annotation/halueval.json --output_file halueval/llama-3.2-3b/human_annotation_comparison_results.json
    parser = argparse.ArgumentParser(description='Compare human annotation with ROUGE and LLM')
    parser.add_argument('--input_file', type=str,
                        required=True,
                        help='Human annotation file (e.g. ../dataset/human_annotation/halueval.json)')
    parser.add_argument('--output_file', type=str,
                        default=None,
                        help='Path to save comparison results (optional)')

    args = parser.parse_args()

    # Load data
    print(f"Loading human annotation from: {args.input_file}")
    all_results = load_human_annotation(args.input_file)

    # Filter only annotated samples
    results = get_annotated_samples(all_results)
    human_labels = get_human_labels(results)

    print(f"\nTotal samples in file: {sum(len(r['sample_evaluations']) for r in all_results)}")
    print(f"Annotated samples: {len(human_labels)}")
    print(f"Annotation progress: {len(human_labels) / sum(len(r['sample_evaluations']) for r in all_results) * 100:.1f}%")
    print(f"Human labels - True: {sum(human_labels)}, False: {len(human_labels) - sum(human_labels)}")

    # Find optimal ROUGE threshold
    print("\n" + "="*80)
    print("Finding optimal ROUGE threshold...")
    print("="*80)

    thresholds = [i * 0.05 for i in range(21)]  # 0.0 to 1.0 with 0.05 step
    best_threshold, rouge_results = find_optimal_rouge_threshold(results, thresholds)

    print(f"\nROUGE Threshold Analysis:")
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<5} {'TN':<5} {'FP':<5} {'FN':<5}")
    print("-" * 80)

    for result in rouge_results:
        print(f"{result['threshold']:<10.2f} "
              f"{result['accuracy']:<10.4f} "
              f"{result['precision']:<10.4f} "
              f"{result['recall']:<10.4f} "
              f"{result['f1']:<10.4f} "
              f"{result['tp']:<5d} "
              f"{result['tn']:<5d} "
              f"{result['fp']:<5d} "
              f"{result['fn']:<5d}")

    print("\n" + "="*80)
    print(f"OPTIMAL ROUGE THRESHOLD: {best_threshold:.2f}")
    best_rouge_metrics = [r for r in rouge_results if r['threshold'] == best_threshold][0]
    print(f"  Accuracy:  {best_rouge_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_rouge_metrics['precision']:.4f}")
    print(f"  Recall:    {best_rouge_metrics['recall']:.4f}")
    print(f"  F1 Score:  {best_rouge_metrics['f1']:.4f}")
    print("="*80)

    # Evaluate Exact Match
    print("\n" + "="*80)
    print("Exact Match Evaluation (case-insensitive, normalized):")
    print("="*80)

    exact_match_predictions = get_exact_match_predictions(results)
    exact_match_metrics = calculate_metrics(exact_match_predictions, human_labels)

    print(f"  Accuracy:  {exact_match_metrics['accuracy']:.4f}")
    print(f"  Precision: {exact_match_metrics['precision']:.4f}")
    print(f"  Recall:    {exact_match_metrics['recall']:.4f}")
    print(f"  F1 Score:  {exact_match_metrics['f1']:.4f}")
    print(f"  TP: {exact_match_metrics['tp']}, TN: {exact_match_metrics['tn']}, FP: {exact_match_metrics['fp']}, FN: {exact_match_metrics['fn']}")
    print("="*80)

    # Evaluate LLM
    print("\n" + "="*80)
    print("LLM Evaluation Comparison with Human Annotation:")
    print("="*80)

    llm_predictions = get_llm_predictions(results)
    llm_metrics = calculate_metrics(llm_predictions, human_labels)

    print(f"  Accuracy:  {llm_metrics['accuracy']:.4f}")
    print(f"  Precision: {llm_metrics['precision']:.4f}")
    print(f"  Recall:    {llm_metrics['recall']:.4f}")
    print(f"  F1 Score:  {llm_metrics['f1']:.4f}")
    print(f"  TP: {llm_metrics['tp']}, TN: {llm_metrics['tn']}, FP: {llm_metrics['fp']}, FN: {llm_metrics['fn']}")
    print("="*80)

    # Compare ROUGE (at different thresholds) vs Exact Match vs LLM
    print("\n" + "="*80)
    print("Summary Comparison:")
    print("="*80)
    print(f"\nBest ROUGE (threshold={best_threshold:.2f}):")
    print(f"  F1 Score: {best_rouge_metrics['f1']:.4f}")
    print(f"  Accuracy: {best_rouge_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_rouge_metrics['precision']:.4f}, Recall: {best_rouge_metrics['recall']:.4f}")

    print(f"\nExact Match (case-insensitive):")
    print(f"  F1 Score: {exact_match_metrics['f1']:.4f}")
    print(f"  Accuracy: {exact_match_metrics['accuracy']:.4f}")
    print(f"  Precision: {exact_match_metrics['precision']:.4f}, Recall: {exact_match_metrics['recall']:.4f}")

    print(f"\nLLM Evaluation:")
    print(f"  F1 Score: {llm_metrics['f1']:.4f}")
    print(f"  Accuracy: {llm_metrics['accuracy']:.4f}")
    print(f"  Precision: {llm_metrics['precision']:.4f}, Recall: {llm_metrics['recall']:.4f}")

    # Find best method
    best_f1 = max(best_rouge_metrics['f1'], exact_match_metrics['f1'], llm_metrics['f1'])
    if llm_metrics['f1'] == best_f1:
        print(f"\n✓ Best method: LLM (F1: {llm_metrics['f1']:.4f})")
    elif exact_match_metrics['f1'] == best_f1:
        print(f"\n✓ Best method: Exact Match (F1: {exact_match_metrics['f1']:.4f})")
    else:
        print(f"\n✓ Best method: ROUGE threshold={best_threshold:.2f} (F1: {best_rouge_metrics['f1']:.4f})")

    # Save results
    total_samples_in_file = sum(len(r['sample_evaluations']) for r in all_results)
    annotated_samples = len(human_labels)

    # Prepare output data (will be saved after question-level analysis)
    output_data = {
        'sample_level': {
            'summary': {
                'total_samples_in_file': total_samples_in_file,
                'annotated_samples': annotated_samples,
                'annotation_progress': annotated_samples / total_samples_in_file if total_samples_in_file > 0 else 0,
                'human_true_count': sum(human_labels),
                'human_false_count': len(human_labels) - sum(human_labels),
                'optimal_rouge_threshold': best_threshold,
                'best_rouge_metrics': best_rouge_metrics,
                'exact_match_metrics': exact_match_metrics,
                'llm_metrics': llm_metrics
            },
            'rouge_threshold_analysis': rouge_results,
            'comparison': {
                'rouge_vs_llm_f1_diff': best_rouge_metrics['f1'] - llm_metrics['f1'],
                'rouge_vs_llm_accuracy_diff': best_rouge_metrics['accuracy'] - llm_metrics['accuracy'],
                'exact_match_vs_llm_f1_diff': exact_match_metrics['f1'] - llm_metrics['f1'],
                'exact_match_vs_llm_accuracy_diff': exact_match_metrics['accuracy'] - llm_metrics['accuracy']
            }
        }
    }

    # Additional analysis: Check current threshold (0.6)
    print("\n" + "="*80)
    print("Current ROUGE Threshold (0.6) Analysis:")
    print("="*80)
    current_threshold = 0.6
    current_metrics = [r for r in rouge_results if abs(r['threshold'] - current_threshold) < 0.01][0]
    print(f"  Accuracy:  {current_metrics['accuracy']:.4f}")
    print(f"  Precision: {current_metrics['precision']:.4f}")
    print(f"  Recall:    {current_metrics['recall']:.4f}")
    print(f"  F1 Score:  {current_metrics['f1']:.4f}")

    if best_threshold != current_threshold:
        improvement = best_rouge_metrics['f1'] - current_metrics['f1']
        print(f"\n  → Changing from 0.6 to {best_threshold:.2f} would improve F1 by {improvement:.4f}")
    else:
        print(f"\n  → Current threshold (0.6) is already optimal!")

    # Question-level analysis (using all_results, not just annotated)
    print("\n\n" + "="*80)
    print("QUESTION-LEVEL ANALYSIS")
    print("="*80)
    print("(A question is correct if at least one of its annotated samples is correct)")
    print("="*80)

    # Get question-level labels from all questions
    question_human_labels = get_question_level_human_labels(all_results)
    num_questions = len(all_results)

    # Filter out None labels (questions without annotations)
    annotated_question_indices = [i for i, label in enumerate(question_human_labels) if label is not None]
    annotated_question_labels = [label for label in question_human_labels if label is not None]
    num_annotated_questions = len(annotated_question_labels)

    print(f"\nTotal questions: {num_questions}")
    print(f"Questions with annotations: {num_annotated_questions}")
    print(f"Questions without annotations: {num_questions - num_annotated_questions}")
    print(f"\nAmong annotated questions:")
    print(f"  Questions with at least one correct sample (by human): {sum(annotated_question_labels)}")
    print(f"  Questions with no correct samples (by human): {num_annotated_questions - sum(annotated_question_labels)}")

    # Find optimal ROUGE threshold at question level
    print("\n" + "-"*80)
    print("Finding optimal ROUGE threshold (Question-Level)...")
    print("-"*80)

    question_best_threshold = None
    question_best_f1 = -1
    question_rouge_results = []

    for threshold in thresholds:
        # Get predictions for all questions, considering only annotated samples
        all_question_rouge_predictions = evaluate_question_level_rouge(all_results, threshold, only_annotated=True)

        # Filter to only annotated questions
        question_rouge_predictions = [all_question_rouge_predictions[i] for i in annotated_question_indices]

        question_metrics = calculate_metrics(question_rouge_predictions, annotated_question_labels)

        question_rouge_results.append({
            'threshold': threshold,
            **question_metrics
        })

        if question_metrics['f1'] > question_best_f1:
            question_best_f1 = question_metrics['f1']
            question_best_threshold = threshold

    print(f"\n{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<5} {'TN':<5} {'FP':<5} {'FN':<5}")
    print("-" * 80)

    for result in question_rouge_results:
        print(f"{result['threshold']:<10.2f} "
              f"{result['accuracy']:<10.4f} "
              f"{result['precision']:<10.4f} "
              f"{result['recall']:<10.4f} "
              f"{result['f1']:<10.4f} "
              f"{result['tp']:<5d} "
              f"{result['tn']:<5d} "
              f"{result['fp']:<5d} "
              f"{result['fn']:<5d}")

    question_best_rouge_metrics = [r for r in question_rouge_results if r['threshold'] == question_best_threshold][0]

    print("\n" + "-"*80)
    print(f"OPTIMAL ROUGE THRESHOLD (Question-Level): {question_best_threshold:.2f}")
    print(f"  Accuracy:  {question_best_rouge_metrics['accuracy']:.4f}")
    print(f"  Precision: {question_best_rouge_metrics['precision']:.4f}")
    print(f"  Recall:    {question_best_rouge_metrics['recall']:.4f}")
    print(f"  F1 Score:  {question_best_rouge_metrics['f1']:.4f}")
    print("-"*80)

    # Exact Match at question level
    print("\n" + "-"*80)
    print("Exact Match (Question-Level):")
    print("-"*80)

    all_question_exact_match_predictions = get_question_level_exact_match(all_results, only_annotated=True)
    question_exact_match_predictions = [all_question_exact_match_predictions[i] for i in annotated_question_indices]
    question_exact_match_metrics = calculate_metrics(question_exact_match_predictions, annotated_question_labels)

    print(f"  Accuracy:  {question_exact_match_metrics['accuracy']:.4f}")
    print(f"  Precision: {question_exact_match_metrics['precision']:.4f}")
    print(f"  Recall:    {question_exact_match_metrics['recall']:.4f}")
    print(f"  F1 Score:  {question_exact_match_metrics['f1']:.4f}")
    print(f"  TP: {question_exact_match_metrics['tp']}, TN: {question_exact_match_metrics['tn']}, "
          f"FP: {question_exact_match_metrics['fp']}, FN: {question_exact_match_metrics['fn']}")
    print("-"*80)

    # LLM at question level
    print("\n" + "-"*80)
    print("LLM Evaluation (Question-Level):")
    print("-"*80)

    all_question_llm_predictions = get_question_level_llm_predictions(all_results, only_annotated=True)
    question_llm_predictions = [all_question_llm_predictions[i] for i in annotated_question_indices]
    question_llm_metrics = calculate_metrics(question_llm_predictions, annotated_question_labels)

    print(f"  Accuracy:  {question_llm_metrics['accuracy']:.4f}")
    print(f"  Precision: {question_llm_metrics['precision']:.4f}")
    print(f"  Recall:    {question_llm_metrics['recall']:.4f}")
    print(f"  F1 Score:  {question_llm_metrics['f1']:.4f}")
    print(f"  TP: {question_llm_metrics['tp']}, TN: {question_llm_metrics['tn']}, "
          f"FP: {question_llm_metrics['fp']}, FN: {question_llm_metrics['fn']}")
    print("-"*80)

    # Question-level summary comparison
    print("\n" + "-"*80)
    print("Summary Comparison (Question-Level):")
    print("-"*80)
    print(f"\nBest ROUGE (threshold={question_best_threshold:.2f}):")
    print(f"  F1 Score: {question_best_rouge_metrics['f1']:.4f}")
    print(f"  Accuracy: {question_best_rouge_metrics['accuracy']:.4f}")
    print(f"  Precision: {question_best_rouge_metrics['precision']:.4f}, Recall: {question_best_rouge_metrics['recall']:.4f}")

    print(f"\nExact Match (case-insensitive):")
    print(f"  F1 Score: {question_exact_match_metrics['f1']:.4f}")
    print(f"  Accuracy: {question_exact_match_metrics['accuracy']:.4f}")
    print(f"  Precision: {question_exact_match_metrics['precision']:.4f}, Recall: {question_exact_match_metrics['recall']:.4f}")

    print(f"\nLLM Evaluation:")
    print(f"  F1 Score: {question_llm_metrics['f1']:.4f}")
    print(f"  Accuracy: {question_llm_metrics['accuracy']:.4f}")
    print(f"  Precision: {question_llm_metrics['precision']:.4f}, Recall: {question_llm_metrics['recall']:.4f}")

    # Find best method at question level
    question_best_f1 = max(question_best_rouge_metrics['f1'], question_exact_match_metrics['f1'], question_llm_metrics['f1'])
    if question_llm_metrics['f1'] == question_best_f1:
        print(f"\n✓ Best method (Question-Level): LLM (F1: {question_llm_metrics['f1']:.4f})")
    elif question_exact_match_metrics['f1'] == question_best_f1:
        print(f"\n✓ Best method (Question-Level): Exact Match (F1: {question_exact_match_metrics['f1']:.4f})")
    else:
        print(f"\n✓ Best method (Question-Level): ROUGE threshold={question_best_threshold:.2f} (F1: {question_best_rouge_metrics['f1']:.4f})")

    # Current threshold (0.6) at question level
    print("\n" + "-"*80)
    print("Current ROUGE Threshold (0.6) Analysis (Question-Level):")
    print("-"*80)
    question_current_metrics = [r for r in question_rouge_results if abs(r['threshold'] - current_threshold) < 0.01][0]
    print(f"  Accuracy:  {question_current_metrics['accuracy']:.4f}")
    print(f"  Precision: {question_current_metrics['precision']:.4f}")
    print(f"  Recall:    {question_current_metrics['recall']:.4f}")
    print(f"  F1 Score:  {question_current_metrics['f1']:.4f}")

    if question_best_threshold != current_threshold:
        improvement = question_best_rouge_metrics['f1'] - question_current_metrics['f1']
        print(f"\n  → Changing from 0.6 to {question_best_threshold:.2f} would improve F1 by {improvement:.4f}")
    else:
        print(f"\n  → Current threshold (0.6) is already optimal!")

    # Update output data with question-level analysis
    output_data['question_level'] = {
        'summary': {
            'total_questions': num_questions,
            'annotated_questions': num_annotated_questions,
            'unannotated_questions': num_questions - num_annotated_questions,
            'questions_with_correct_sample': sum(annotated_question_labels),
            'questions_with_no_correct_sample': num_annotated_questions - sum(annotated_question_labels),
            'optimal_rouge_threshold': question_best_threshold,
            'best_rouge_metrics': question_best_rouge_metrics,
            'exact_match_metrics': question_exact_match_metrics,
            'llm_metrics': question_llm_metrics
        },
        'rouge_threshold_analysis': question_rouge_results,
        'comparison': {
            'rouge_vs_llm_f1_diff': question_best_rouge_metrics['f1'] - question_llm_metrics['f1'],
            'rouge_vs_llm_accuracy_diff': question_best_rouge_metrics['accuracy'] - question_llm_metrics['accuracy'],
            'exact_match_vs_llm_f1_diff': question_exact_match_metrics['f1'] - question_llm_metrics['f1'],
            'exact_match_vs_llm_accuracy_diff': question_exact_match_metrics['accuracy'] - question_llm_metrics['accuracy']
        }
    }

    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results (including question-level analysis) saved to: {args.output_file}")


if __name__ == '__main__':
    main()
