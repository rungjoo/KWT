#!/usr/bin/env python3
import json
import argparse
from collections import Counter
import os

def analyze_sample_statistics(evaluated_file):
    """
    Analyze statistics from evaluated results file.
    Shows distribution of how many samples were correct (out of 5 samples per question).
    """

    print(f"\nAnalyzing file: {evaluated_file}")

    # Load evaluated results
    with open(evaluated_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get summary info
    summary = data.get('summary', {})
    results = data.get('results', [])

    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    print(f"Evaluation method: {summary.get('eval_method', 'N/A')}")
    if summary.get('threshold'):
        print(f"Threshold: {summary.get('threshold')}")
    print(f"Total questions: {summary.get('total_questions', len(results))}")
    print(f"Total samples: {summary.get('total_samples', 0)}")
    print(f"Correct samples: {summary.get('correct_samples', 0)}")
    print(f"Sample-level accuracy: {summary.get('sample_accuracy', 0):.2f}%")
    print(f"Average score: {summary.get('average_score', 0):.4f}")

    # Analyze distribution of correct samples per question
    correct_counts = []
    for item in results:
        samples_correct = item.get('samples_correct', 0)
        correct_counts.append(samples_correct)

    # Count distribution
    distribution = Counter(correct_counts)
    total_questions = len(results)

    # Get max samples (usually 5)
    max_samples = max(correct_counts) if correct_counts else 0
    samples_per_question = results[0].get('num_samples', 5) if results else 5

    print("\n" + "="*60)
    print(f"DISTRIBUTION: How many samples (out of {samples_per_question}) got correct answer per question")
    print("="*60)

    # Print distribution from max to min
    for i in range(samples_per_question, -1, -1):
        count = distribution.get(i, 0)
        percentage = (count / total_questions * 100) if total_questions > 0 else 0
        bar = '#' * int(percentage / 2)  # Scale bar to fit screen
        print(f"{i}/{samples_per_question} correct: {count:5d} questions ({percentage:5.2f}%) {bar}")

    # Calculate additional statistics
    print("\n" + "="*60)
    print("QUESTION-LEVEL STATISTICS")
    print("="*60)

    # At least one correct
    at_least_one = sum(1 for c in correct_counts if c >= 1)
    at_least_one_pct = (at_least_one / total_questions * 100) if total_questions > 0 else 0
    print(f"Questions with >=1 correct sample: {at_least_one:5d} ({at_least_one_pct:.2f}%)")

    # Majority correct (>= 3 out of 5)
    majority_correct = sum(1 for c in correct_counts if c >= (samples_per_question + 1) // 2)
    majority_pct = (majority_correct / total_questions * 100) if total_questions > 0 else 0
    print(f"Questions with >={(samples_per_question + 1) // 2} correct samples: {majority_correct:5d} ({majority_pct:.2f}%)")

    # All correct
    all_correct = sum(1 for c in correct_counts if c == samples_per_question)
    all_correct_pct = (all_correct / total_questions * 100) if total_questions > 0 else 0
    print(f"Questions with all {samples_per_question} correct:  {all_correct:5d} ({all_correct_pct:.2f}%)")

    # None correct
    none_correct = sum(1 for c in correct_counts if c == 0)
    none_correct_pct = (none_correct / total_questions * 100) if total_questions > 0 else 0
    print(f"Questions with 0 correct:     {none_correct:5d} ({none_correct_pct:.2f}%)")

    # Average correct per question
    avg_correct = sum(correct_counts) / len(correct_counts) if correct_counts else 0
    print(f"\nAverage correct samples per question: {avg_correct:.2f}/{samples_per_question}")

    print("\n" + "="*60)

    return {
        'total_questions': total_questions,
        'samples_per_question': samples_per_question,
        'distribution': dict(distribution),
        'at_least_one': at_least_one,
        'majority_correct': majority_correct,
        'all_correct': all_correct,
        'none_correct': none_correct,
        'avg_correct': avg_correct
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze statistics from answer check results',
        epilog='''
Examples:
python3 03_stats.py --input_file halueval/qwen2.5-3b/base_model_temp0.7_samples5_fewshot3_evaluated_rouge0.35.json
  python3 03_stats.py --input_file sciq/qwen3-4b/base_model_temp0.7_samples5_fewshot3_evaluated_em.json
  python3 03_stats.py --input_file halueval/qwen2.5-3b/base_model_temp0.7_samples5_fewshot3_evaluated_llm.json
  python3 03_stats.py --input_file medqa/qwen3-4b/base_model_temp0.7_samples5_fewshot3_evaluated_bertscore.json
  python3 03_stats.py --input_file sciq/qwen3-4b/base_model_temp0.7_samples5_fewshot3_evaluated_bertscore.json
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to evaluated JSON file (output from 02_answer_check.py)')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return

    # Analyze statistics
    stats = analyze_sample_statistics(args.input_file)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
