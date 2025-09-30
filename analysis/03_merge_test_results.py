#!/usr/bin/env python3
import json
import os
import argparse
from typing import Dict, List, Any
from collections import defaultdict
import pandas as pd
from pathlib import Path


def load_evaluated_results(file_path: str) -> Dict[str, Any]:
    """Load evaluated results from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_model_results(base_results: Dict, comparison_results: Dict,
                         base_name: str = "base", comparison_name: str = "comparison") -> Dict[str, Any]:
    """Compare base model results with another model's results."""

    base_data = base_results['results']
    comp_data = comparison_results['results']

    # Create index mapping for faster lookup (using question as key)
    comp_dict = {item['question']: item for item in comp_data}

    # Initialize counters
    comparison_stats = {
        'both_correct': 0,
        'both_incorrect': 0,
        'base_correct_other_incorrect': 0,
        'base_incorrect_other_correct': 0,
        'total_compared': 0,
        'missing_in_comparison': 0
    }

    # Initialize IDK-based cases (Base: O/X, Compare: 4 cases)
    # Total 8 combinations: Base(O/X) × Compare(IDK-O, NO-IDK-O, IDK-X, NO-IDK-X)
    idk_stats = {
        # Base Correct (O) cases
        'base_O_comp_O_no_idk': 0,    # Base correct, Comp correct without IDK
        'base_O_comp_O_with_idk': 0,  # Base correct, Comp correct with IDK
        'base_O_comp_X_no_idk': 0,    # Base correct, Comp incorrect without IDK
        'base_O_comp_X_with_idk': 0,  # Base correct, Comp incorrect with IDK

        # Base Incorrect (X) cases
        'base_X_comp_O_no_idk': 0,    # Base incorrect, Comp correct without IDK
        'base_X_comp_O_with_idk': 0,  # Base incorrect, Comp correct with IDK
        'base_X_comp_X_no_idk': 0,    # Base incorrect, Comp incorrect without IDK
        'base_X_comp_X_with_idk': 0,  # Base incorrect, Comp incorrect with IDK
    }

    # Detailed comparison results
    detailed_comparisons = []

    for base_item in base_data:
        question = base_item['question']

        if question in comp_dict:
            comp_item = comp_dict[question]
            base_eval = base_item['evaluation']
            comp_eval = comp_item['evaluation']

            # Get IDK token presence (only for comparison model, base never has IDK)
            comp_has_idk = comp_item.get('had_idk_token', False) or comp_item.get('contains_i_dont_know', False)

            # Count different scenarios (handle both 'correct'/'incorrect' and 'match'/'no_match')
            base_is_correct = base_eval in ['correct', 'match']
            comp_is_correct = comp_eval in ['correct', 'match']

            # Determine main status
            if base_is_correct and comp_is_correct:
                comparison_stats['both_correct'] += 1
                status = 'both_correct'
            elif not base_is_correct and not comp_is_correct:
                comparison_stats['both_incorrect'] += 1
                status = 'both_incorrect'
            elif base_is_correct and not comp_is_correct:
                comparison_stats['base_correct_other_incorrect'] += 1
                status = 'base_correct_other_incorrect'
            elif not base_is_correct and comp_is_correct:
                comparison_stats['base_incorrect_other_correct'] += 1
                status = 'base_incorrect_other_correct'
            else:
                status = 'unknown'

            # Track 8 cases: Base(O/X) × Comp(IDK-O, NO-IDK-O, IDK-X, NO-IDK-X)
            if base_is_correct:  # Base O
                if comp_is_correct:  # Comp O
                    if comp_has_idk:
                        idk_stats['base_O_comp_O_with_idk'] += 1
                        idk_status = 'comp_O_with_idk'
                    else:
                        idk_stats['base_O_comp_O_no_idk'] += 1
                        idk_status = 'comp_O_no_idk'
                else:  # Comp X
                    if comp_has_idk:
                        idk_stats['base_O_comp_X_with_idk'] += 1
                        idk_status = 'comp_X_with_idk'
                    else:
                        idk_stats['base_O_comp_X_no_idk'] += 1
                        idk_status = 'comp_X_no_idk'
            else:  # Base X
                if comp_is_correct:  # Comp O
                    if comp_has_idk:
                        idk_stats['base_X_comp_O_with_idk'] += 1
                        idk_status = 'comp_O_with_idk'
                    else:
                        idk_stats['base_X_comp_O_no_idk'] += 1
                        idk_status = 'comp_O_no_idk'
                else:  # Comp X
                    if comp_has_idk:
                        idk_stats['base_X_comp_X_with_idk'] += 1
                        idk_status = 'comp_X_with_idk'
                    else:
                        idk_stats['base_X_comp_X_no_idk'] += 1
                        idk_status = 'comp_X_no_idk'

            comparison_stats['total_compared'] += 1

            # Store detailed comparison
            detailed_comparisons.append({
                'question': question,
                'base_evaluation': base_eval,
                'base_answer': base_item.get('model_answer', ''),
                'comparison_evaluation': comp_eval,
                'comparison_answer': comp_item.get('model_answer', ''),
                'right_answer': base_item.get('right_answer', ''),
                'hallucinated_answer': base_item.get('hallucinated_answer', ''),
                'status': status,
                'idk_status': idk_status,
                'comparison_had_idk': comp_has_idk,
                'comparison_contains_idk': comp_item.get('contains_i_dont_know', False)
            })
        else:
            comparison_stats['missing_in_comparison'] += 1

    # Calculate percentages
    total = comparison_stats['total_compared']
    if total > 0:
        comparison_stats['both_correct_pct'] = round(100 * comparison_stats['both_correct'] / total, 2)
        comparison_stats['both_incorrect_pct'] = round(100 * comparison_stats['both_incorrect'] / total, 2)
        comparison_stats['base_correct_other_incorrect_pct'] = round(100 * comparison_stats['base_correct_other_incorrect'] / total, 2)
        comparison_stats['base_incorrect_other_correct_pct'] = round(100 * comparison_stats['base_incorrect_other_correct'] / total, 2)

        # Calculate IDK percentages for each category
        keys = list(idk_stats.keys())
        for key in keys:
            if not key.endswith('_pct'):
                idk_stats[f'{key}_pct'] = round(100 * idk_stats[key] / total, 2) if total > 0 else 0

    return {
        'comparison_name': f"{base_name}_vs_{comparison_name}",
        'base_summary': base_results['summary'],
        'comparison_summary': comparison_results['summary'],
        'comparison_statistics': comparison_stats,
        'idk_detailed_statistics': idk_stats,
        'detailed_comparisons': detailed_comparisons
    }


def print_comparison_summary(comparison_result: Dict[str, Any]):
    """Print a formatted summary of the comparison."""
    print(f"\n{'='*80}")
    print(f"Comparison: {comparison_result['comparison_name']}")
    print(f"{'='*80}")

    print("\nBase Model Summary:")
    base_summary = comparison_result['base_summary']
    print(f"  - Total: {base_summary['total']}")
    print(f"  - Correct: {base_summary['correct']} ({base_summary['accuracy']}%)")
    incorrect_rate = round(100 - base_summary['accuracy'], 2)
    print(f"  - Incorrect: {base_summary['incorrect']} ({incorrect_rate}%)")

    print("\nComparison Model Summary:")
    comp_summary = comparison_result['comparison_summary']
    print(f"  - Total: {comp_summary['total']}")
    print(f"  - Correct: {comp_summary['correct']} ({comp_summary['accuracy']}%)")
    comp_incorrect_rate = round(100 - comp_summary['accuracy'], 2)
    print(f"  - Incorrect: {comp_summary['incorrect']} ({comp_incorrect_rate}%)")

    print("\nOverall Comparison Statistics:")
    stats = comparison_result['comparison_statistics']
    print(f"  - Total Compared Samples: {stats['total_compared']}")
    print(f"  - Both Correct: {stats['both_correct']} ({stats.get('both_correct_pct', 0)}%)")
    print(f"  - Both Incorrect: {stats['both_incorrect']} ({stats.get('both_incorrect_pct', 0)}%)")
    print(f"  - Base Correct, Other Incorrect: {stats['base_correct_other_incorrect']} ({stats.get('base_correct_other_incorrect_pct', 0)}%)")
    print(f"  - Base Incorrect, Other Correct: {stats['base_incorrect_other_correct']} ({stats.get('base_incorrect_other_correct_pct', 0)}%)")

    # Print IDK-based detailed statistics
    if 'idk_detailed_statistics' in comparison_result:
        print("\n📊 IDK Token Analysis (Base: O/X, Compare: with/without IDK):")
        idk_stats = comparison_result['idk_detailed_statistics']

        print("\n  ✅ Base Correct (O) Cases:")
        print(f"    - Comp O (no IDK): {idk_stats['base_O_comp_O_no_idk']} ({idk_stats['base_O_comp_O_no_idk_pct']}%)")
        print(f"    - Comp O (with IDK): {idk_stats['base_O_comp_O_with_idk']} ({idk_stats['base_O_comp_O_with_idk_pct']}%)")
        print(f"    - Comp X (no IDK): {idk_stats['base_O_comp_X_no_idk']} ({idk_stats['base_O_comp_X_no_idk_pct']}%)")
        print(f"    - Comp X (with IDK): {idk_stats['base_O_comp_X_with_idk']} ({idk_stats['base_O_comp_X_with_idk_pct']}%)")

        print("\n  ❌ Base Incorrect (X) Cases:")
        print(f"    - Comp O (no IDK): {idk_stats['base_X_comp_O_no_idk']} ({idk_stats['base_X_comp_O_no_idk_pct']}%)")
        print(f"    - Comp O (with IDK): {idk_stats['base_X_comp_O_with_idk']} ({idk_stats['base_X_comp_O_with_idk_pct']}%)")
        print(f"    - Comp X (no IDK): {idk_stats['base_X_comp_X_no_idk']} ({idk_stats['base_X_comp_X_no_idk_pct']}%)")
        print(f"    - Comp X (with IDK): {idk_stats['base_X_comp_X_with_idk']} ({idk_stats['base_X_comp_X_with_idk_pct']}%)")

    if stats['missing_in_comparison'] > 0:
        print(f"\n  - Missing in Comparison: {stats['missing_in_comparison']}")


def save_comparison_results(comparison_result: Dict[str, Any], output_path: str):
    """Save comparison results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, indent=2, ensure_ascii=False)
    print(f"\nComparison results saved to: {output_path}")


def create_comparison_csv(comparison_result: Dict[str, Any], output_path: str):
    """Create a CSV file with detailed comparisons for easy analysis."""
    df = pd.DataFrame(comparison_result['detailed_comparisons'])

    # Reorder columns for better readability
    columns_order = [
        'question', 'status', 'idk_status',
        'base_evaluation', 'comparison_evaluation',
        'comparison_had_idk', 'comparison_contains_idk',
        'right_answer', 'hallucinated_answer',
        'base_answer', 'comparison_answer'
    ]

    # Only include columns that exist
    columns_order = [col for col in columns_order if col in df.columns]
    df = df[columns_order]

    # Save to CSV
    csv_path = output_path.replace('.json', '.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Detailed comparison CSV saved to: {csv_path}")

    # Print distribution of statuses
    print("\nStatus Distribution:")
    status_counts = df['status'].value_counts()
    for status, count in status_counts.items():
        print(f"  - {status}: {count}")

    # Print IDK status distribution
    if 'idk_status' in df.columns:
        print("\nIDK Status Distribution:")
        idk_status_counts = df['idk_status'].value_counts()
        for status, count in idk_status_counts.items():
            print(f"  - {status}: {count}")


def main():
    # python3 03_merge_test_results.py --base halueval/llama_3.2-3b/base_model_evaluated.json --compare halueval/llama_3.2-3b/sft_evaluated.json halueval/llama_3.2-3b/sft_dpo_evaluated.json halueval/llama_3.2-3b/sft_dpo2_evaluated.json --output_dir halueval/llama_3.2-3b/comparison_results
    # python3 03_merge_test_results.py --base medqa/llama_3.2-3b/base_model_evaluated.json --compare medqa/llama_3.2-3b/sft_evaluated.json medqa/llama_3.2-3b/sft_dpo_evaluated.json medqa/llama_3.2-3b/sft_dpo2_evaluated.json --output_dir medqa/llama_3.2-3b/comparison_results
    parser = argparse.ArgumentParser(description='Merge and compare HaluEval evaluation results')
    parser.add_argument('--base', type=str, required=True,
                       help='Path to base model evaluated.json file')
    parser.add_argument('--compare', type=str, nargs='+', required=True,
                       help='Path(s) to comparison model evaluated.json files')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                       help='Directory to save comparison results')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load base model results
    print(f"Loading base model results from: {args.base}")
    base_results = load_evaluated_results(args.base)
    base_name = Path(args.base).stem.replace('_evaluated', '')

    # Compare with each comparison model
    all_comparisons = []

    for comp_path in args.compare:
        print(f"\nLoading comparison model results from: {comp_path}")
        comp_results = load_evaluated_results(comp_path)
        comp_name = Path(comp_path).stem.replace('_evaluated', '')

        # Perform comparison
        comparison = compare_model_results(
            base_results, comp_results,
            base_name, comp_name
        )

        # Print summary
        print_comparison_summary(comparison)

        # Collect for summary only (no individual files)
        all_comparisons.append(comparison)

    # Save all comparisons summary
    summary_output_path = os.path.join(args.output_dir, 'all_comparisons_summary.json')
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'base_model': base_name,
            'comparisons': [
                {
                    'name': comp['comparison_name'],
                    'statistics': comp['comparison_statistics'],
                    'idk_detailed_statistics': comp['idk_detailed_statistics'],
                }
                for comp in all_comparisons
            ]
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"All comparisons summary saved to: {summary_output_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()