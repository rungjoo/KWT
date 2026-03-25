import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def get_score_range(score):
    """Get the score range category for a given score."""
    if score is None or score == 0:
        return '0.0'
    elif score < 0.2:
        return '0.0-0.2'
    elif score < 0.4:
        return '0.2-0.4'
    elif score < 0.6:
        return '0.4-0.6'
    elif score < 0.8:
        return '0.6-0.8'
    elif score < 1.0:
        return '0.8-1.0'
    else:  # score == 1.0
        return '1.0'

def extract_diverse_samples(rouge_file, llm_file, output_file, num_items=100):
    """
    Extract samples with diverse score distribution for human annotation.

    Strategy:
    1. Group questions by their samples' score characteristics
    2. Sample from each score range to ensure diversity
    3. Add both rouge and llm scores
    4. Add 'human' field for annotation
    """
    print(f"Processing rouge: {rouge_file}")
    print(f"Processing llm:   {llm_file}")

    # Load both input files
    with open(rouge_file, 'r', encoding='utf-8') as f:
        rouge_data = json.load(f)

    with open(llm_file, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)

    rouge_results = rouge_data.get('results', [])
    llm_results = llm_data.get('results', [])

    # Verify both have same number of results
    if len(rouge_results) != len(llm_results):
        print(f"  ⚠ Warning: rouge has {len(rouge_results)} results, llm has {len(llm_results)} results")

    # Categorize questions by score range of their samples
    questions_by_score_range = defaultdict(list)

    for idx, (rouge_item, llm_item) in enumerate(zip(rouge_results, llm_results)):
        if 'sample_evaluations' not in rouge_item or 'sample_evaluations' not in llm_item:
            continue

        # Combine rouge and llm data into one item
        combined_item = rouge_item.copy()

        # Merge sample evaluations with both rouge and llm scores
        combined_samples = []
        for rouge_sample, llm_sample in zip(rouge_item['sample_evaluations'], llm_item['sample_evaluations']):
            merged_sample = rouge_sample.copy()
            # Rename rouge fields
            merged_sample['rouge_evaluation'] = merged_sample.pop('evaluation')
            merged_sample['rouge_match'] = merged_sample.pop('match')
            merged_sample['rouge_score'] = merged_sample.pop('score')
            # Add llm fields
            merged_sample['llm_evaluation'] = llm_sample.get('evaluation')
            merged_sample['llm_match'] = llm_sample.get('match')
            merged_sample['llm_score'] = llm_sample.get('score')
            combined_samples.append(merged_sample)

        combined_item['sample_evaluations'] = combined_samples

        # Get all rouge scores from this question's samples (use rouge for diversity calculation)
        scores = []
        for sample in combined_samples:
            score = sample.get('rouge_score', 0)
            if score is None:
                score = 0
            scores.append(score)

        # Check which score ranges are present in this question
        score_ranges = set(get_score_range(s) for s in scores)

        # Calculate diversity (number of different score ranges)
        diversity = len(score_ranges)

        # Also calculate if there are any intermediate scores (not just 0 or 1)
        has_intermediate = any(0 < s < 1 for s in scores)

        # Priority: questions with intermediate scores and high diversity
        priority = (has_intermediate, diversity, np.std(scores))

        # Add to each score range category
        for score_range in score_ranges:
            questions_by_score_range[score_range].append((idx, combined_item, priority))

    # Print distribution
    print("\n  Score range distribution (questions):")
    total_available = {}
    for range_name in ['0.0', '0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '1.0']:
        count = len(questions_by_score_range[range_name])
        total_available[range_name] = count
        print(f"    {range_name:>10}: {count:5d} questions")

    # Strategy: Sample proportionally from each range, but prioritize intermediate scores
    # Desired distribution for 100 samples:
    target_distribution = {
        '0.0': 10,
        '0.0-0.2': 15,
        '0.2-0.4': 20,
        '0.4-0.6': 20,
        '0.6-0.8': 20,
        '0.8-1.0': 15,
        '1.0': 10
    }

    # Adjust if not enough samples in some ranges
    adjusted_distribution = {}
    remaining = num_items
    for range_name in ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '0.0', '1.0']:
        available = total_available[range_name]
        target = target_distribution[range_name]
        allocated = min(available, target, remaining)
        adjusted_distribution[range_name] = allocated
        remaining -= allocated

    print(f"\n  Target distribution for {num_items} questions:")
    for range_name in ['0.0', '0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '1.0']:
        print(f"    {range_name:>10}: {adjusted_distribution[range_name]:3d} questions")

    # Select questions from each range
    selected_indices = set()
    selected_items = []

    for range_name in ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '0.0', '1.0']:
        candidates = questions_by_score_range[range_name]
        target_count = adjusted_distribution[range_name]

        if target_count == 0:
            continue

        # Sort by priority (has_intermediate, diversity, std)
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Select top candidates that haven't been selected yet
        selected_from_range = 0
        for idx, item, priority in candidates:
            if idx not in selected_indices and selected_from_range < target_count:
                selected_indices.add(idx)
                selected_items.append((idx, item))
                selected_from_range += 1

            if selected_from_range >= target_count:
                break

    # Sort selected items by original index to maintain order
    selected_items.sort(key=lambda x: x[0])

    # Add 'human' field to selected items
    annotated_results = []
    total_samples = 0

    for idx, item in selected_items:
        annotated_item = item.copy()

        if 'sample_evaluations' in annotated_item:
            for sample in annotated_item['sample_evaluations']:
                sample['human'] = None  # null value - to be filled with true/false
                total_samples += 1

        annotated_results.append(annotated_item)

    # Create output data structure
    output_data = {
        "annotation_info": {
            "num_questions": len(annotated_results),
            "total_samples": total_samples,
            "source_files": {
                "rouge": str(rouge_file),
                "llm": str(llm_file)
            },
            "sampling_strategy": "diverse_score_distribution",
            "target_distribution": adjusted_distribution,
            "description": "Human annotation dataset with diverse score distribution - includes both rouge and llm scores - each sample_evaluation has a 'human' field to be filled with true/false"
        },
        "results": annotated_results
    }

    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Extracted {len(annotated_results)} questions with {total_samples} samples")
    print(f"  ✓ Saved to: {output_file}\n")

    return len(annotated_results), total_samples

def main():
    # Base directory
    base_dir = Path("/mnt/fr20tb/rungjoo/hall/dataset_type")

    # Dataset names
    datasets = ["halueval", "medqa", "sciq"]

    # Input file name patterns
    rouge_filename = "base_model_temp0.7_samples5_fewshot3_evaluated_rouge0.6.json"
    llm_filename = "base_model_temp0.7_samples5_fewshot3_evaluated_llm.json"

    # Output file name pattern
    output_filename = "human_annotation_100samples_diverse.json"

    print("="*70)
    print("Extracting samples with diverse score distribution")
    print("="*70)
    print()

    results_summary = []

    # Process each dataset
    for dataset in datasets:
        rouge_file = base_dir / dataset / rouge_filename
        llm_file = base_dir / dataset / llm_filename
        output_file = base_dir / dataset / output_filename

        # Check if input files exist
        if not rouge_file.exists():
            print(f"⚠ Warning: File not found - {rouge_file}")
            print()
            continue

        if not llm_file.exists():
            print(f"⚠ Warning: File not found - {llm_file}")
            print()
            continue

        # Extract samples
        num_questions, num_samples = extract_diverse_samples(
            rouge_file,
            llm_file,
            output_file,
            num_items=100
        )

        results_summary.append({
            "dataset": dataset,
            "questions": num_questions,
            "samples": num_samples
        })

    # Print summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    for result in results_summary:
        print(f"{result['dataset']:>10}: {result['questions']} questions, {result['samples']} samples")
    print()
    print(f"Total: {sum(r['questions'] for r in results_summary)} questions, "
          f"{sum(r['samples'] for r in results_summary)} samples")
    print("="*70)

if __name__ == "__main__":
    main()
