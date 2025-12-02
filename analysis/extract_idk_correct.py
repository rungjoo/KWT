#!/usr/bin/env python3
"""
Extract cases where <IDK> token appears and score is 1.0 from evaluation files.

Usage:
    python extract_idk_correct.py <input_file>

Example:
    python extract_idk_correct.py halueval/llama_3.2-3b/sft_evaluated_llm.json

Output will be saved to: halueval/llama_3.2-3b/sft_evaluated_llm_filtered.json
"""

import json
import sys
import os
from pathlib import Path


def extract_idk_correct_cases(input_file, output_file):
    """
    Extract cases where had_idk_token is True and score is 1.0

    Args:
        input_file: Path to the input evaluation JSON file
        output_file: Path to save the filtered results

    Returns:
        List of filtered cases
    """
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract results
    if 'results' not in data:
        print(f"Error: 'results' key not found in {input_file}")
        return []

    results = data['results']

    # Filter cases with <IDK> token and score 1.0
    filtered_cases = []
    for item in results:
        if item.get('had_idk_token', False) and item.get('score', 0) == 1.0:
            filtered_cases.append(item)

    # Print summary
    print(f"\nInput file: {input_file}")
    print(f"Total cases: {len(results)}")
    print(f"Cases with <IDK> and score 1.0: {len(filtered_cases)}")

    if len(filtered_cases) > 0:
        print(f"Percentage: {len(filtered_cases) / len(results) * 100:.2f}%")

        # Show first few examples
        print("\n--- First 3 Examples ---")
        for i, case in enumerate(filtered_cases[:3]):
            print(f"\n[Example {i+1}]")
            print(f"Question: {case.get('question', 'N/A')}")
            print(f"Model Answer: {case.get('model_answer', 'N/A')[:200]}...")
            print(f"Filtered Answer: {case.get('filtered_model_answer', 'N/A')}")
            print(f"Score: {case.get('score', 'N/A')}")
            print(f"Evaluation: {case.get('evaluation', 'N/A')}")

    # Save to output file
    output_data = {
        'input_file': input_file,
        'total_cases': len(results),
        'filtered_count': len(filtered_cases),
        'filtered_cases': filtered_cases
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nFiltered cases saved to: {output_file}")

    return filtered_cases


def main():
    # python3 extract_idk_correct.py halueval/qwen2.5-3b/seal_evaluated_llm.json
    if len(sys.argv) < 2:
        print("Usage: python extract_idk_correct.py <input_file>")
        print("\nExample:")
        print("  python extract_idk_correct.py halueval/llama_3.2-3b/sft_evaluated_llm.json")
        print("\nOutput will be saved with '_filtered' suffix before the extension")
        sys.exit(1)

    input_file = sys.argv[1]

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    # Generate output file name by adding '_filtered' before the extension
    input_path = Path(input_file)
    output_file = str(input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}")

    # Extract cases
    filtered_cases = extract_idk_correct_cases(input_file, output_file)

    return filtered_cases


if __name__ == "__main__":
    main()
