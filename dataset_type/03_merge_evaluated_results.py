#!/usr/bin/env python3
"""
Script to merge and compare evaluated.json files
Creates a merged file with combined results and detailed statistics
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def load_evaluated_json(file_path: str) -> Dict:
    """Load evaluated.json file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_and_analyze_results(file_paths: List[str], base_dir) -> None:
    """Merge results from multiple files and analyze agreement"""
    
    print("=" * 80)
    print("Merging and Analyzing Evaluated JSON Files")
    print("=" * 80)
    
    # Load all files
    all_data = {}
    all_results = {}
    file_names = []
    
    for file_path in file_paths:
        name = Path(file_path).stem
        file_names.append(name)
        print(f"\nLoading {file_path}...")
        
        data = load_evaluated_json(file_path)
        all_data[name] = data
        
        # Print summary if available
        if isinstance(data, dict) and 'summary' in data:
            summary = data['summary']
            print(f"  - Summary: Total={summary.get('total', 'N/A')}, "
                  f"Correct={summary.get('correct', 'N/A')}, "
                  f"Accuracy={summary.get('accuracy', 'N/A')}%")
        
        # Extract results
        if isinstance(data, dict) and 'results' in data:
            all_results[name] = data['results']
        else:
            all_results[name] = data
            
        print(f"  - Loaded {len(all_results[name])} items")
    
    # Create merged results with question as key
    print("\n" + "=" * 80)
    print("Merging Results")
    print("=" * 80)
    
    merged_by_question = defaultdict(dict)
    
    for source_name, results in all_results.items():
        for item in results:
            question = item.get('question', '')
            if question:
                # Store the entire item for each source
                merged_by_question[question][source_name] = item
    
    print(f"Total unique questions: {len(merged_by_question)}")
    
    # Analyze agreement and create merged results
    merged_results = []
    
    # Statistics
    stats = {
        'total_questions': len(merged_by_question),
        'all_correct': 0,
        'all_incorrect': 0,
        'only_base_correct': 0,
        'only_instruct_correct': 0,
        'only_self_sft_correct': 0,
        'base_and_instruct_correct': 0,
        'base_and_self_sft_correct': 0,
        'instruct_and_self_sft_correct': 0,
        'missing_in_base': 0,
        'missing_in_instruct': 0,
        'missing_in_self_sft': 0,
        'all_present': 0,
        'two_present': 0,
        'one_present': 0
    }
    
    # Detailed tracking
    all_correct_questions = []
    all_incorrect_questions = []
    disagreement_questions = []
    
    for question, sources in merged_by_question.items():
        # Create merged item
        merged_item = {
            'question': question
        }
        
        # Check if question exists in all three files
        base_key = 'base_model_results_evaluated'
        instruct_key = 'instruct_model_results_evaluated'
        self_sft_key = 'self_sft_model_results_evaluated'
        
        base_item = sources.get(base_key)
        instruct_item = sources.get(instruct_key)
        self_sft_item = sources.get(self_sft_key)
        
        # Count how many models have this question
        present_count = sum([base_item is not None, instruct_item is not None, self_sft_item is not None])
        
        if present_count == 3:
            stats['all_present'] += 1
        elif present_count == 2:
            stats['two_present'] += 1
        elif present_count == 1:
            stats['one_present'] += 1
            
        # Track missing items
        if not base_item:
            stats['missing_in_base'] += 1
        if not instruct_item:
            stats['missing_in_instruct'] += 1
        if not self_sft_item:
            stats['missing_in_self_sft'] += 1
        
        # Get common fields from any available item
        first_item = base_item or instruct_item or self_sft_item
        if first_item:
            merged_item['knowledge'] = first_item.get('knowledge', '')
            merged_item['right_answer'] = first_item.get('right_answer', '')
            merged_item['hallucinated_answer'] = first_item.get('hallucinated_answer', '')
        
        # Add results from base model if available
        if base_item:
            merged_item['base_model'] = {
                'model_answer': base_item.get('model_answer', ''),
                'filtered_model_answer': base_item.get('filtered_model_answer', ''),
                'evaluation': base_item.get('evaluation', ''),
                'match': base_item.get('match', False)
            }
        
        # Add results from instruct model if available
        if instruct_item:
            merged_item['instruct_model'] = {
                'model_answer': instruct_item.get('model_answer', ''),
                'filtered_model_answer': instruct_item.get('filtered_model_answer', ''),
                'evaluation': instruct_item.get('evaluation', ''),
                'match': instruct_item.get('match', False)
            }
        
        # Add results from self_sft model if available
        if self_sft_item:
            merged_item['self_sft_model'] = {
                'model_answer': self_sft_item.get('model_answer', ''),
                'filtered_model_answer': self_sft_item.get('filtered_model_answer', ''),
                'evaluation': self_sft_item.get('evaluation', ''),
                'match': self_sft_item.get('match', False)
            }
        
        # Analyze agreement when all three models are present
        if base_item and instruct_item and self_sft_item:
            base_correct = base_item.get('match', False)
            instruct_correct = instruct_item.get('match', False)
            self_sft_correct = self_sft_item.get('match', False)
            
            correct_count = sum([base_correct, instruct_correct, self_sft_correct])
            
            if correct_count == 3:
                stats['all_correct'] += 1
                merged_item['agreement'] = 'all_correct'
                all_correct_questions.append({
                    'question': question[:100],
                    'right_answer': merged_item['right_answer']
                })
            elif correct_count == 0:
                stats['all_incorrect'] += 1
                merged_item['agreement'] = 'all_incorrect'
                all_incorrect_questions.append({
                    'question': question[:100],
                    'right_answer': merged_item['right_answer'],
                    'base_answer': base_item.get('filtered_model_answer', ''),
                    'instruct_answer': instruct_item.get('filtered_model_answer', ''),
                    'self_sft_answer': self_sft_item.get('filtered_model_answer', '')
                })
            elif correct_count == 2:
                if base_correct and instruct_correct:
                    stats['base_and_instruct_correct'] += 1
                    merged_item['agreement'] = 'base_and_instruct_correct'
                elif base_correct and self_sft_correct:
                    stats['base_and_self_sft_correct'] += 1
                    merged_item['agreement'] = 'base_and_self_sft_correct'
                else:  # instruct_correct and self_sft_correct
                    stats['instruct_and_self_sft_correct'] += 1
                    merged_item['agreement'] = 'instruct_and_self_sft_correct'
                    
                disagreement_questions.append({
                    'question': question[:100],
                    'right_answer': merged_item['right_answer'],
                    'base_answer': base_item.get('filtered_model_answer', ''),
                    'base_correct': base_correct,
                    'instruct_answer': instruct_item.get('filtered_model_answer', ''),
                    'instruct_correct': instruct_correct,
                    'self_sft_answer': self_sft_item.get('filtered_model_answer', ''),
                    'self_sft_correct': self_sft_correct,
                    'correct_count': correct_count
                })
            else:  # correct_count == 1
                if base_correct:
                    stats['only_base_correct'] += 1
                    merged_item['agreement'] = 'only_base_correct'
                elif instruct_correct:
                    stats['only_instruct_correct'] += 1
                    merged_item['agreement'] = 'only_instruct_correct'
                else:  # self_sft_correct
                    stats['only_self_sft_correct'] += 1
                    merged_item['agreement'] = 'only_self_sft_correct'
                    
                disagreement_questions.append({
                    'question': question[:100],
                    'right_answer': merged_item['right_answer'],
                    'base_answer': base_item.get('filtered_model_answer', ''),
                    'base_correct': base_correct,
                    'instruct_answer': instruct_item.get('filtered_model_answer', ''),
                    'instruct_correct': instruct_correct,
                    'self_sft_answer': self_sft_item.get('filtered_model_answer', ''),
                    'self_sft_correct': self_sft_correct,
                    'correct_count': correct_count
                })
        else:
            # Handle cases where not all three models are present
            merged_item['agreement'] = f'present_in_{present_count}_models'
        
        merged_results.append(merged_item)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Comparison Statistics")
    print("=" * 80)
    
    print(f"\nCoverage:")
    print(f"  - Questions in all three models: {stats['all_present']}")
    print(f"  - Questions in two models: {stats['two_present']}")
    print(f"  - Questions in one model: {stats['one_present']}")
    print(f"  - Missing in base model: {stats['missing_in_base']}")
    print(f"  - Missing in instruct model: {stats['missing_in_instruct']}")
    print(f"  - Missing in self_sft model: {stats['missing_in_self_sft']}")
    
    if stats['all_present'] > 0:
        print(f"\nFor questions in all three models ({stats['all_present']} questions):")
        print(f"  - All models correct: {stats['all_correct']} ({stats['all_correct']/stats['all_present']*100:.2f}%)")
        print(f"  - All models incorrect: {stats['all_incorrect']} ({stats['all_incorrect']/stats['all_present']*100:.2f}%)")
        print(f"  - Only base model correct: {stats['only_base_correct']} ({stats['only_base_correct']/stats['all_present']*100:.2f}%)")
        print(f"  - Only instruct model correct: {stats['only_instruct_correct']} ({stats['only_instruct_correct']/stats['all_present']*100:.2f}%)")
        print(f"  - Only self_sft model correct: {stats['only_self_sft_correct']} ({stats['only_self_sft_correct']/stats['all_present']*100:.2f}%)")
        print(f"  - Base and instruct correct: {stats['base_and_instruct_correct']} ({stats['base_and_instruct_correct']/stats['all_present']*100:.2f}%)")
        print(f"  - Base and self_sft correct: {stats['base_and_self_sft_correct']} ({stats['base_and_self_sft_correct']/stats['all_present']*100:.2f}%)")
        print(f"  - Instruct and self_sft correct: {stats['instruct_and_self_sft_correct']} ({stats['instruct_and_self_sft_correct']/stats['all_present']*100:.2f}%)")
        
        print(f"\nAgreement Analysis:")
        full_agreement_count = stats['all_correct'] + stats['all_incorrect']
        partial_agreement_count = stats['base_and_instruct_correct'] + stats['base_and_self_sft_correct'] + stats['instruct_and_self_sft_correct']
        no_agreement_count = stats['only_base_correct'] + stats['only_instruct_correct'] + stats['only_self_sft_correct']
        print(f"  - All models agree (all right or all wrong): {full_agreement_count} ({full_agreement_count/stats['all_present']*100:.2f}%)")
        print(f"  - Two models correct, one wrong: {partial_agreement_count} ({partial_agreement_count/stats['all_present']*100:.2f}%)")
        print(f"  - Only one model correct: {no_agreement_count} ({no_agreement_count/stats['all_present']*100:.2f}%)")
    
    # Show sample questions
    print("\n" + "=" * 80)
    print("Sample Questions")
    print("=" * 80)
    
    if all_correct_questions:
        print(f"\nSample questions all models got CORRECT (showing first 3):")
        for item in all_correct_questions[:3]:
            print(f"  Q: {item['question'][:80]}...")
            print(f"     Answer: {item['right_answer']}")
    
    if all_incorrect_questions:
        print(f"\nSample questions all models got WRONG (showing first 3):")
        for item in all_incorrect_questions[:3]:
            print(f"  Q: {item['question'][:80]}...")
            print(f"     Correct answer: {item['right_answer']}")
            print(f"     Base answer: {item['base_answer']}")
            print(f"     Instruct answer: {item['instruct_answer']}")
            print(f"     Self-SFT answer: {item['self_sft_answer']}")
    
    if disagreement_questions:
        print(f"\nSample questions where models DISAGREE (showing first 5):")
        for item in disagreement_questions[:5]:
            print(f"  Q: {item['question'][:80]}...")
            print(f"     Correct answer: {item['right_answer']}")
            print(f"     Base answer: {item['base_answer']} {'✓' if item.get('base_correct', False) else '✗'}")
            print(f"     Instruct answer: {item['instruct_answer']} {'✓' if item.get('instruct_correct', False) else '✗'}")
            print(f"     Self-SFT answer: {item['self_sft_answer']} {'✓' if item.get('self_sft_correct', False) else '✗'}")
            print(f"     ({item['correct_count']} model(s) correct)")
    
    # Calculate new summary statistics
    total_merged = len(merged_results)
    
    # Create output data structure
    output_data = {
        'summary': {
            'total': total_merged,
            'all_models_correct': stats['all_correct'],
            'at_least_one_correct': (
                stats['all_correct'] + 
                stats['only_base_correct'] + 
                stats['only_instruct_correct'] + 
                stats['only_self_sft_correct'] +
                stats['base_and_instruct_correct'] +
                stats['base_and_self_sft_correct'] +
                stats['instruct_and_self_sft_correct']
            ),
            'at_least_two_correct': (
                stats['all_correct'] +
                stats['base_and_instruct_correct'] +
                stats['base_and_self_sft_correct'] +
                stats['instruct_and_self_sft_correct']
            ),
            'all_models_incorrect': stats['all_incorrect'],
            'full_agreement_rate': (stats['all_correct'] + stats['all_incorrect']) / stats['all_present'] * 100 if stats['all_present'] > 0 else 0,
            'statistics': stats
        },
        'results': merged_results
    }
    
    # Save merged results
    output_dir = f"{base_dir}/merged"
    os.makedirs(output_dir, exist_ok=True)  # 폴더 없으면 생성

    output_file = f"{output_dir}/merged_evaluated.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Merged results saved to: {output_file}")
    print(f"   - Total questions: {total_merged}")
    print(f"   - Questions all three models got correct: {stats['all_correct']}")
    print(f"   - Questions at least two models got correct: {output_data['summary']['at_least_two_correct']}")
    print(f"   - Questions at least one model got correct: {output_data['summary']['at_least_one_correct']}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Merge and analyze evaluated results')
    parser.add_argument('--dataname', type=str, required=True, choices=['halueval', 'medqa'],
                       help='Dataset name (halueval or medqa)')
    args = parser.parse_args()

    # Find all evaluated.json files
    base_dir = f"./{args.dataname}"

    evaluated_files = []

    # Look for files with 'evaluated' in the name
    for file in os.listdir(base_dir):
        if 'evaluated' in file and file.endswith('.json'):
            evaluated_files.append(os.path.join(base_dir, file))

    if not evaluated_files:
        print(f"No evaluated.json files found in {base_dir}!")
        return

    print(f"Found {len(evaluated_files)} evaluated files:")
    for f in evaluated_files:
        print(f"  - {f}")

    # Analyze and merge the files
    merge_and_analyze_results(evaluated_files, base_dir)

if __name__ == "__main__":
    main()