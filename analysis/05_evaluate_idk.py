#!/usr/bin/env python3
"""
Evaluation script for testing model's ability to:
1. Output <IDK> on unanswerable questions (NEC, RefuNQ, selfAware unanswerable)
2. Provide correct answers on answerable questions (selfAware answerable)

Supports NEC, RefuNQ, and selfAware datasets.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import List, Dict, Any
import re


def load_nec_unanswerable(file_path: str) -> List[Dict[str, Any]]:
    """Load NEC unanswerable dataset (JSONL format)."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({
                'question': item['prompt'],
                'dataset': 'NEC',
                'category': item.get('category', 'unknown')
            })
    return data


def load_nec_answerable(file_path: str) -> List[Dict[str, Any]]:
    """Load NEC answerable dataset (JSONL format).
    Note: NEC answerable doesn't have ground truth answers, only questions."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({
                'question': item['prompt'],
                'dataset': 'NEC',
                'category': item.get('category', 'unknown'),
                'answer': ''  # No ground truth available
            })
    return data


def load_refunq_unanswerable(file_path: str) -> List[Dict[str, Any]]:
    """Load RefuNQ unanswerable dataset (JSONL format)."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append({
                'question': item['prompt'],
                'dataset': 'RefuNQ',
                'label': item.get('label', 'NEC')
            })
    return data


def load_refunq_answerable(file_path: str) -> List[Dict[str, Any]]:
    """Load RefuNQ answerable dataset (JSONL format)."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # label is a list of answers
            answers = item.get('label', [])
            # Use the first answer as the primary answer
            answer = answers[0] if answers else ''
            data.append({
                'question': item['prompt'],
                'dataset': 'RefuNQ',
                'answer': answer,
                'all_answers': answers  # Keep all possible answers
            })
    return data


def load_selfaware_data(file_path: str) -> List[Dict[str, Any]]:
    """Load selfAware dataset (JSON array format) - both answerable and unanswerable."""
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    # Load both answerable and unanswerable questions
    data = []
    for item in all_data:
        data.append({
            'question': item['question'],
            'dataset': 'selfAware',
            'question_id': item.get('question_id'),
            'source': item.get('source', 'unknown'),
            'answerable': item.get('answerable', True),
            'answer': item.get('answer', '') if item.get('answerable', True) else ''
        })
    return data


def load_all_datasets(base_dir: str) -> List[Dict[str, Any]]:
    """Load all datasets (both answerable and unanswerable)."""
    base_path = Path(base_dir)

    all_data = []

    # Load NEC (both answerable and unanswerable)
    nec_unanswerable_path = base_path / 'NEC' / 'NEC_unanswerable.json'
    if nec_unanswerable_path.exists():
        nec_data = load_nec_unanswerable(str(nec_unanswerable_path))
        # Mark as unanswerable
        for item in nec_data:
            item['answerable'] = False
        all_data.extend(nec_data)
        print(f"Loaded {len(nec_data)} examples from NEC (unanswerable)")

    nec_answerable_path = base_path / 'NEC' / 'NEC_answerable.json'
    if nec_answerable_path.exists():
        nec_answerable_data = load_nec_answerable(str(nec_answerable_path))
        # Mark as answerable (but no ground truth)
        for item in nec_answerable_data:
            item['answerable'] = True
        all_data.extend(nec_answerable_data)
        print(f"Loaded {len(nec_answerable_data)} examples from NEC (answerable, no ground truth)")

    # Load RefuNQ (both answerable and unanswerable)
    refunq_unanswerable_path = base_path / 'RefuNQ' / 'RefuNQ_unanswerable.json'
    if refunq_unanswerable_path.exists():
        refunq_data = load_refunq_unanswerable(str(refunq_unanswerable_path))
        # Mark as unanswerable
        for item in refunq_data:
            item['answerable'] = False
        all_data.extend(refunq_data)
        print(f"Loaded {len(refunq_data)} examples from RefuNQ (unanswerable)")

    refunq_answerable_path = base_path / 'RefuNQ' / 'RefuNQ_answerable.json'
    if refunq_answerable_path.exists():
        refunq_answerable_data = load_refunq_answerable(str(refunq_answerable_path))
        # Mark as answerable
        for item in refunq_answerable_data:
            item['answerable'] = True
        all_data.extend(refunq_answerable_data)
        print(f"Loaded {len(refunq_answerable_data)} examples from RefuNQ (answerable)")

    # Load selfAware (both answerable and unanswerable)
    selfaware_path = base_path / 'selfAware' / 'train.json'
    if selfaware_path.exists():
        selfaware_data = load_selfaware_data(str(selfaware_path))
        answerable_count = sum(1 for item in selfaware_data if item.get('answerable', True))
        unanswerable_count = len(selfaware_data) - answerable_count
        all_data.extend(selfaware_data)
        print(f"Loaded {len(selfaware_data)} examples from selfAware ({answerable_count} answerable, {unanswerable_count} unanswerable)")

    return all_data

def create_instruct_our_promt(question):
    prompt = f"Question: {question}\n\nAnswer:"
    return prompt    


def check_idk_token(text: str) -> bool:
    """Check if text contains <IDK> token"""
    return "<IDK>" in text


def normalize_text(text) -> str:
    """Simple text normalization: lowercase, strip whitespace, collapse multiple spaces"""
    # Handle list inputs by taking the first element
    if isinstance(text, list):
        text = text[0] if text else ''
    # Convert to string if not already
    text = str(text) if text is not None else ''
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def evaluate_answer(model, tokenizer, question, right_answer, model_answer):
    """Use LLM to evaluate if model_answer matches right_answer"""

    # First check for exact match after normalization
    right_answer_norm = normalize_text(right_answer)
    model_answer_norm = normalize_text(model_answer)

    if right_answer_norm == model_answer_norm:
        return 'correct', True

    # Use LLM to evaluate semantic equivalence
    prompt = f"""Question: {question}

Answer1: {right_answer}
Answer2: {model_answer}

Are Answer1 and Answer2 semantically equivalent?
Answer only 'yes' or 'no':"""

    # Apply chat template and tokenize
    messages = [{"role": "user", "content": prompt}]
    prompt_with_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_with_template, return_tensors="pt", truncation=True, max_length=2048)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response = response.strip().lower()

    # Extract yes/no from response
    if 'yes' in response:
        return 'correct', True
    else:
        return 'incorrect', False


def inference_model(model_path: str, data: List[Dict[str, Any]], device: str = 'cuda') -> List[Dict[str, Any]]:
    """Run inference on the model."""
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    results = []

    for item in tqdm(data, desc="Running inference"):
        prompt = create_instruct_our_promt(item['question'])

        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        prompt_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_tokens = outputs[0][prompt_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()

        # Check if IDK is in the response
        has_idk = check_idk_token(answer)

        result = {
            'question': item['question'],
            'model_answer': answer,
            'has_idk': has_idk,
            'dataset': item['dataset'],
            'prompt': prompt
        }

        # Add original metadata
        for key in item:
            if key not in result:
                result[key] = item[key]

        results.append(result)

    return results


def evaluate_results(results: List[Dict[str, Any]], judge_model=None, judge_tokenizer=None) -> Dict[str, Any]:
    """Evaluate the results and compute metrics for both answerable and unanswerable questions."""

    # Separate answerable and unanswerable
    answerable_results = [r for r in results if r.get('answerable', False)]
    unanswerable_results = [r for r in results if not r.get('answerable', False)]

    # Evaluate answerable questions (need judge model)
    answerable_stats = {'total': 0, 'correct': 0, 'with_idk': 0, 'accuracy': 0.0, 'skipped': 0}
    if answerable_results and judge_model and judge_tokenizer:
        print(f"\nEvaluating {len(answerable_results)} answerable questions with judge model...")
        for result in tqdm(answerable_results, desc="Evaluating answerable"):
            answer = result.get('answer', '')
            model_answer = result['model_answer']
            question = result['question']

            # Remove <IDK> for comparison
            contains_idk = check_idk_token(model_answer)
            clean_answer = model_answer.replace('<IDK>', '').replace('<|end_of_text|>', '').strip()
            result['has_idk'] = contains_idk

            # Skip evaluation if no ground truth answer
            if not answer:
                result['is_correct'] = None
                result['eval_result'] = 'skipped_no_ground_truth'
                answerable_stats['skipped'] += 1
                if contains_idk:
                    answerable_stats['with_idk'] += 1
                continue

            # Evaluate with judge model
            eval_result, is_correct = evaluate_answer(
                judge_model, judge_tokenizer, question, answer, clean_answer
            )

            result['is_correct'] = is_correct
            result['eval_result'] = eval_result

            if is_correct:
                answerable_stats['correct'] += 1
            if contains_idk:
                answerable_stats['with_idk'] += 1

        answerable_stats['total'] = len(answerable_results)
        evaluatable = answerable_stats['total'] - answerable_stats['skipped']
        answerable_stats['accuracy'] = round((answerable_stats['correct'] / evaluatable * 100), 2) if evaluatable > 0 else 0.0
        answerable_stats['idk_rate'] = round((answerable_stats['with_idk'] / answerable_stats['total'] * 100), 2) if answerable_stats['total'] > 0 else 0.0

    # Evaluate unanswerable questions (IDK is correct)
    unanswerable_stats = {'total': len(unanswerable_results), 'idk_count': 0, 'accuracy': 0.0}
    if unanswerable_results:
        unanswerable_stats['idk_count'] = sum(1 for r in unanswerable_results if r['has_idk'])
        unanswerable_stats['accuracy'] = round((unanswerable_stats['idk_count'] / unanswerable_stats['total'] * 100), 2) if unanswerable_stats['total'] > 0 else 0.0

    # Compute per-dataset statistics
    dataset_stats = {}
    for dataset in set(r['dataset'] for r in results):
        dataset_results = [r for r in results if r['dataset'] == dataset]
        ds_answerable = [r for r in dataset_results if r.get('answerable', False)]
        ds_unanswerable = [r for r in dataset_results if not r.get('answerable', False)]

        # For answerable questions, break down by IDK presence
        answerable_with_idk = [r for r in ds_answerable if r.get('has_idk', False)]
        answerable_without_idk = [r for r in ds_answerable if not r.get('has_idk', False)]

        answerable_stats = {
            'total': len(ds_answerable),
            'correct': sum(1 for r in ds_answerable if r.get('is_correct', False)) if judge_model else 0,
            'accuracy': round((sum(1 for r in ds_answerable if r.get('is_correct', False)) / len(ds_answerable) * 100), 2) if len(ds_answerable) > 0 and judge_model else 0.0,
            # IDK presence breakdown
            'with_idk': {
                'total': len(answerable_with_idk),
                'correct': sum(1 for r in answerable_with_idk if r.get('is_correct', False)) if judge_model else 0,
                'incorrect': sum(1 for r in answerable_with_idk if r.get('is_correct') == False) if judge_model else 0,
                'accuracy': round((sum(1 for r in answerable_with_idk if r.get('is_correct', False)) / len(answerable_with_idk) * 100), 2) if len(answerable_with_idk) > 0 and judge_model else 0.0
            },
            'without_idk': {
                'total': len(answerable_without_idk),
                'correct': sum(1 for r in answerable_without_idk if r.get('is_correct', False)) if judge_model else 0,
                'incorrect': sum(1 for r in answerable_without_idk if r.get('is_correct') == False) if judge_model else 0,
                'accuracy': round((sum(1 for r in answerable_without_idk if r.get('is_correct', False)) / len(answerable_without_idk) * 100), 2) if len(answerable_without_idk) > 0 and judge_model else 0.0
            }
        }

        dataset_stats[dataset] = {
            'total': len(dataset_results),
            'answerable': answerable_stats,
            'unanswerable': {
                'total': len(ds_unanswerable),
                'idk_count': sum(1 for r in ds_unanswerable if r['has_idk']),
                'accuracy': round((sum(1 for r in ds_unanswerable if r['has_idk']) / len(ds_unanswerable) * 100), 2) if len(ds_unanswerable) > 0 else 0.0
            }
        }

    overall_stats = {
        'total': len(results),
        'answerable': answerable_stats,
        'unanswerable': unanswerable_stats,
        'dataset_stats': dataset_stats
    }

    return overall_stats


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\nResults saved to {output_path}")


def save_evaluation(stats: Dict[str, Any], output_path: str):
    """Save evaluation statistics to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Evaluation statistics saved to {output_path}")


def print_summary(stats: Dict[str, Any]):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nOverall Statistics:")
    print(f"  Total questions: {stats['total']}")

    # Answerable statistics
    answerable = stats['answerable']
    print(f"\n  Answerable Questions ({answerable['total']}):")
    if answerable['total'] > 0:
        evaluatable = answerable['total'] - answerable.get('skipped', 0)
        print(f"    Correct answers: {answerable['correct']}/{evaluatable} ({answerable['accuracy']:.2f}%)")
        if answerable.get('skipped', 0) > 0:
            print(f"    Skipped (no ground truth): {answerable['skipped']}")
        print(f"    With <IDK> token: {answerable['with_idk']} ({answerable.get('idk_rate', 0):.2f}%)")
    else:
        print(f"    No answerable questions")

    # Unanswerable statistics
    unanswerable = stats['unanswerable']
    print(f"\n  Unanswerable Questions ({unanswerable['total']}):")
    if unanswerable['total'] > 0:
        print(f"    IDK responses: {unanswerable['idk_count']} ({unanswerable['accuracy']:.2f}%)")
    else:
        print(f"    No unanswerable questions")

    # Per-Dataset Statistics
    print(f"\nPer-Dataset Statistics:")
    for dataset, ds_stats in stats['dataset_stats'].items():
        print(f"\n  {dataset} (Total: {ds_stats['total']}):")

        # Answerable
        if ds_stats['answerable']['total'] > 0:
            print(f"    Answerable ({ds_stats['answerable']['total']}):")
            print(f"      Correct: {ds_stats['answerable']['correct']} ({ds_stats['answerable']['accuracy']:.2f}%)")

        # Unanswerable
        if ds_stats['unanswerable']['total'] > 0:
            print(f"    Unanswerable ({ds_stats['unanswerable']['total']}):")
            print(f"      IDK responses: {ds_stats['unanswerable']['idk_count']} ({ds_stats['unanswerable']['accuracy']:.2f}%)")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate model on both answerable and unanswerable questions'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the model to evaluate'
    )
    parser.add_argument(
        '--judge_model_path',
        type=str,
        default='../../model/gemma-3-12b-it',
        help='Path to judge model for evaluating answerable questions'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='../dataset_type',
        help='Directory containing NEC, RefuNQ, and selfAware datasets'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./idk_evaluation',
        help='Directory to save evaluation results'
    )

    parser.add_argument(
        '--train_data_name',
        type=str,
        default='halueval',
        help='Training data name (for output naming)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for inference'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['NEC', 'RefuNQ', 'selfAware', 'all'],
        default='all',
        help='Which dataset to evaluate on'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print(f"Loading datasets from {args.data_dir}")
    all_data = load_all_datasets(args.data_dir)

    # Filter by dataset if specified
    if args.dataset != 'all':
        all_data = [d for d in all_data if d['dataset'] == args.dataset]
        print(f"Filtered to {len(all_data)} examples from {args.dataset}")

    answerable_count = sum(1 for d in all_data if d.get('answerable', False))
    unanswerable_count = len(all_data) - answerable_count
    print(f"\nTotal questions: {len(all_data)} ({answerable_count} answerable, {unanswerable_count} unanswerable)")

    # Run inference
    print(f"\nRunning inference with model: {args.model_path}")
    results = inference_model(args.model_path, all_data, device=args.device)

    # Prepare file paths
    p = Path(args.model_path)
    model_name = f"{p.parent.name}_{p.name}"

    results_file = output_dir / f"{model_name}_idk_results.jsonl"

    # Load judge model for evaluating answerable questions
    judge_model = None
    judge_tokenizer = None
    if answerable_count > 0:
        print(f"\nLoading judge model from {args.judge_model_path}")
        judge_model = AutoModelForCausalLM.from_pretrained(
            args.judge_model_path,
            device_map='auto'
        )
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model_path)
        if judge_tokenizer.pad_token is None:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token

    # Evaluate results
    print("\nEvaluating results...")
    stats = evaluate_results(results, judge_model, judge_tokenizer)

    # Save results AFTER evaluation (with is_correct field)
    print("\nSaving results (after evaluation with is_correct)...")
    save_results(results, str(results_file))

    # Save evaluation stats
    stats_file = output_dir / f"{model_name}_idk_evaluation.json"
    save_evaluation(stats, str(stats_file))

    # Save per-dataset results
    print("\nSaving per-dataset results...")
    dataset_names = set(r['dataset'] for r in results)

    # Print summary
    print_summary(stats)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
