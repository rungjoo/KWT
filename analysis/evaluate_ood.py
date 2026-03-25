#!/usr/bin/env python3
"""
Evaluation script for OOD (Out-of-Distribution) datasets: RefuNQ and SelfAware
Evaluates only answerable questions across three model types:
1. Base model (3-shot)
2. SFT model
3. Ours (trained) model

Evaluation method: LLM judge for semantic equivalence checking
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import List, Dict, Any
import re
from rouge_score import rouge_scorer


def get_model_name(model_path):
    """Extract model name from model path for directory naming"""
    return Path(model_path).name.lower()


def load_refunq_answerable(file_path: str) -> List[Dict[str, Any]]:
    """Load RefuNQ answerable dataset (JSONL format)."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # label is a list of answers
            answers = item.get('label', [])
            answer = answers[0] if answers else ''
            data.append({
                'question': item['prompt'],
                'dataset': 'RefuNQ',
                'answer': answer,
                'all_answers': answers
            })
    return data


def load_selfaware_answerable(file_path: str) -> List[Dict[str, Any]]:
    """Load selfAware dataset (JSON array format) - only answerable questions."""
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    data = []
    for item in all_data:
        # Only load answerable questions
        if not item.get('answerable', False):
            continue

        answers = item.get('answer', [])
        if not isinstance(answers, list):
            answers = [answers] if answers else []
        answer = answers[0] if answers else ''

        data.append({
            'question': item['question'],
            'dataset': 'selfAware',
            'question_id': item.get('question_id'),
            'source': item.get('source', 'unknown'),
            'answer': answer,
            'all_answers': answers
        })
    return data


def load_ood_answerable_datasets(base_dir: str, dataset: str = 'all') -> List[Dict[str, Any]]:
    """Load OOD answerable datasets (RefuNQ and/or SelfAware)."""
    base_path = Path(base_dir)
    all_data = []

    # Load RefuNQ answerable
    if dataset in ['all', 'refunq']:
        refunq_path = base_path / 'RefuNQ' / 'RefuNQ_answerable.json'
        if refunq_path.exists():
            refunq_data = load_refunq_answerable(str(refunq_path))
            all_data.extend(refunq_data)
            print(f"Loaded {len(refunq_data)} answerable examples from RefuNQ")

    # Load selfAware answerable
    if dataset in ['all', 'selfaware']:
        selfaware_path = base_path / 'selfAware' / 'train.json'
        if selfaware_path.exists():
            selfaware_data = load_selfaware_answerable(str(selfaware_path))
            all_data.extend(selfaware_data)
            print(f"Loaded {len(selfaware_data)} answerable examples from selfAware")

    return all_data


def create_few_shot_prompt(question, examples=None):
    """Create few-shot prompt for base model."""
    prompt = ""

    if examples:
        prompt += "Here are some examples:\n\n"
        for ex in examples:
            prompt += f"Question: {ex['question']}\n"
            answer = ex.get('answer', '')
            prompt += f"Answer: {answer}\n\n"

    prompt += f"Question: {question}\n"
    prompt += "Answer:"

    return prompt


def create_instruct_prompt(question):
    """Create prompt for trained models (SFT/Ours)."""
    prompt = f"Question: {question}\n\nAnswer:"
    return prompt


def normalize_text(text) -> str:
    """Simple text normalization."""
    if isinstance(text, list):
        text = text[0] if text else ''
    text = str(text) if text is not None else ''
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def check_idk_token(text: str) -> bool:
    """Check if text contains <IDK> token"""
    return "<IDK>" in text


def evaluate_answer_llm(model, tokenizer, question, answer, model_answer):
    """Use LLM to evaluate if model_answer matches answer."""
    if not answer:
        return 'skipped_no_ground_truth', None, None

    right_answer_norm = normalize_text(answer)
    model_answer_norm = normalize_text(model_answer)

    if right_answer_norm == model_answer_norm:
        return 'correct', True, 1.0

    prompt = f"""Question: {question}

Answer1: {answer}
Answer2: {model_answer}

Are Answer1 and Answer2 semantically equivalent?
Answer only 'yes' or 'no':"""

    messages = [{"role": "user", "content": prompt}]
    prompt_with_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_with_template, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response = response.strip().lower()

    if 'yes' in response:
        return 'correct', True, 1.0
    else:
        return 'incorrect', False, 0.0


def evaluate_answer_rouge(all_answers, model_answer, threshold=0.6):
    """Use ROUGE score to evaluate if model_answer matches any answer."""
    if not all_answers:
        return 'skipped_no_ground_truth', None, None

    model_answer_norm = normalize_text(model_answer)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []

    for right_answer in all_answers:
        right_answer_norm = normalize_text(right_answer)
        if right_answer_norm == model_answer_norm:
            scores.append(1.0)
            continue
        rouge_scores = scorer.score(right_answer_norm, model_answer_norm)
        rougeL_f1 = rouge_scores['rougeL'].fmeasure
        scores.append(rougeL_f1)

    max_score = max(scores) if scores else 0.0
    is_correct = max_score >= threshold
    eval_result = 'correct' if is_correct else 'incorrect'

    return eval_result, is_correct, max_score


def inference_base_model(model_path: str, data: List[Dict[str, Any]], fewshot: int = 3, device: str = 'cuda') -> List[Dict[str, Any]]:
    """Run inference on base model with few-shot prompting."""
    print(f"Loading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Use first N examples as few-shot examples
    few_shot_examples = data[:fewshot]
    results = []

    for item in tqdm(data, desc="Base model inference (3-shot)"):
        prompt = create_few_shot_prompt(item['question'], examples=few_shot_examples)

        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip()

        model_answer_full = answer
        filtered_model_answer = model_answer_full.split('\n')[0].strip() if model_answer_full else ''
        has_idk = check_idk_token(filtered_model_answer)

        result = {
            'question': item['question'],
            'model_answer': model_answer_full,
            'filtered_model_answer': filtered_model_answer,
            'has_idk': has_idk,
            'dataset': item['dataset'],
            'answer': item.get('answer', ''),
            'all_answers': item.get('all_answers', []),
            'prompt': prompt
        }

        for key in item:
            if key not in result:
                result[key] = item[key]

        results.append(result)

    del model
    torch.cuda.empty_cache()
    return results


def inference_trained_model(model_path: str, data: List[Dict[str, Any]], device: str = 'cuda') -> List[Dict[str, Any]]:
    """Run inference on trained model (SFT or Ours)."""
    print(f"Loading trained model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    results = []

    for item in tqdm(data, desc="Trained model inference"):
        prompt = create_instruct_prompt(item['question'])

        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        prompt_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_tokens = outputs[0][prompt_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()

        model_answer_full = answer
        filtered_model_answer = model_answer_full.split('\n')[0].strip() if model_answer_full else ''
        has_idk = check_idk_token(filtered_model_answer)

        result = {
            'question': item['question'],
            'model_answer': model_answer_full,
            'filtered_model_answer': filtered_model_answer,
            'has_idk': has_idk,
            'dataset': item['dataset'],
            'answer': item.get('answer', ''),
            'all_answers': item.get('all_answers', []),
            'prompt': prompt
        }

        for key in item:
            if key not in result:
                result[key] = item[key]

        results.append(result)

    del model
    torch.cuda.empty_cache()
    return results


def evaluate_results(results: List[Dict[str, Any]], eval_method: str = 'llm',
                     judge_model=None, judge_tokenizer=None, threshold: float = 0.6) -> Dict[str, Any]:
    """Evaluate results and compute metrics."""
    stats = {'total': len(results), 'correct': 0, 'with_idk': 0, 'skipped': 0}
    scores = []

    print(f"\nEvaluating {len(results)} answerable questions using {eval_method} method...")

    for result in tqdm(results, desc="Evaluating"):
        answer = result.get('answer', '')
        all_answers = result.get('all_answers', [])
        model_answer = result['model_answer']
        question = result['question']

        contains_idk = check_idk_token(model_answer)
        clean_answer = model_answer.replace('<IDK>', '').replace('<|end_of_text|>', '').strip()
        result['has_idk'] = contains_idk

        if contains_idk:
            stats['with_idk'] += 1

        if eval_method == 'llm':
            if not answer:
                result['is_correct'] = None
                result['eval_result'] = 'skipped_no_ground_truth'
                stats['skipped'] += 1
                continue
            eval_result, is_correct, score = evaluate_answer_llm(
                judge_model, judge_tokenizer, question, answer, clean_answer
            )
        elif eval_method == 'rouge':
            if not all_answers:
                result['is_correct'] = None
                result['eval_result'] = 'skipped_no_ground_truth'
                stats['skipped'] += 1
                continue
            eval_result, is_correct, score = evaluate_answer_rouge(all_answers, clean_answer, threshold)
        else:
            raise ValueError(f"Unknown evaluation method: {eval_method}")

        result['is_correct'] = is_correct
        result['eval_result'] = eval_result
        result['eval_method'] = eval_method
        result['score'] = score

        if score is not None:
            scores.append(score)

        if is_correct:
            stats['correct'] += 1

    evaluatable = stats['total'] - stats['skipped']
    stats['accuracy'] = round((stats['correct'] / evaluatable * 100), 2) if evaluatable > 0 else 0.0
    stats['idk_rate'] = round((stats['with_idk'] / stats['total'] * 100), 2) if stats['total'] > 0 else 0.0
    stats['average_score'] = round(sum(scores) / len(scores), 4) if scores else 0.0

    # Per-dataset statistics
    dataset_stats = {}
    for dataset in ['RefuNQ', 'selfAware']:
        ds_results = [r for r in results if r['dataset'] == dataset]
        if not ds_results:
            continue
        ds_evaluated = [r for r in ds_results if r.get('is_correct') is not None]
        ds_correct = sum(1 for r in ds_evaluated if r.get('is_correct', False))
        ds_with_idk = sum(1 for r in ds_results if r.get('has_idk', False))

        dataset_stats[dataset] = {
            'total': len(ds_results),
            'correct': ds_correct,
            'skipped': len(ds_results) - len(ds_evaluated),
            'accuracy': round((ds_correct / len(ds_evaluated) * 100), 2) if ds_evaluated else 0.0,
            'with_idk': ds_with_idk,
            'idk_rate': round((ds_with_idk / len(ds_results) * 100), 2) if ds_results else 0.0
        }

    stats['dataset_stats'] = dataset_stats
    stats['eval_method'] = eval_method

    return stats


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Results saved to {output_path}")


def save_evaluation(stats: Dict[str, Any], output_path: str):
    """Save evaluation statistics to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Evaluation saved to {output_path}")


def print_summary(model_name: str, stats: Dict[str, Any]):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY: {model_name}")
    print("="*60)
    print(f"Evaluation Method: {stats.get('eval_method', 'llm')}")
    print(f"\nOverall Statistics:")
    print(f"  Total: {stats['total']}")
    print(f"  Correct: {stats['correct']} ({stats['accuracy']:.2f}%)")
    print(f"  With <IDK>: {stats['with_idk']} ({stats['idk_rate']:.2f}%)")
    if stats.get('skipped', 0) > 0:
        print(f"  Skipped: {stats['skipped']}")

    print(f"\nPer-Dataset Statistics:")
    for dataset, ds_stats in stats.get('dataset_stats', {}).items():
        print(f"  {dataset}:")
        print(f"    Total: {ds_stats['total']}, Correct: {ds_stats['correct']} ({ds_stats['accuracy']:.2f}%)")
        print(f"    With <IDK>: {ds_stats['with_idk']} ({ds_stats['idk_rate']:.2f}%)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate OOD answerable questions on Base/SFT/Ours models'
    )

    parser.add_argument('--judge_model_path', type=str, default='../../model/gemma-3-12b-it',
                        help='Path to judge model for LLM evaluation')
    parser.add_argument('--data_dir', type=str, default='../dataset_type',
                        help='Directory containing RefuNQ and selfAware datasets')
    parser.add_argument('--output_dir', type=str, default='./pre_exp',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--dataset', type=str, choices=['refunq', 'selfaware', 'all'], default='all',
                        help='Which OOD dataset to evaluate')
    parser.add_argument('--eval_method', type=str, default='llm', choices=['llm', 'rouge'],
                        help='Evaluation method: llm (default) or rouge')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Threshold for rouge evaluation (default: 0.6)')
    parser.add_argument('--fewshot', type=int, default=3, help='Number of few-shot examples for base model')

    # Model paths
    parser.add_argument('--base_model', type=str, default='../../model/Llama-3.2-3B',
                        help='Path to base model')
    parser.add_argument('--base_model_name', type=str, default='llama-3.2-3b',
                        help='Base model name for trained model path')
    parser.add_argument('--dataname', type=str, required=True, choices=['halueval', 'medqa', 'sciq'],
                        help='Dataset name used for training')
    parser.add_argument('--save_run_name', type=str, default='sample_weight_reverse_smooth',
                        help='Run name for trained model')
    parser.add_argument('--data_eval_method', type=str, default='llm', choices=['llm', 'rouge', 'em'],
                        help='Eval method used for training data')
    parser.add_argument('--data_threshold', type=str, default='',
                        help='Threshold used for training data')
    parser.add_argument('--sft_idk_weight', type=float, default=0.16,
                        help='IDK weight used in training')

    # Model selection
    parser.add_argument('--models', type=str, nargs='+', default=['base', 'sft', 'ours'],
                        choices=['base', 'sft', 'ours'],
                        help='Models to evaluate (default: all three)')

    args = parser.parse_args()

    # Setup data threshold
    if args.data_eval_method in ["llm", "em"]:
        args.data_threshold = ""
    else:
        args.data_threshold = float(args.data_threshold) if args.data_threshold else 0.6

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print(f"Loading OOD answerable datasets from {args.data_dir}")
    all_data = load_ood_answerable_datasets(args.data_dir, args.dataset)
    print(f"\nTotal answerable questions: {len(all_data)}")

    if not all_data:
        print("No data loaded. Exiting.")
        return

    # Load judge model for LLM evaluation
    judge_model = None
    judge_tokenizer = None
    if args.eval_method == 'llm':
        print(f"\nLoading judge model from {args.judge_model_path}")
        judge_model = AutoModelForCausalLM.from_pretrained(args.judge_model_path, device_map='auto')
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model_path)
        if judge_tokenizer.pad_token is None:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token

    # Evaluate each model type
    all_results = {}

    # 1. Base Model (3-shot)
    if 'base' in args.models:
        print("\n" + "="*60)
        print("Running Base Model (3-shot)")
        print("="*60)
        base_results = inference_base_model(args.base_model, all_data, fewshot=args.fewshot, device=args.device)
        base_stats = evaluate_results(base_results, args.eval_method, judge_model, judge_tokenizer, args.threshold)

        save_results(base_results, str(output_dir / f"base_3shot_{args.dataname}_results.jsonl"))
        save_evaluation(base_stats, str(output_dir / f"base_3shot_{args.dataname}_evaluation.json"))
        print_summary("Base Model (3-shot)", base_stats)
        all_results['base'] = base_stats

    # 2. SFT Model
    if 'sft' in args.models:
        print("\n" + "="*60)
        print("Running SFT Model")
        print("="*60)
        sft_model_path = f"/mnt/frdata/rungjoo/hall/halu_model/{args.dataname}/{args.base_model_name}/sft/sft"

        if Path(sft_model_path).exists():
            sft_results = inference_trained_model(sft_model_path, all_data, device=args.device)
            sft_stats = evaluate_results(sft_results, args.eval_method, judge_model, judge_tokenizer, args.threshold)

            save_results(sft_results, str(output_dir / f"sft_{args.dataname}_results.jsonl"))
            save_evaluation(sft_stats, str(output_dir / f"sft_{args.dataname}_evaluation.json"))
            print_summary("SFT Model", sft_stats)
            all_results['sft'] = sft_stats
        else:
            print(f"SFT model not found at {sft_model_path}")

    # 3. Ours (Trained) Model
    if 'ours' in args.models:
        print("\n" + "="*60)
        print("Running Ours (Trained) Model")
        print("="*60)
        ours_model_path = f"/mnt/frdata/rungjoo/hall/halu_model/{args.dataname}/{args.base_model_name}/{args.save_run_name}/sw_{args.data_eval_method}{args.data_threshold}_idk{args.sft_idk_weight}"

        if Path(ours_model_path).exists():
            ours_results = inference_trained_model(ours_model_path, all_data, device=args.device)
            ours_stats = evaluate_results(ours_results, args.eval_method, judge_model, judge_tokenizer, args.threshold)

            result_name = f"ours_{args.save_run_name}_{args.data_eval_method}{args.data_threshold}_idk{args.sft_idk_weight}_{args.dataname}"
            save_results(ours_results, str(output_dir / f"{result_name}_results.jsonl"))
            save_evaluation(ours_stats, str(output_dir / f"{result_name}_evaluation.json"))
            print_summary("Ours (Trained) Model", ours_stats)
            all_results['ours'] = ours_stats
        else:
            print(f"Ours model not found at {ours_model_path}")


if __name__ == "__main__":
    main()
