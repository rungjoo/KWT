#!/usr/bin/env python3
"""
Evaluation script for testing model's ability to:
1. Output <IDK> on unanswerable questions (NEC, RefuNQ, selfAware unanswerable)
2. Provide correct answers on answerable questions (selfAware answerable)

Supports NEC, RefuNQ, and selfAware datasets.

Evaluation methods for answerable questions:
- llm: Use LLM judge model for semantic equivalence checking
- rouge: Use ROUGE-L F1 score with configurable threshold (default: 0.6)

For unanswerable questions, only checks for <IDK> token presence.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import List, Dict, Any
import re
import random
from rouge_score import rouge_scorer


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
                'answer': '',  # No ground truth available
                'all_answers': []  # No ground truth available
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
        # Get answer list from the original data
        answers = item.get('answer', []) if item.get('answerable', True) else []
        # Convert to list if it's not already
        if not isinstance(answers, list):
            answers = [answers] if answers else []
        # Use first answer as the primary answer (string)
        answer = answers[0] if answers else ''

        data.append({
            'question': item['question'],
            'dataset': 'selfAware',
            'question_id': item.get('question_id'),
            'source': item.get('source', 'unknown'),
            'answerable': item.get('answerable', True),
            'answer': answer,  # String: first answer
            'all_answers': answers  # List: all possible answers
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

def create_few_shot_prompt(question, examples=None):
    prompt = ""

    if examples:
        prompt += "Here are some examples:\n\n"
        for ex in examples:
            prompt += f"Question: {ex['question']}\n"
            # Handle different field names for answers
            answer = ex.get('answer', ex.get('right_answer', ex.get('correct_answer', '')))
            prompt += f"Answer: {answer}\n\n"

    prompt += f"Question: {question}\n"
    prompt += "Answer:"

    return prompt


def create_few_shot_prompt_idk(question, examples=None):
    prompt = "If the answer is unknown or not in your knowledge, answer with <IDK>.\n"

    if examples:
        prompt += "Here are some examples:\n\n"
        # Randomly select one example to add <IDK>
        idk_idx = random.randint(0, len(examples) - 1)

        for idx, ex in enumerate(examples):
            prompt += f"Question: {ex['question']}\n"
            # Handle different field names for answers
            answer = ex.get('answer', ex.get('right_answer', ex.get('correct_answer', '')))
            # Add <IDK> to randomly selected example
            if idx == idk_idx:
                prompt += f"Answer: {answer} <IDK>\n\n"
            else:
                prompt += f"Answer: {answer}\n\n"

    prompt += f"Question: {question}\n"
    prompt += "Answer:"

    return prompt


def create_instruct_our_promt(question):
    prompt = f"Question: {question}\n\nAnswer:"
    return prompt


def create_instruct_our_promt_no_idk(question):
    prompt = (
        "Give a short and concise answer. "
        f"Question: {question}\n\nAnswer:"
    )
    return prompt


def create_instruct_our_promt_idk(question):
    prompt = (
        "Give a short and concise answer. "
        "If the answer is unknown or not in your knowledge, answer with <IDK>.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
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


def evaluate_answer_llm(model, tokenizer, question, answer, model_answer):
    """Use LLM to evaluate if model_answer matches answer.

    Args:
        model: Judge model
        tokenizer: Judge tokenizer
        question: Question text
        answer: The correct answer (string)
        model_answer: Model's answer

    Returns:
        tuple: (eval_result, is_correct, score)
    """
    if not answer:
        return 'skipped_no_ground_truth', None, None

    # First check for exact match after normalization
    right_answer_norm = normalize_text(answer)
    model_answer_norm = normalize_text(model_answer)

    if right_answer_norm == model_answer_norm:
        return 'correct', True, 1.0

    # Use LLM to evaluate semantic equivalence
    prompt = f"""Question: {question}

Answer1: {answer}
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
        return 'correct', True, 1.0
    else:
        return 'incorrect', False, 0.0


def evaluate_answer_rouge(all_answers, model_answer, threshold=0.6):
    """Use ROUGE score to evaluate if model_answer matches any of the answers in all_answers.
    Returns correct if ANY answer exceeds the threshold.

    Args:
        all_answers: List of possible correct answers
        model_answer: Model's answer
        threshold: ROUGE-L F1 threshold for determining correctness

    Returns:
        tuple: (eval_result, is_correct, max_score)
    """
    if not all_answers:
        return 'skipped_no_ground_truth', None, None

    model_answer_norm = normalize_text(model_answer)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []

    for right_answer in all_answers:
        right_answer_norm = normalize_text(right_answer)

        # Check for exact match after normalization
        if right_answer_norm == model_answer_norm:
            scores.append(1.0)
            continue

        # Calculate ROUGE scores
        rouge_scores = scorer.score(right_answer_norm, model_answer_norm)

        # Use ROUGE-L F1 score as the main metric
        rougeL_f1 = rouge_scores['rougeL'].fmeasure
        scores.append(rougeL_f1)

    # Use max score (if any answer exceeds threshold, it's correct)
    max_score = max(scores) if scores else 0.0

    # For ROUGE: max >= threshold is considered correct
    is_correct = max_score >= threshold
    eval_result = 'correct' if is_correct else 'incorrect'

    return eval_result, is_correct, max_score


def inference_our_model(model_path: str, data: List[Dict[str, Any]], device: str = 'cuda') -> List[Dict[str, Any]]:
    """Run inference on the trained model (from checkpoint)."""
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
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_tokens = outputs[0][prompt_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()

        # Extract first paragraph from model answer
        model_answer_full = answer
        filtered_model_answer = model_answer_full.split('\n')[0].strip() if model_answer_full else ''

        # Check if IDK is in the response
        has_idk = check_idk_token(filtered_model_answer)

        result = {
            'question': item['question'],
            'model_answer': model_answer_full,
            'filtered_model_answer': filtered_model_answer,
            'has_idk': has_idk,
            'dataset': item['dataset'],
            'prompt': prompt
        }

        # Add answer field (from original data)
        if 'answer' in item:
            result['answer'] = item['answer']

        # Add all_answers field
        if 'all_answers' in item:
            result['all_answers'] = item['all_answers']

        # Add original metadata
        for key in item:
            if key not in result:
                result[key] = item[key]

        results.append(result)

    return results


def inference_instruct_model(model_path: str, data: List[Dict[str, Any]], device: str = 'cuda', method: str = "default", fewshot: int = 3) -> List[Dict[str, Any]]:
    """Run inference on the instruct model using create_few_shot_prompt_idk."""
    print(f"Loading instruct model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare few-shot examples from answerable questions only
    answerable_data = [item for item in data if item.get('answerable', True)]
    few_shot_examples = answerable_data[:fewshot] if answerable_data else []

    results = []

    for item in tqdm(data, desc="Running instruct model inference"):
        # Select prompt based on method
        if method == "idk":
            prompt = create_few_shot_prompt_idk(item['question'], examples=few_shot_examples)
        elif method == "no_idk":
            prompt = create_few_shot_prompt(item['question'], examples=few_shot_examples)
        else:  # default
            prompt = create_instruct_our_promt(item['question'])

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

        # Decode only the generated tokens (excluding the prompt)
        generated_tokens = outputs[0][prompt_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()

        # Extract first paragraph from model answer
        model_answer_full = answer
        filtered_model_answer = model_answer_full.split('\n')[0].strip() if model_answer_full else ''

        # Check if IDK is in the response
        has_idk = check_idk_token(filtered_model_answer)

        result = {
            'question': item['question'],
            'model_answer': model_answer_full,
            'filtered_model_answer': filtered_model_answer,
            'has_idk': has_idk,
            'dataset': item['dataset'],
            'prompt': prompt
        }

        # Add answer field (from original data)
        if 'answer' in item:
            result['answer'] = item['answer']

        # Add all_answers field
        if 'all_answers' in item:
            result['all_answers'] = item['all_answers']

        # Add original metadata
        for key in item:
            if key not in result:
                result[key] = item[key]

        results.append(result)

    return results


def evaluate_results(results: List[Dict[str, Any]], eval_method='llm', judge_model=None, judge_tokenizer=None, threshold=None) -> Dict[str, Any]:
    """Evaluate the results and compute metrics for both answerable and unanswerable questions.

    Args:
        results: List of result dictionaries
        eval_method: Evaluation method ('llm' or 'rouge')
        judge_model: Model for LLM evaluation (required if eval_method='llm')
        judge_tokenizer: Tokenizer for LLM evaluation (required if eval_method='llm')
        threshold: Threshold for rouge evaluation (default: 0.6)
    """

    # Set default threshold based on method
    if threshold is None:
        if eval_method == 'rouge':
            threshold = 0.6

    # Separate answerable and unanswerable
    answerable_results = [r for r in results if r.get('answerable', False)]
    unanswerable_results = [r for r in results if not r.get('answerable', False)]

    # Evaluate answerable questions
    answerable_stats = {'total': 0, 'correct': 0, 'with_idk': 0, 'accuracy': 0.0, 'skipped': 0}
    scores = []

    if answerable_results:
        print(f"\nEvaluating {len(answerable_results)} answerable questions using {eval_method} method...")
        if eval_method == 'rouge':
            print(f"ROUGE threshold: {threshold}")

        for result in tqdm(answerable_results, desc="Evaluating answerable"):
            answer = result.get('answer', '')
            all_answers = result.get('all_answers', [])
            model_answer = result['model_answer']
            question = result['question']

            # Remove <IDK> for comparison
            contains_idk = check_idk_token(model_answer)
            clean_answer = model_answer.replace('<IDK>', '').replace('<|end_of_text|>', '').strip()
            result['has_idk'] = contains_idk

            # Skip evaluation if no ground truth answer
            if eval_method == 'llm':
                # For LLM, check single answer
                if not answer:
                    result['is_correct'] = None
                    result['eval_result'] = 'skipped_no_ground_truth'
                    result['eval_method'] = None
                    result['score'] = None
                    answerable_stats['skipped'] += 1
                    if contains_idk:
                        answerable_stats['with_idk'] += 1
                    continue
            else:
                # For ROUGE, check all_answers
                if not all_answers:
                    result['is_correct'] = None
                    result['eval_result'] = 'skipped_no_ground_truth'
                    result['eval_method'] = None
                    result['score'] = None
                    answerable_stats['skipped'] += 1
                    if contains_idk:
                        answerable_stats['with_idk'] += 1
                    continue

            # Evaluate based on selected method
            if eval_method == 'llm':
                if judge_model is None or judge_tokenizer is None:
                    raise ValueError("judge_model and judge_tokenizer are required for llm evaluation method")
                eval_result, is_correct, score = evaluate_answer_llm(
                    judge_model, judge_tokenizer, question, answer, clean_answer
                )
            elif eval_method == 'rouge':
                eval_result, is_correct, score = evaluate_answer_rouge(
                    all_answers, clean_answer, threshold
                )
            else:
                raise ValueError(f"Unknown evaluation method: {eval_method}")

            result['is_correct'] = is_correct
            result['eval_result'] = eval_result
            result['eval_method'] = eval_method
            result['score'] = score
            if threshold is not None and eval_method == 'rouge':
                result['threshold'] = threshold

            # Only append score if it's not None (handles skipped_no_ground_truth cases)
            if score is not None:
                scores.append(score)

            if is_correct:
                answerable_stats['correct'] += 1
            if contains_idk:
                answerable_stats['with_idk'] += 1

        answerable_stats['total'] = len(answerable_results)
        evaluatable = answerable_stats['total'] - answerable_stats['skipped']
        answerable_stats['accuracy'] = round((answerable_stats['correct'] / evaluatable * 100), 2) if evaluatable > 0 else 0.0
        answerable_stats['idk_rate'] = round((answerable_stats['with_idk'] / answerable_stats['total'] * 100), 2) if answerable_stats['total'] > 0 else 0.0
        answerable_stats['average_score'] = round(sum(scores) / len(scores), 4) if scores else 0.0

    # Evaluate unanswerable questions (IDK is correct)
    unanswerable_stats = {'total': len(unanswerable_results), 'idk_count': 0, 'accuracy': 0.0}
    if unanswerable_results:
        unanswerable_stats['idk_count'] = sum(1 for r in unanswerable_results if r['has_idk'])
        unanswerable_stats['accuracy'] = round((unanswerable_stats['idk_count'] / unanswerable_stats['total'] * 100), 2) if unanswerable_stats['total'] > 0 else 0.0

    # Compute per-dataset statistics
    dataset_stats = {}
    # Define desired order: NEC, RefuNQ, selfAware
    dataset_order = ['NEC', 'RefuNQ', 'selfAware']
    available_datasets = [d for d in dataset_order if any(r['dataset'] == d for r in results)]
    for dataset in available_datasets:
        dataset_results = [r for r in results if r['dataset'] == dataset]
        ds_answerable = [r for r in dataset_results if r.get('answerable', False)]
        ds_unanswerable = [r for r in dataset_results if not r.get('answerable', False)]

        # For answerable questions, break down by IDK presence
        answerable_with_idk = [r for r in ds_answerable if r.get('has_idk', False)]
        answerable_without_idk = [r for r in ds_answerable if not r.get('has_idk', False)]

        # Count correctly evaluated (not skipped) results
        ds_answerable_evaluated = [r for r in ds_answerable if r.get('is_correct') is not None]
        answerable_with_idk_evaluated = [r for r in answerable_with_idk if r.get('is_correct') is not None]
        answerable_without_idk_evaluated = [r for r in answerable_without_idk if r.get('is_correct') is not None]

        answerable_stats = {
            'total': len(ds_answerable),
            'correct': sum(1 for r in ds_answerable_evaluated if r.get('is_correct', False)),
            'skipped': len(ds_answerable) - len(ds_answerable_evaluated),
            'accuracy': round((sum(1 for r in ds_answerable_evaluated if r.get('is_correct', False)) / len(ds_answerable_evaluated) * 100), 2) if len(ds_answerable_evaluated) > 0 else 0.0,
            # IDK presence breakdown
            'with_idk': {
                'total': len(answerable_with_idk),
                'correct': sum(1 for r in answerable_with_idk_evaluated if r.get('is_correct', False)),
                'incorrect': sum(1 for r in answerable_with_idk_evaluated if r.get('is_correct') == False),
                'skipped': len(answerable_with_idk) - len(answerable_with_idk_evaluated),
                'accuracy': round((sum(1 for r in answerable_with_idk_evaluated if r.get('is_correct', False)) / len(answerable_with_idk_evaluated) * 100), 2) if len(answerable_with_idk_evaluated) > 0 else 0.0
            },
            'without_idk': {
                'total': len(answerable_without_idk),
                'correct': sum(1 for r in answerable_without_idk_evaluated if r.get('is_correct', False)),
                'incorrect': sum(1 for r in answerable_without_idk_evaluated if r.get('is_correct') == False),
                'skipped': len(answerable_without_idk) - len(answerable_without_idk_evaluated),
                'accuracy': round((sum(1 for r in answerable_without_idk_evaluated if r.get('is_correct', False)) / len(answerable_without_idk_evaluated) * 100), 2) if len(answerable_without_idk_evaluated) > 0 else 0.0
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
        'eval_method': eval_method,
        'threshold': threshold if eval_method == 'rouge' else None,
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
    print(f"\nEvaluation Method: {stats.get('eval_method', 'llm')}")
    if stats.get('threshold') is not None:
        print(f"Threshold: {stats['threshold']}")
    print(f"\nOverall Statistics:")
    print(f"  Total questions: {stats['total']}")

    # Answerable statistics
    answerable = stats['answerable']
    print(f"\n  Answerable Questions ({answerable['total']}):")
    if answerable['total'] > 0:
        evaluatable = answerable['total'] - answerable.get('skipped', 0)
        print(f"    Correct answers: {answerable['correct']}/{evaluatable} ({answerable['accuracy']:.2f}%)")
        if answerable.get('average_score') is not None:
            print(f"    Average score: {answerable['average_score']:.4f}")
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
    parser.add_argument(
        '--eval_method',
        type=str,
        default='llm',
        choices=['llm', 'rouge'],
        help='Evaluation method for answerable questions: llm (default) or rouge'
    )
    parser.add_argument(
        '--data_eval_method',
        type=str,
        default='llm',
        choices=['llm', 'rouge', 'em'],
        help='Evaluation method for answerable questions: llm (default) or rouge'
    )
    parser.add_argument(
        '--threshold',
        type=str,
        default="",
        help='Threshold for rouge evaluation (default: 0.6 for rouge)'
    )
    parser.add_argument(
        '--data_threshold',
        type=str,
        default="",
        help='Threshold for rouge evaluation (default: 0.6 for rouge)'
    )    

    parser.add_argument('--dataname', type=str, choices=['halueval', 'medqa', 'sciq'], help='Dataset name (halueval or medqa)')
    parser.add_argument('--save_run_name', type=str, default="sample_weighted", help='WandB run name (optional)')
    parser.add_argument('--sft_idk_weight', type=float, default=0.5, help='SFT IDK weight used in training')

    # Add model type selection
    parser.add_argument('--model_type', type=str, default='trained', choices=['trained', 'instruct', 'seal'],
                       help='Model type: trained (default) or instruct')
    parser.add_argument('--instruct_method', type=str, default='idk', choices=['idk', 'no_idk', 'default'],
                       help='Instruct model method: idk (default), no_idk, or default')
    parser.add_argument('--fewshot', type=int, default=3, help='Number of few-shot examples for instruct model (default: 3)')

    args = parser.parse_args()

    # Set model_path based on model_type
    if args.data_eval_method in ["llm", "em"]:
        args.data_threshold = ""
    else:
        args.data_threshold = float(args.data_threshold)

    if args.eval_method == "llm":
        args.threshold = ""
    else:
        args.threshold = float(args.threshold)        

    if args.model_type == 'trained':
        args.model_path = f"/mnt/frdata/rungjoo/hall/halu_model/{args.dataname}/{args.save_run_name}/sw_{args.data_eval_method}{args.data_threshold}_idk{args.sft_idk_weight}"
    elif args.model_type == "instruct":
        args.model_path = '../../model/Llama-3.2-3B-Instruct'
    elif args.model_type == "seal":
        args.model_path = f"/mnt/frdata/rungjoo/hall/halu_model/{args.dataname}/{args.save_run_name}/seal"

    # python3 05_evaluate_idk.py --dataset all --sft_idk_weight 1.0 --data_eval_method llm --eval_method llm --model_type trained
    # python3 05_evaluate_idk.py --dataset all --eval_method llm --model_type instruct --instruct_method idk    

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

    # Add eval method and threshold to filename
    eval_suffix = f"_{args.eval_method}"
    if args.eval_method == 'rouge' and args.threshold is not None:
        eval_suffix += f"{args.threshold}"    

    # Run inference
    print(f"\nRunning inference with model: {args.model_path}")
    if args.model_type == 'instruct':
        results = inference_instruct_model(args.model_path, all_data, device=args.device, method=args.instruct_method, fewshot=args.fewshot)
        model_name = f"instruct_{args.instruct_method}"
        result_file_path = f"{model_name}_idk_results{eval_suffix}.jsonl"
        stats_file_path = f"{model_name}_idk_evaluation{eval_suffix}.json"
    elif args.model_type == 'trained':
        results = inference_our_model(args.model_path, all_data, device=args.device)
        p = Path(args.model_path)
        model_name = f"{p.parent.name}_{p.name}"
        result_file_path = f"{model_name}_{args.dataname}_idk_results{eval_suffix}.jsonl"
        stats_file_path = f"{model_name}_{args.dataname}_idk_evaluation{eval_suffix}.json"
    elif args.model_type == 'seal':
        results = inference_our_model(args.model_path, all_data, device=args.device)
        model_name = "seal"
        result_file_path = f"{model_name}_{args.dataname}_idk_results{eval_suffix}.jsonl"
        stats_file_path = f"{model_name}_{args.dataname}_idk_evaluation{eval_suffix}.json"

    # Load judge model for evaluating answerable questions (only for llm method)
    judge_model = None
    judge_tokenizer = None
    if answerable_count > 0 and args.eval_method == 'llm':
        print(f"\nLoading judge model from {args.judge_model_path}")
        judge_model = AutoModelForCausalLM.from_pretrained(
            args.judge_model_path,
            device_map='auto'
        )
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model_path)
        if judge_tokenizer.pad_token is None:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token
    elif answerable_count > 0 and args.eval_method == 'rouge':
        print(f"\nUsing ROUGE method - no judge model loading required")

    # Evaluate results
    print(f"\nEvaluating results using {args.eval_method} method...")
    stats = evaluate_results(results, args.eval_method, judge_model, judge_tokenizer, args.threshold)

    # Save results AFTER evaluation (with is_correct field)
    print("\nSaving results (after evaluation with is_correct)...")
    results_file = output_dir / result_file_path
    save_results(results, str(results_file))

    # Save evaluation stats
    stats_file = output_dir / stats_file_path
    save_evaluation(stats, str(stats_file))

    # Save per-dataset results
    print("\nSaving per-dataset results...")
    dataset_names = set(r['dataset'] for r in results)

    # Print summary
    print_summary(stats)

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
