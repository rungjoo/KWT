#!/usr/bin/env python3
"""Judge test-set generations of fine-tuned models (output of evaluate_results.py).

Each item is labeled correct/incorrect (LLM-as-a-judge or ROUGE-L) after removing
<IDK>, and `had_idk_token` records whether <IDK> was generated. These two fields
define the A/B/C/D confusion matrix used by compute_metrics.py."""
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import re
from rouge_score import rouge_scorer

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto"
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def normalize_text(text: str) -> str:
    """Simple text normalization: lowercase, strip whitespace, collapse multiple spaces"""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def check_idk_token(text: str) -> bool:
    """Check if text contains <IDK> token"""
    return "<IDK>" in text

def evaluate_answer_llm(model, tokenizer, question, knowledge, right_answer, model_answer):
    """Use LLM to evaluate if model_answer matches right_answer"""

    # First check for exact match after normalization
    right_answer_norm = normalize_text(right_answer)
    model_answer_norm = normalize_text(model_answer)

    # Empty model answer should always be incorrect
    if not model_answer_norm:
        return 'incorrect', False, 0.0

    if right_answer_norm == model_answer_norm:
        return 'correct', True, 1.0

    # Use LLM to evaluate semantic equivalence
    prompt = f"""Question: {question}
Knowledge: {knowledge.strip()}

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
        return 'correct', True, 1.0
    else:
        return 'incorrect', False, 0.0

def evaluate_answer_rouge(right_answer, model_answer, threshold=0.5):
    """Use ROUGE score to evaluate if model_answer matches right_answer"""
    right_answer = normalize_text(right_answer)
    model_answer = normalize_text(model_answer)

    # Empty model answer should always be incorrect
    if not model_answer:
        return 'incorrect', False, 0.0

    if right_answer == model_answer:
        return 'correct', True, 1.0

    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(right_answer, model_answer)

    # Use ROUGE-L F1 score as the main metric
    rougeL_f1 = scores['rougeL'].fmeasure

    is_match = rougeL_f1 >= threshold
    return 'correct' if is_match else 'incorrect', is_match, rougeL_f1


def process_results_file(file_path, model, tokenizer, eval_method='llm', threshold=None):
    """Evaluate a results file

    Args:
        file_path: Path to the JSON or JSONL file
        model: Model for LLM evaluation (None if not using llm method)
        tokenizer: Tokenizer for LLM evaluation (None if not using llm method)
        eval_method: Evaluation method ('llm' or 'rouge')
        threshold: Threshold for rouge (default: 0.6)
    """

    print(f"\nProcessing file: {file_path}")
    print(f"Evaluation method: {eval_method}")

    if eval_method == 'rouge' and threshold is None:
        threshold = 0.6

    if eval_method == 'rouge':
        print(f"Threshold (Rouge): {threshold}")

    # Load data - support both JSONL and JSON formats
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

        # Try to parse as single JSON object first
        try:
            if file_path.endswith('.json'):
                # JSON format
                json_data = json.loads(content)
                if 'results' in json_data:
                    data = json_data['results']
                else:
                    data = [json_data]
            else:
                # JSONL format
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        except json.JSONDecodeError:
            # Fallback to JSONL parsing
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    data.append(json.loads(line))

    total = len(data)
    correct = 0
    incorrect = 0
    avoided_hallucination = 0

    # Detailed breakdown by IDK presence
    correct_with_idk = 0
    correct_without_idk = 0
    incorrect_with_idk = 0
    incorrect_without_idk = 0

    results = []
    scores = []

    print(f"Total samples: {total}")
    print("\nEvaluating answers...")

    for item in tqdm(data, desc="Evaluating"):
        # Get answers - use filtered_model_answer for evaluation
        right_answer = item.get('right_answer', '')
        hallucinated_answer = item.get('hallucinated_answer', '')
        filtered_model_answer = item.get('filtered_model_answer', '')

        # Fallback to model_answer if filtered_model_answer doesn't exist
        if not filtered_model_answer:
            filtered_model_answer = item.get('model_answer', '')

        # Check if answer contains <IDK> token before cleaning
        contains_idk = check_idk_token(filtered_model_answer)

        # Remove <|end_of_text|> token from the answer
        cleaned_answer = filtered_model_answer.replace('<|end_of_text|>', '').strip()
        cleaned_answer = cleaned_answer.replace('<|endoftext|>', '').strip()

        # Also remove <IDK> for comparison purposes
        answer_for_comparison = cleaned_answer.replace('<IDK>', '').strip()

        # Skip if essential data is missing
        if not right_answer or not filtered_model_answer:
            results.append({
                **item,
                'evaluation': 'skipped',
                'category': 'skipped',
                'had_idk_token': contains_idk,
                'score': None
            })
            continue

        # Evaluate against right answer using cleaned answer
        knowledge = item.get('knowledge', '')
        question = item.get('question', '')

        # Evaluate based on selected method
        if eval_method == 'llm':
            eval_result, is_correct, score = evaluate_answer_llm(model, tokenizer, question, knowledge, right_answer, answer_for_comparison)
        elif eval_method == 'rouge':
            eval_result, is_correct, score = evaluate_answer_rouge(right_answer, answer_for_comparison, threshold)
        else:
            raise ValueError(f"Unknown evaluation method: {eval_method}")

        scores.append(score)

        # Count results - now track both correctness and IDK presence
        if eval_result == 'correct':
            correct += 1
            if contains_idk:
                # Correct answer with IDK token
                correct_with_idk += 1
                avoided_hallucination += 1
            else:
                # Correct answer without IDK token
                correct_without_idk += 1
        else:
            incorrect += 1
            if contains_idk:
                # Incorrect but tried to avoid with IDK
                incorrect_with_idk += 1
                avoided_hallucination += 1
            else:
                # Incorrect without IDK token
                incorrect_without_idk += 1

        # Store result
        results.append({
            **item,
            'evaluation': eval_result,
            'had_idk_token': contains_idk,
            'cleaned_answer': cleaned_answer,
            'answer_for_comparison': answer_for_comparison,
            'score': score
        })

    # Calculate statistics
    accuracy = (correct / total * 100) if total > 0 else 0
    avoidance_rate = (avoided_hallucination / total * 100) if total > 0 else 0
    incorrect_rate = (incorrect / total * 100) if total > 0 else 0

    avg_score = sum(scores) / len(scores) if scores else 0

    # Count answers with IDK token
    idk_count = sum(1 for r in results if r.get('had_idk_token', False))
    idk_rate = (idk_count / total * 100) if total > 0 else 0

    print(f"\n" + "="*60)
    print(f"Results Summary:")
    print(f"  Evaluation method: {eval_method}")
    if eval_method == 'rouge':
        print(f"  Threshold (Rouge): {threshold}")
    print(f"  Total samples: {total}")
    print(f"\n  Overall:")
    print(f"    Correct answers: {correct} ({accuracy:.2f}%)")
    print(f"    Incorrect answers: {incorrect} ({incorrect_rate:.2f}%)")
    print(f"    Average score: {avg_score:.4f}")
    print(f"\n  Breakdown by <IDK> presence:")
    print(f"    Correct WITH <IDK>: {correct_with_idk} ({correct_with_idk/total*100:.2f}%)")
    print(f"    Correct WITHOUT <IDK>: {correct_without_idk} ({correct_without_idk/total*100:.2f}%)")
    print(f"    Incorrect WITH <IDK>: {incorrect_with_idk} ({incorrect_with_idk/total*100:.2f}%)")
    print(f"    Incorrect WITHOUT <IDK>: {incorrect_without_idk} ({incorrect_without_idk/total*100:.2f}%)")
    print(f"\n  <IDK> Statistics:")
    print(f"    Total with <IDK>: {idk_count} ({idk_rate:.2f}%)")
    print(f"    Attempted avoidance (had IDK): {avoided_hallucination} ({avoidance_rate:.2f}%)")
    print("="*60)

    # Save detailed results
    threshold_str = str(threshold) if eval_method == 'rouge' else ''
    output_file = file_path.replace('.jsonl', f'_evaluated_{eval_method}{threshold_str}.json')
    if '.jsonl' not in file_path:
        # Handle both .json and other extensions
        base_name = file_path.rsplit('.', 1)[0]
        # Remove any existing _evaluated_* suffix
        base_name = base_name.replace('_evaluated_rouge', '').replace('_evaluated_llm', '')
        output_file = f'{base_name}_evaluated_{eval_method}{threshold_str}.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'eval_method': eval_method,
                'threshold': threshold if eval_method == 'rouge' else None,
                'total': total,
                'correct': correct,
                'incorrect': incorrect,
                'avoided_hallucination': avoided_hallucination,
                'idk_count': idk_count,
                'correct_with_idk': correct_with_idk,
                'correct_without_idk': correct_without_idk,
                'incorrect_with_idk': incorrect_with_idk,
                'incorrect_without_idk': incorrect_without_idk,
                'accuracy': accuracy,
                'incorrect_rate': incorrect_rate,
                'avoidance_rate': avoidance_rate,
                'idk_rate': idk_rate,
                'correct_with_idk_rate': (correct_with_idk / total * 100) if total > 0 else 0,
                'correct_without_idk_rate': (correct_without_idk / total * 100) if total > 0 else 0,
                'incorrect_with_idk_rate': (incorrect_with_idk / total * 100) if total > 0 else 0,
                'incorrect_without_idk_rate': (incorrect_without_idk / total * 100) if total > 0 else 0,
                'average_score': avg_score
            },
            'results': results  # Save all results for inspection
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_file}")

    return accuracy, incorrect_rate, avoidance_rate

def main():
    parser = argparse.ArgumentParser(description='Judge test-set generations of fine-tuned models')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Output of evaluate_results.py (e.g. halueval/llama-3.2-3b/sample_weight_reverse_smooth_llm_idk0.16.jsonl)')
    parser.add_argument('--eval_method', type=str, default='llm', choices=['llm', 'rouge'],
                        help='Evaluation method: llm (default, used in the paper) or rouge')
    parser.add_argument('--model_path', type=str, default='google/gemma-3-12b-it',
                        help='Judge model (needed for the llm method)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='ROUGE-L threshold for the rouge method (default: 0.6)')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return

    model = None
    tokenizer = None
    if args.eval_method == 'llm':
        model, tokenizer = load_model(args.model_path)
    else:
        print(f"Using {args.eval_method} method - no model loading required")

    accuracy, incorrect_rate, avoidance_rate = process_results_file(
        args.input_file,
        model,
        tokenizer,
        eval_method=args.eval_method,
        threshold=args.threshold
    )

    print(f"\nEvaluation complete!")
    print(f"Final metrics:")
    print(f"  - Accuracy: {accuracy:.2f}%")
    print(f"  - Incorrect rate: {incorrect_rate:.2f}%")
    print(f"  - Avoidance rate: {avoidance_rate:.2f}%")

if __name__ == "__main__":
    main()
