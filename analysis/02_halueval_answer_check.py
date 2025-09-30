#!/usr/bin/env python3
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import re

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

def evaluate_answer(model, tokenizer, question, knowledge, right_answer, model_answer):
    """Use LLM to evaluate if model_answer matches right_answer"""

    # First check for exact match after normalization
    right_answer_norm = normalize_text(right_answer)
    model_answer_norm = normalize_text(model_answer)

    if right_answer_norm == model_answer_norm:
        return 'correct', True

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
        return 'correct', True
    else:
        return 'incorrect', False


def process_results_file(file_path, model, tokenizer):
    """Process a HaluEval results JSONL file and evaluate answers"""

    print(f"\nProcessing file: {file_path}")

    # Load JSONL data
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    total = len(data)
    correct = 0
    incorrect = 0
    avoided_hallucination = 0
    results = []

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

        # Also remove <IDK> for comparison purposes
        answer_for_comparison = cleaned_answer.replace('<IDK>', '').strip()

        # Skip if essential data is missing
        if not right_answer or not filtered_model_answer:
            results.append({
                **item,
                'evaluation': 'skipped',
                'category': 'skipped',
                'had_idk_token': contains_idk
            })
            continue

        # Evaluate against right answer using cleaned answer
        knowledge = item.get('knowledge', '')
        question = item.get('question', '')

        eval_result, is_correct = evaluate_answer(model, tokenizer, question, knowledge, right_answer, answer_for_comparison)

        # Count results - now track both correctness and IDK presence
        if eval_result == 'correct':
            if contains_idk:
                # Correct answer with IDK token
                correct += 1
                avoided_hallucination += 1
            else:
                # Correct answer without IDK token
                correct += 1
        else:
            if contains_idk:
                # Incorrect but tried to avoid with IDK
                avoided_hallucination += 1
            incorrect += 1

        # Store result
        results.append({
            **item,
            'evaluation': eval_result,
            'had_idk_token': contains_idk,
            'cleaned_answer': cleaned_answer,
            'answer_for_comparison': answer_for_comparison
        })

    # Calculate statistics
    accuracy = (correct / total * 100) if total > 0 else 0
    avoidance_rate = (avoided_hallucination / total * 100) if total > 0 else 0
    incorrect_rate = (incorrect / total * 100) if total > 0 else 0

    # Count answers with IDK token
    idk_count = sum(1 for r in results if r.get('had_idk_token', False))
    idk_rate = (idk_count / total * 100) if total > 0 else 0

    print(f"\n" + "="*60)
    print(f"Results Summary:")
    print(f"  Total samples: {total}")
    print(f"  Correct answers: {correct} ({accuracy:.2f}%)")
    print(f"  Incorrect answers: {incorrect} ({incorrect_rate:.2f}%)")
    print(f"  Answers with <IDK> token: {idk_count} ({idk_rate:.2f}%)")
    print(f"  Attempted avoidance (had IDK): {avoided_hallucination} ({avoidance_rate:.2f}%)")
    print("="*60)

    # Save detailed results
    output_file = file_path.replace('.jsonl', '_evaluated.json')
    if '.jsonl' not in file_path:
        output_file = file_path.rsplit('.', 1)[0] + '_evaluated.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': total,
                'correct': correct,
                'incorrect': incorrect,
                'avoided_hallucination': avoided_hallucination,
                'idk_count': idk_count,
                'accuracy': accuracy,
                'incorrect_rate': incorrect_rate,
                'avoidance_rate': avoidance_rate,
                'idk_rate': idk_rate
            },
            'results': results  # Save all results for inspection
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_file}")

    return accuracy, incorrect_rate, avoidance_rate

def main():
    # python3 02_halueval_answer_check.py --input_file halueval/llama_3.2-3b/sft.jsonl
    # python3 02_halueval_answer_check.py --input_file halueval/llama_3.2-3b/sft_dpo.jsonl
    parser = argparse.ArgumentParser(description='Evaluate HaluEval model answers using LLM')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to HaluEval JSONL file (e.g., halueval/llama_3.2-3b/base_model.jsonl)')
    parser.add_argument('--model_path', type=str,
                       default='../../model/gemma-3-12b-it',
                       help='Path to evaluation model (e.g., gemma or llama instruct model)')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return

    # Load evaluation model
    model, tokenizer = load_model(args.model_path)

    # Process the results file
    accuracy, incorrect_rate, avoidance_rate = process_results_file(args.input_file, model, tokenizer)

    print(f"\nEvaluation complete!")
    print(f"Final metrics:")
    print(f"  - Accuracy: {accuracy:.2f}%")
    print(f"  - Incorrect rate: {incorrect_rate:.2f}%")
    print(f"  - Avoidance rate: {avoidance_rate:.2f}%")

if __name__ == "__main__":
    main()