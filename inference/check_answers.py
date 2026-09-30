#!/usr/bin/env python3
"""Knowledge estimation step 2: judge every sampled response against the gold answer
with EM, ROUGE-L or LLM-as-a-judge. The per-question count of correct samples
(samples_correct / samples_total) is the knowledge score used for fine-tuning."""
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os, re
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

def evaluate_answer_llm(model, tokenizer, question, knowledge, right_answer, model_answer):
    """Use LLM to evaluate if model_answer matches right_answer"""
    right_answer = normalize_text(right_answer)
    model_answer = normalize_text(model_answer)
    if right_answer == model_answer:
        return True, 1.0

    # Create evaluation prompt
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
        return True, 1.0
    elif 'no' in response:
        return False, 0.0
    else:
        # If unclear, default to False
        print(f"  Warning: Unclear evaluation response: {response}")
        return False, 0.0

def evaluate_answer_rouge(right_answer, model_answer, threshold=0.5):
    """Use ROUGE score to evaluate if model_answer matches right_answer"""
    right_answer = normalize_text(right_answer)
    model_answer = normalize_text(model_answer)

    if right_answer == model_answer:
        return True, 1.0

    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(right_answer, model_answer)

    # Use ROUGE-L F1 score as the main metric
    rougeL_f1 = scores['rougeL'].fmeasure

    is_match = rougeL_f1 >= threshold
    return is_match, rougeL_f1

def evaluate_answer_em(right_answer, model_answer):
    """Use exact match to evaluate if model_answer matches right_answer"""
    right_answer = normalize_text(right_answer)
    model_answer = normalize_text(model_answer)

    is_match = right_answer == model_answer
    score = 1.0 if is_match else 0.0

    return is_match, score

def process_results_file(file_path, model, tokenizer, eval_method='llm', threshold=None):
    """Process a results JSONL file and evaluate answers

    Args:
        file_path: Path to the JSONL file
        model: Model for LLM evaluation (None if not using llm method)
        tokenizer: Tokenizer for LLM evaluation (None if not using llm method)
        eval_method: Evaluation method ('llm', 'rouge', 'em')
        threshold: ROUGE-L threshold (not used for llm/em)
    """

    print(f"\nProcessing file: {file_path}")
    print(f"Evaluation method: {eval_method}")

    if eval_method == 'rouge' and threshold is None:
        threshold = 0.35

    if eval_method == 'rouge':
        print(f"Threshold: {threshold}")

    # Load JSONL data
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if not (data and 'samples' in data[0]):
        raise ValueError(f"{file_path} is not an output of run_inference.py (missing 'samples' field)")
    return process_samples_format(data, model, tokenizer, eval_method, threshold, file_path)

def process_samples_format(data, model, tokenizer, eval_method, threshold, file_path):
    """Evaluate every sampled response of every question"""
    total_questions = len(data)
    total_samples = 0
    correct_samples = 0
    results = []
    all_scores = []

    print(f"Total questions: {total_questions}")

    # Count total samples
    for item in data:
        total_samples += len(item.get('samples', []))

    print(f"Total samples: {total_samples}")
    print("\nEvaluating answers...")

    for item in tqdm(data, desc="Evaluating"):
        # Get right answer
        right_answer = item.get('right_answer', '')
        knowledge = item.get('knowledge', "")
        question = item['question']

        # Process each sample
        sample_results = []
        for idx, sample in enumerate(item.get('samples', [])):
            # Get model answer (split by \n\n and take first part)
            model_answer_full = sample.get('model_answer', '')
            filtered_model_answer = model_answer_full.split('\n\n')[0].strip() if model_answer_full else ''

            # Skip if either answer is missing
            if not right_answer or not filtered_model_answer:
                sample_results.append({
                    'sample_index': idx,
                    'model_answer': model_answer_full,
                    'filtered_model_answer': filtered_model_answer,
                    'prompt': sample.get('prompt', ''),
                    'evaluation': 'skipped',
                    'match': False,
                    'score': None
                })
                continue

            # Evaluate based on selected method
            if eval_method == 'llm':
                is_match, score = evaluate_answer_llm(model, tokenizer, question, knowledge, right_answer, filtered_model_answer)
            elif eval_method == 'rouge':
                is_match, score = evaluate_answer_rouge(right_answer, filtered_model_answer, threshold)
            elif eval_method == 'em':
                is_match, score = evaluate_answer_em(right_answer, filtered_model_answer)
            else:
                raise ValueError(f"Unknown evaluation method: {eval_method}")

            if is_match:
                correct_samples += 1

            all_scores.append(score)

            # Store sample result
            sample_results.append({
                'sample_index': idx,
                'model_answer': model_answer_full,
                'filtered_model_answer': filtered_model_answer,
                'prompt': sample.get('prompt', ''),
                'evaluation': 'match' if is_match else 'mismatch',
                'match': is_match,
                'score': score
            })

        # Store question result with all sample evaluations
        results.append({
            'question': question,
            'knowledge': knowledge,
            'right_answer': right_answer,
            'hallucinated_answer': item.get('hallucinated_answer', ''),
            'num_samples': item.get('num_samples', len(sample_results)),
            'temperature': item.get('temperature', None),
            'sample_evaluations': sample_results,
            'samples_correct': sum(1 for s in sample_results if s['match']),
            'samples_total': len(sample_results)
        })

    # Calculate accuracy and average score
    accuracy = (correct_samples / total_samples * 100) if total_samples > 0 else 0
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

    print(f"\n" + "="*50)
    print(f"Results Summary:")
    print(f"  Evaluation method: {eval_method}")
    if eval_method == 'rouge':
        print(f"  Threshold: {threshold}")
    print(f"  Total questions: {total_questions}")
    print(f"  Total samples: {total_samples}")
    print(f"  Correct samples: {correct_samples}")
    print(f"  Incorrect samples: {total_samples - correct_samples}")
    print(f"  Sample-level accuracy: {accuracy:.2f}%")
    print(f"  Average score: {avg_score:.4f}")
    print("="*50)

    # Save detailed results
    if eval_method == 'rouge':
        output_file = file_path.replace('.jsonl', f'_evaluated_{eval_method}{threshold}.json')
        if '.jsonl' not in file_path:
            output_file = file_path.rsplit('.', 1)[0] + f'_evaluated_{eval_method}{threshold}.json'
    else:
        output_file = file_path.replace('.jsonl', f'_evaluated_{eval_method}.json')
        if '.jsonl' not in file_path:
            output_file = file_path.rsplit('.', 1)[0] + f'_evaluated_{eval_method}.json'        

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'eval_method': eval_method,
                'threshold': threshold if eval_method == 'rouge' else None,
                'total_questions': total_questions,
                'total_samples': total_samples,
                'correct_samples': correct_samples,
                'incorrect_samples': total_samples - correct_samples,
                'sample_accuracy': accuracy,
                'average_score': avg_score
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_file}")

    return accuracy, correct_samples, total_samples

def main():
    parser = argparse.ArgumentParser(description='Judge sampled base-model responses (knowledge estimation)')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Output of run_inference.py, e.g. halueval/llama-3.2-3b/base_model_temp0.7_samples5_fewshot3.jsonl')
    parser.add_argument('--eval_method', type=str, default='llm', choices=['llm', 'rouge', 'em'],
                        help='Matching function: llm (LLM-as-a-judge), rouge (ROUGE-L >= threshold) or em (exact match)')
    parser.add_argument('--model_path', type=str, default='google/gemma-3-12b-it',
                        help='Judge model (only needed for the llm method)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='ROUGE-L threshold (paper: 0.35 for halueval, 0.6 for medqa/sciq)')

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

    process_results_file(args.input_file, model, tokenizer, eval_method=args.eval_method, threshold=args.threshold)
    print(f"\nEvaluation complete!")

if __name__ == "__main__":
    main()
