#!/usr/bin/env python3
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os, re
from bert_score import score as bert_score
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
    """간단한 문자열 정규화: 소문자 변환, 앞뒤 공백 제거, 다중 공백 축소"""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)  # 여러 공백을 하나로
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

def evaluate_answer_bertscore(right_answer, model_answer, threshold=0.85):
    """Use BERTScore to evaluate if model_answer matches right_answer"""
    right_answer = normalize_text(right_answer)
    model_answer = normalize_text(model_answer)

    if right_answer == model_answer:
        return True, 1.0

    # Calculate BERTScore with GPU support
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    P, R, F1 = bert_score([model_answer], [right_answer], lang='en', verbose=False, device=device)
    f1_score = F1.item()

    is_match = f1_score >= threshold
    return is_match, f1_score

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
        eval_method: Evaluation method ('llm', 'bertscore', 'rouge', 'em')
        threshold: Threshold for bertscore/rouge (default: 0.7 for bertscore, 0.6 for rouge, not used for em)
    """

    print(f"\nProcessing file: {file_path}")
    print(f"Evaluation method: {eval_method}")

    # Set default threshold based on method
    if threshold is None:
        if eval_method == 'bertscore':
            threshold = 0.7
        elif eval_method == 'rouge':
            threshold = 0.35

    if eval_method in ['bertscore', 'rouge'] and threshold is not None:
        print(f"Threshold: {threshold}")

    # Load JSONL data
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    # Check if data has the new format (with 'samples' field)
    has_samples = data and 'samples' in data[0]
    if has_samples:
        print(f"Detected new format with multiple samples per question")
        return process_samples_format(data, model, tokenizer, eval_method, threshold, file_path)
    else:
        print(f"Detected old format with single answer per question")

def process_samples_format(data, model, tokenizer, eval_method, threshold, file_path):
    """Process new format where each item has multiple samples"""
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
            elif eval_method == 'bertscore':
                is_match, score = evaluate_answer_bertscore(right_answer, filtered_model_answer, threshold)
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
    if eval_method in ['bertscore', 'rouge']:
        print(f"  Threshold: {threshold}")
    print(f"  Total questions: {total_questions}")
    print(f"  Total samples: {total_samples}")
    print(f"  Correct samples: {correct_samples}")
    print(f"  Incorrect samples: {total_samples - correct_samples}")
    print(f"  Sample-level accuracy: {accuracy:.2f}%")
    print(f"  Average score: {avg_score:.4f}")
    print("="*50)

    # Save detailed results    
    if eval_method in ['bertscore', 'rouge']:
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
                'threshold': threshold if eval_method in ['bertscore', 'rouge'] else None,
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
    parser = argparse.ArgumentParser(description='Evaluate model answers using various methods')
    # python3 02_answer_check.py --input_file halueval/qwen3-4b/base_model_temp0.7_samples5_fewshot3.jsonl --eval_method em
    # python3 02_answer_check.py --input_file medqa/qwen3-4b/base_model_temp0.7_samples5_fewshot3.jsonl --eval_method em
    # python3 02_answer_check.py --input_file sciq/qwen3-4b/base_model_temp0.7_samples5_fewshot3.jsonl --eval_method em

    # python3 02_answer_check.py --input_file halueval/qwen3-4b/base_model_temp0.7_samples5_fewshot3.jsonl --eval_method bertscore --threshold 0.7
    # python3 02_answer_check.py --input_file medqa/qwen3-4b/base_model_temp0.7_samples5_fewshot3.jsonl --eval_method bertscore --threshold 0.7
    # python3 02_answer_check.py --input_file sciq/qwen3-4b/base_model_temp0.7_samples5_fewshot3.jsonl --eval_method bertscore --threshold 0.7

    # python3 02_answer_check.py --input_file halueval/qwen3-4b/base_model_temp0.7_samples5_fewshot3.jsonl --eval_method rouge --threshold 0.35
    # python3 02_answer_check.py --input_file medqa/qwen3-4b/base_model_temp0.7_samples5_fewshot3.jsonl --eval_method rouge --threshold 0.35
    # python3 02_answer_check.py --input_file sciq/qwen3-4b/base_model_temp0.7_samples5_fewshot3.jsonl --eval_method rouge --threshold 0.35

    # python3 02_answer_check.py --input_file halueval/qwen3-4b/base_model_greedy_samples_fewshot3.jsonl --eval_method em
    # python3 02_answer_check.py --input_file medqa/qwen3-4b/base_model_greedy_samples_fewshot3.jsonl --eval_method em
    # python3 02_answer_check.py --input_file sciq/qwen3-4b/base_model_greedy_samples_fewshot3.jsonl --eval_method em
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to JSONL file (e.g., halueval/base_model_results.jsonl)')
    parser.add_argument('--eval_method', type=str, default='llm',
                       choices=['llm', 'bertscore', 'rouge', 'em'],
                       help='Evaluation method: llm (default), bertscore, rouge, or em (exact match)')
    parser.add_argument('--model_path', type=str,
                       default='../../model/gemma-3-12b-it',
                       help='Path to Instruction model (only needed for llm method)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Threshold for bertscore/rouge (default: 0.7 for bertscore, 0.35 for rouge)')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return

    # Load model only if using LLM method
    model = None
    tokenizer = None
    if args.eval_method == 'llm':
        model, tokenizer = load_model(args.model_path)
    else:
        print(f"Using {args.eval_method} method - no model loading required")

    # Process the results file
    accuracy, correct, total = process_results_file(
        args.input_file,
        model,
        tokenizer,
        eval_method=args.eval_method,
        threshold=args.threshold
    )

    print(f"\nEvaluation complete!")

if __name__ == "__main__":
    main()