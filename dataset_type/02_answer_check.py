#!/usr/bin/env python3
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os, re

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

def evaluate_answer(model, tokenizer, question, knowledge, right_answer, model_answer):
    right_answer = normalize_text(right_answer) 
    model_answer = normalize_text(model_answer)
    if right_answer == model_answer:
        return True
    
    """Use LLM to evaluate if model_answer matches right_answer"""
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
    # inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
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
        return True
    elif 'no' in response:
        return False
    else:
        # If unclear, default to False
        print(f"  Warning: Unclear evaluation response: {response}")
        return False

def process_results_file(file_path, model, tokenizer):
    """Process a results JSONL file and evaluate answers"""
    
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
    results = []
    
    print(f"Total samples: {total}")
    print("\nEvaluating answers...")
    
    for item in tqdm(data, desc="Evaluating"):
        # Get right answer
        right_answer = item.get('right_answer', '')
        
        # Get model answer (split by \n\n and take first part)
        model_answer_full = item.get('model_answer', '')
        model_answer = model_answer_full.split('\n\n')[0].strip() if model_answer_full else ''
        
        # Skip if either answer is missing
        if not right_answer or not model_answer:
            print(f"  Skipping item due to missing answer")
            results.append({
                **item,
                'evaluation': 'skipped',
                'match': False
            })
            continue
        
        # Evaluate using LLM
        knowledge = item.get('knowledge', "")
        question = item['question']
        is_match = evaluate_answer(model, tokenizer, question, knowledge, right_answer, model_answer)
        
        if is_match:
            correct += 1
        
        # Store result
        results.append({
            **item,
            'filtered_model_answer': model_answer,
            'evaluation': 'match' if is_match else 'mismatch',
            'match': is_match
        })
    
    # Calculate accuracy
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n" + "="*50)
    print(f"Results Summary:")
    print(f"  Total samples: {total}")
    print(f"  Correct matches: {correct}")
    print(f"  Incorrect: {total - correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print("="*50)
    
    # Save detailed results
    output_file = file_path.replace('.jsonl', '_evaluated.json')
    if '.jsonl' not in file_path:
        output_file = file_path.rsplit('.', 1)[0] + '_evaluated.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': total,
                'correct': correct,
                'incorrect': total - correct,
                'accuracy': accuracy
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return accuracy, correct, total

def main():
    parser = argparse.ArgumentParser(description='Evaluate model answers using LLM')
    # python3 02_answer_check.py --input_file halueval/base_model_results.jsonl --model_path ../model/Llama-3.1-8B-Instruct
    # python3 02_answer_check.py --input_file halueval/instruct_model_results.jsonl
    # python3 02_answer_check.py --input_file halueval/self_sft_model_results.jsonl
    # python3 02_answer_check.py --input_file medqa/self_sft_model_results.jsonl
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to JSONL file (e.g., halueval/base_model_results.jsonl)')
    parser.add_argument('--model_path', type=str, 
                       default='../model/gemma-3-12b-it',
                       help='Path to Instruction model')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    # Process the results file
    accuracy, correct, total = process_results_file(args.input_file, model, tokenizer)
    
    print(f"\nEvaluation complete!")

if __name__ == "__main__":
    main()