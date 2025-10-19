import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path

def load_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def create_few_shot_prompt(question, examples=None):
    prompt = ""

    if examples:
        prompt += "Here are some examples:\n\n"
        for ex in examples:
            prompt += f"Question: {ex['question']}\n"
            answer_key = 'answer' if 'answer' in ex else ('correct_answer' if 'correct_answer' in ex else 'right_answer')
            prompt += f"Answer: {ex[answer_key]}\n\n"

    prompt += f"Question: {question}\n"
    prompt += "Answer:"

    return prompt

def create_instruct_prompt(question):
    prompt = f"""Answer the question briefly and accurately.

Question: {question}

Answer:"""
    return prompt

def inference_base_model(model_path, data, fewshot=3, device='cuda', batch_size=1):
    print(f"Loading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    few_shot_examples = data[:fewshot]
    
    results = []
    
    for item in tqdm(data, desc="Base model inference"):
        knowledge = item.get('knowledge') or item.get('support')
        prompt = create_few_shot_prompt(
            item['question'],
            examples=few_shot_examples
        )
        
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
        
        result = {
            'question': item['question'],
            'model_answer': answer,
            'prompt': prompt
        }

        # Add fields if they exist in the original data
        if 'knowledge' in item:
            result['knowledge'] = item['knowledge']
        elif 'support' in item:
            result['knowledge'] = item['support']

        if 'right_answer' in item:
            result['right_answer'] = item['right_answer']
        elif 'correct_answer' in item:
            result['right_answer'] = item['correct_answer']
        elif 'answer' in item:
            result['right_answer'] = item['answer']

        if 'hallucinated_answer' in item:
            result['hallucinated_answer'] = item['hallucinated_answer']
        if 'options' in item:
            result['options'] = item['options']
        if 'answer_idx' in item:
            result['answer_idx'] = item['answer_idx']

        results.append(result)
    
    return results

def inference_instruct_model(model_path, data, device='cuda', batch_size=1):
    print(f"Loading instruct model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    results = []
    
    for item in tqdm(data, desc="Instruct model inference"):
        prompt = create_instruct_prompt(item['question'])
        
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

        result = {
            'question': item['question'],
            'model_answer': answer,
            'prompt': prompt
        }

        # Add fields if they exist in the original data
        if 'knowledge' in item:
            result['knowledge'] = item['knowledge']
        elif 'support' in item:
            result['knowledge'] = item['support']

        if 'right_answer' in item:
            result['right_answer'] = item['right_answer']
        elif 'correct_answer' in item:
            result['right_answer'] = item['correct_answer']
        elif 'answer' in item:
            result['right_answer'] = item['answer']

        if 'hallucinated_answer' in item:
            result['hallucinated_answer'] = item['hallucinated_answer']

        results.append(result)
    
    return results


def save_results(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Results saved to {output_path}")

def main():
    # python3 01_inference_models.py --dataname halueval
    # python3 01_inference_models.py --dataname medqa
    # python3 01_inference_models.py --dataname sciq
    parser = argparse.ArgumentParser(description='Inference on train.jsonl using two models')
    parser.add_argument('--dataname', type=str, required=True, choices=['halueval', 'medqa', 'sciq'], help='Dataset name (halueval, medqa, or sciq)')
    parser.add_argument('--base_model', type=str, default='../../model/Llama-3.2-3B', help='Path to base model')
    parser.add_argument('--instruct_model', type=str, default='../../model/Llama-3.2-3B-Instruct', help='Path to instruct model')
    parser.add_argument('--self_sft_model', type=str, default='../ref_model/Llama-3.2-3B-SFT', help='Path to instruct model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--fewshot', type=int, default=3, help='The number of fewshot samples')
    parser.add_argument('--model', type=str, choices=['base', 'instruct', 'self_sft', 'all'], default='all', help='Which model to run')

    args = parser.parse_args()

    # Set data_path and output_dir based on dataname
    if args.dataname == 'halueval':
        args.data_path = '../dataset/new_sft/halueval/halueval_train.jsonl'
        args.output_dir = './halueval'
    elif args.dataname == 'medqa':
        args.data_path = '../dataset/new_sft/medqa/train.jsonl'
        args.output_dir = './medqa'
    elif args.dataname == 'sciq':
        args.data_path = '../dataset/new_sft/sciq/train.jsonl'
        args.output_dir = './sciq'
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    data = load_dataset(args.data_path)
    print(f"Loaded {len(data)} examples from {args.data_path}")
    
    if args.model in ['base', 'all']:
        print("\n=== Running Base Model (Few-shot) ===")
        base_results = inference_base_model(args.base_model, data, args.fewshot, device=args.device)
        save_results(base_results, f"{args.output_dir}/base_model_results.jsonl")
    
    if args.model in ['instruct', 'all']:
        print("\n=== Running Instruct Model ===")
        instruct_results = inference_instruct_model(args.instruct_model, data, device=args.device)
        save_results(instruct_results, f"{args.output_dir}/instruct_model_results.jsonl")

    if args.model in ['self_sft', 'all']:
        print("\n=== Running Self SFT Model ===")
        sft_results = inference_instruct_model(args.self_sft_model, data, device=args.device)
        save_results(sft_results, f"{args.output_dir}/self_sft_model_results.jsonl")
    
    print("\nInference completed!")

if __name__ == "__main__":
    main()