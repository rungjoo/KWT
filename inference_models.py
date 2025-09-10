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

def create_few_shot_prompt(knowledge, question, examples=None):
    prompt = ""
    
    if examples:
        prompt += "Here are some examples:\n\n"
        for ex in examples:
            # prompt += f"Knowledge: {ex['knowledge']}\n"
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Answer: {ex['right_answer']}\n\n"
    
    # prompt += f"Knowledge: {knowledge}\n"
    prompt += f"Question: {question}\n"
    prompt += "Answer:"
    
    return prompt

def create_instruct_prompt(knowledge, question):
    prompt = f"""Given the following knowledge, answer the question briefly and accurately.

Question: {question}

Answer:"""
    return prompt

def inference_base_model(model_path, data, fewshot=3, device='cuda', batch_size=1):
    print(f"Loading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    few_shot_examples = data[:fewshot]
    
    results = []
    
    for item in tqdm(data, desc="Base model inference"):
        prompt = create_few_shot_prompt(
            item['knowledge'], 
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
        
        results.append({
            'knowledge': item['knowledge'],
            'question': item['question'],
            'right_answer': item['right_answer'],
            'hallucinated_answer': item['hallucinated_answer'],
            'model_answer': answer,
            "prompt": prompt
        })
    
    return results

def inference_instruct_model(model_path, data, device='cuda', batch_size=1):
    print(f"Loading instruct model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    results = []
    
    for item in tqdm(data, desc="Instruct model inference"):
        prompt = create_instruct_prompt(item['knowledge'], item['question'])
        
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

        results.append({
            'knowledge': item['knowledge'],
            'question': item['question'],
            'right_answer': item['right_answer'],
            'hallucinated_answer': item['hallucinated_answer'],
            'model_answer': answer,
            'prompt': prompt
        })
    
    return results


def save_results(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Inference on train.jsonl using two models')
    parser.add_argument('--data_path', type=str, default='dataset/new_sft/halueval_train.jsonl', help='Path to train.jsonl')
    parser.add_argument('--base_model', type=str, default='../model/Llama-3.2-3B', help='Path to base model')
    parser.add_argument('--instruct_model', type=str, default='../model/Llama-3.2-3B-Instruct', help='Path to instruct model')
    parser.add_argument('--self_sft_model', type=str, default='./model/Llama-3.2-3B-SFT', help='Path to instruct model')
    parser.add_argument('--output_dir', type=str, default='dataset_llama_3.2-3b', help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--fewshot', type=int, default=3, help='The number of fewshot samples')
    parser.add_argument('--model', type=str, choices=['base', 'instruct', 'self_sft', 'all'], default='both', help='Which model to run')
    
    args = parser.parse_args()
    
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