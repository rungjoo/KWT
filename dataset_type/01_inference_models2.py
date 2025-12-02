import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path
import random

def get_model_name(model_path):
    """Extract model name from model path for directory naming"""
    return Path(model_path).name.lower()

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

def inference_base_model(model_path, data, fewshot=3, temperature=0.7, num_samples=1, device='cuda'):
    print(f"Loading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    results = []

    for item in tqdm(data, desc="Base model inference"):
        # Collect multiple samples for the same question
        samples = []

        for sample_idx in range(num_samples):
            # Randomly select few-shot examples for each sample
            available_examples = [ex for ex in data if ex['question'] != item['question']]
            if len(available_examples) >= fewshot:
                few_shot_examples = random.sample(available_examples, fewshot)
            else:
                few_shot_examples = available_examples

            prompt = create_few_shot_prompt(
                item['question'],
                examples=few_shot_examples
            )

            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,  # default value
                    top_p=1.0,  # default value (effectively disabled)
                    pad_token_id=tokenizer.pad_token_id
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text[len(prompt):].strip()

            samples.append({
                'model_answer': answer,
                'prompt': prompt
            })

        # Create a single result for this item with all samples
        result = {
            'question': item['question'],
            'samples': samples,
            'num_samples': num_samples,
            'temperature': temperature
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


def inference_greedy_base_model(model_path, data, fewshot=3, num_samples=1, device='cuda'):
    """
    Inference with greedy decoding (temperature=0, do_sample=False)
    Uses the same few-shot setup as inference_base_model but generates deterministically
    """
    print(f"Loading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    results = []

    for item in tqdm(data, desc="Base model greedy inference"):
        # Collect multiple samples for the same question
        samples = []

        for sample_idx in range(num_samples):
            # Randomly select few-shot examples for each sample
            available_examples = [ex for ex in data if ex['question'] != item['question']]
            if len(available_examples) >= fewshot:
                few_shot_examples = random.sample(available_examples, fewshot)
            else:
                few_shot_examples = available_examples

            prompt = create_few_shot_prompt(
                item['question'],
                examples=few_shot_examples
            )

            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=None,  # Not used when do_sample=False
                    do_sample=False,  # Greedy decoding
                    pad_token_id=tokenizer.pad_token_id
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = generated_text[len(prompt):].strip()

            samples.append({
                'model_answer': answer,
                'prompt': prompt
            })

        # Create a single result for this item with all samples
        result = {
            'question': item['question'],
            'samples': samples,
            'num_samples': num_samples,
            'temperature': 0,  # Greedy decoding
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


def save_results(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Results saved to {output_path}")

def main():
    # python3 01_inference_models.py --dataname halueval --base_model ../../model/Qwen3-4B --temperature 0.7 --num_samples 5
    # python3 01_inference_models.py --dataname medqa --base_model ../../model/Qwen3-4B --temperature 0.7 --num_samples 5
    # python3 01_inference_models.py --dataname sciq --base_model ../../model/Qwen3-4B --temperature 0.7 --num_samples 5
    parser = argparse.ArgumentParser(description='Inference on train.jsonl using base model with sampling')
    parser.add_argument('--dataname', type=str, required=True, choices=['halueval', 'medqa', 'sciq'], help='Dataset name (halueval, medqa, or sciq)')
    parser.add_argument('--base_model', type=str, default='../../model/Qwen2.5-3B', help='Path to base model') # Llama-3.2-3B
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--fewshot', type=int, default=3, help='The number of fewshot samples')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling (default: 0.7)')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate per question (default: 1)')

    args = parser.parse_args()

    # Extract model name for directory structure
    model_name = get_model_name(args.base_model)

    # Set data_path and output_dir based on dataname and model
    if args.dataname == 'halueval':
        args.data_path = '../dataset/new_sft/halueval/halueval_train.jsonl'
        args.output_dir = f'./halueval/{model_name}'
    elif args.dataname == 'medqa':
        args.data_path = '../dataset/new_sft/medqa/train.jsonl'
        args.output_dir = f'./medqa/{model_name}'
    elif args.dataname == 'sciq':
        args.data_path = '../dataset/new_sft/sciq/train.jsonl'
        args.output_dir = f'./sciq/{model_name}'

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.data_path)
    print(f"Loaded {len(data)} examples from {args.data_path}")

    # print("\n=== Running Base Model (Few-shot with Sampling) ===")
    # print(f"Temperature: {args.temperature}, Num Samples: {args.num_samples}, Few-shot: {args.fewshot}")
    # base_results = inference_base_model(
    #     args.base_model,
    #     data,
    #     fewshot=args.fewshot,
    #     temperature=args.temperature,
    #     num_samples=args.num_samples,
    #     device=args.device
    # )

    # output_filename = f"base_model_temp{args.temperature}_samples{args.num_samples}_fewshot{args.fewshot}.jsonl"
    # save_results(base_results, f"{args.output_dir}/{output_filename}")

    print("\n=== Running Base Model (Few-shot with Greedy) ===")
    base_greedy_results = inference_greedy_base_model(
        args.base_model,
        data,
        fewshot=args.fewshot,
        num_samples=args.num_samples,
        device=args.device
    )

    output_filename = f"base_model_greedy_samples_fewshot{args.fewshot}.jsonl"
    save_results(base_greedy_results, f"{args.output_dir}/{output_filename}")

    print("\nInference completed!")

if __name__ == "__main__":
    main()