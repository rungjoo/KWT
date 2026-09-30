"""Knowledge estimation step 1: multi-sampled few-shot inference with the base model.

For every question, S responses are sampled, each with independently resampled
3-shot demonstrations (paper Sec. 3.1). With --greedy, a single greedy response is
generated instead (used to build R-Tuning's known/unknown split).
"""
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import KNOWLEDGE_DIR, get_model_name, split_file, KNOWLEDGE_FILE_PREFIX, GREEDY_FILE_PREFIX

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
    parser = argparse.ArgumentParser(description='Multi-sampled few-shot inference with the base model')
    parser.add_argument('--dataname', type=str, required=True, choices=['halueval', 'medqa', 'sciq'])
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.2-3B', help='Path or HF id of the base model')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help='train: knowledge scores for fine-tuning; test: knowledge scores for analysis (Fig. 1)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fewshot', type=int, default=3, help='Number of few-shot demonstrations')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--num_samples', type=int, default=5, help='Number of sampled responses per question (S)')
    parser.add_argument('--greedy', action='store_true', help='Single greedy response (for R-Tuning)')
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    model_name = get_model_name(args.base_model)
    data_path = split_file(args.dataname, args.split)
    output_dir = KNOWLEDGE_DIR / args.dataname / model_name
    if args.split != 'train':
        output_dir = output_dir / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(data_path)
    print(f"Loaded {len(data)} examples from {data_path}")

    if args.greedy:
        print("\n=== Running Base Model (Few-shot with Greedy) ===")
        results = inference_greedy_base_model(args.base_model, data, fewshot=args.fewshot,
                                              num_samples=1, device=args.device)
        output_file = output_dir / f"{GREEDY_FILE_PREFIX}.jsonl"
    else:
        print("\n=== Running Base Model (Few-shot with Sampling) ===")
        print(f"Temperature: {args.temperature}, Num Samples: {args.num_samples}, Few-shot: {args.fewshot}")
        results = inference_base_model(args.base_model, data, fewshot=args.fewshot, temperature=args.temperature,
                                       num_samples=args.num_samples, device=args.device)
        output_file = output_dir / f"base_model_temp{args.temperature}_samples{args.num_samples}_fewshot{args.fewshot}.jsonl"

    save_results(results, output_file)
    print("\nInference completed!")

if __name__ == "__main__":
    main()
