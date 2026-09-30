"""Generate answers on the in-domain test set with a fine-tuned checkpoint.

The output (<dataname>/<model_name>/<result_stem>.jsonl) is then judged with
check_answers.py and summarized with compute_metrics.py.
"""
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CKPT_DIR, RESULT_DIR, get_model_name, split_file, resolve_threshold, checkpoint_dir, result_stem

def load_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def create_prompt(question):
    return f"Question: {question}\n\nAnswer:"

def create_prompt_knowledge(question, knowledge):
    """Knowledge-provided prompt (paper Sec. 5.1)"""
    return f"Knowledge: {knowledge}\n\nQuestion: {question}\n\nAnswer:"

def inference_trained_model(model_path, data, device='cuda', method="default"):
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    results = []

    for item in tqdm(data, desc="Inference"):
        if method == "knowledge":
            knowledge = item.get('knowledge') or item.get('support')
            prompt = create_prompt_knowledge(item['question'], knowledge=knowledge)
        else:
            prompt = create_prompt(item['question'])

        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        prompt_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode only the generated tokens; keep special tokens so that <IDK> is visible
        generated_tokens = outputs[0][prompt_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()

        # Keep only the first line of the answer
        model_answer_full = answer
        filtered_model_answer = model_answer_full.split('\n')[0].strip() if model_answer_full else ''

        result = {
            'question': item['question'],
            'model_answer': model_answer_full,
            'filtered_model_answer': filtered_model_answer,
            'prompt': prompt
        }

        # Add knowledge field (handle both 'knowledge' and 'support')
        if 'knowledge' in item:
            result['knowledge'] = item['knowledge']
        elif 'support' in item:
            result['knowledge'] = item['support']

        # Add right_answer field (handle different field names)
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
    parser = argparse.ArgumentParser(description='Test-set inference with a fine-tuned checkpoint')
    parser.add_argument('--dataname', type=str, required=True, choices=['halueval', 'medqa', 'sciq'])
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.2-3B',
                        help='Base model the checkpoint was trained from (only its name is used)')
    parser.add_argument('--save_run_name', type=str, default="sample_weight_reverse_smooth",
                        help='Training strategy used in training/ (e.g. sample_weight_reverse_smooth, sft, seal, '
                             'sample_weight_reverse_smooth_noidk)')
    parser.add_argument('--data_eval_method', type=str, default="llm", choices=['llm', 'rouge', 'em'],
                        help='Matching function used for the training knowledge scores')
    parser.add_argument('--threshold', type=str, default=None,
                        help='ROUGE-L threshold used in training (default: 0.35 for halueval, 0.6 for medqa/sciq)')
    parser.add_argument('--sft_idk_weight', type=float, default=0.16, help='sft_idk_weight used in training')
    parser.add_argument('--run_id', type=int, default=0, help='Run ID for repeated SFT runs (0 means no suffix)')
    parser.add_argument('--method', type=str, default='default', choices=['default', 'knowledge'],
                        help='knowledge: prepend the supporting knowledge paragraph to the prompt')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Explicit checkpoint path (overrides the path derived from the options above)')
    parser.add_argument('--ckpt_root', type=str, default=str(CKPT_DIR))
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    model_name = get_model_name(args.base_model)
    data_path = split_file(args.dataname, 'test')
    output_dir = RESULT_DIR / args.dataname / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(data_path)
    print(f"Loaded {len(data)} examples from {data_path}")

    threshold = resolve_threshold(args.dataname, args.data_eval_method, args.threshold)
    model_path = args.model_path or checkpoint_dir(args.ckpt_root, args.dataname, model_name, args.save_run_name,
                                                   args.data_eval_method, threshold, args.sft_idk_weight, args.run_id)
    stem = result_stem(args.save_run_name, args.data_eval_method, threshold, args.sft_idk_weight, args.run_id)
    if args.method != "default":
        stem = f"{stem}_{args.method}"

    results = inference_trained_model(str(model_path), data, device=args.device, method=args.method)
    save_results(results, output_dir / f"{stem}.jsonl")

    print("\nInference completed!")

if __name__ == "__main__":
    main()
