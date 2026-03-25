import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path
import random
import torch.nn.functional as F

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
            ex_knowledge = ex.get('knowledge') or ex.get('support')
            prompt += f"Question: {ex['question']}\n"
            answer = ex.get('right_answer', ex.get('correct_answer', ex.get('answer', '')))
            prompt += f"Answer: {answer}\n\n"

    prompt += f"Question: {question}\n"
    prompt += "Answer:"

    return prompt


def create_few_shot_prompt_idk(question, examples=None):
    prompt = "If the answer is unknown or not in your knowledge, answer with <IDK>.\n"

    if examples:
        prompt += "Here are some examples:\n\n"
        # Randomly select one example to add <IDK>
        idk_idx = random.randint(0, len(examples) - 1)

        for idx, ex in enumerate(examples):
            ex_knowledge = ex.get('knowledge') or ex.get('support')
            prompt += f"Question: {ex['question']}\n"
            answer = ex.get('right_answer', ex.get('correct_answer', ex.get('answer', '')))
            # Add <IDK> to randomly selected example
            if idx == idk_idx:
                prompt += f"Answer: {answer} <IDK>\n\n"
            else:
                prompt += f"Answer: {answer}\n\n"

    prompt += f"Question: {question}\n"
    prompt += "Answer:"

    return prompt

def create_instruct_our_promt(question):
    prompt = f"Question: {question}\n\nAnswer:"
    return prompt

def create_instruct_our_promt_knowledge(question, knowledge):
    prompt = f"Knowledge: {knowledge}\n\nQuestion: {question}\n\nAnswer:"
    return prompt

def calculate_idk_probability(model, tokenizer, output_ids, scores):
    """
    Calculate the probability of <IDK> token appearing right before the EOS token

    Args:
        model: The language model
        tokenizer: The tokenizer
        output_ids: Generated token IDs (shape: [1, seq_len])
        scores: List of logits for each generation step

    Returns:
        float: Probability of <IDK> token before EOS, -1 if not applicable
    """
    # Find <IDK> token ID
    idk_token = "<IDK>"
    idk_token_id = None

    # Try to find the IDK token in the tokenizer
    try:
        idk_token_id = tokenizer.convert_tokens_to_ids(idk_token)
    except:
        # If direct conversion fails, try encoding
        encoded = tokenizer.encode(idk_token, add_special_tokens=False)
        if len(encoded) > 0:
            idk_token_id = encoded[0]

    if idk_token_id is None:
        return -1.0

    # Find the position of EOS token in the generated sequence
    eos_token_id = tokenizer.eos_token_id
    generated_ids = output_ids[0].tolist()

    # Find the last EOS position
    try:
        eos_position = len(generated_ids) - 1 - generated_ids[::-1].index(eos_token_id)
    except ValueError:
        # EOS not found, use the last position
        eos_position = len(generated_ids) - 1

    # Calculate which step in 'scores' corresponds to the token before EOS
    # scores[i] contains logits for predicting token at position i (in the generated sequence)
    # We want the logits that predicted the <IDK> token (which should be right before EOS)
    if eos_position == 0 or len(scores) == 0:
        return -1.0

    # The token before EOS is at position (eos_position - 1)
    # We need scores for predicting that token, which is at scores[eos_position - 1]
    token_before_eos_position = eos_position - 1
    score_index = token_before_eos_position - (len(generated_ids) - len(scores))

    if score_index < 0 or score_index >= len(scores):
        return -1.0

    # Verify the token before EOS is actually <IDK>
    if generated_ids[token_before_eos_position] != idk_token_id:
        return -1.0  # <IDK> is not right before EOS

    # Get logits for predicting the <IDK> token
    logits = scores[score_index][0]  # scores[i] has shape [batch_size, vocab_size]

    # Calculate probability using softmax
    probs = F.softmax(logits, dim=-1)
    idk_prob = probs[idk_token_id].item()

    return idk_prob


def calculate_idk_probability_sequence(tokenizer, output_ids, scores):
    """
    Calculate the probability of <IDK> token at each generation step.

    Args:
        tokenizer: The tokenizer
        output_ids: Generated token IDs (shape: [1, seq_len])
        scores: List of logits for each generation step

    Returns:
        dict: Contains:
            - 'idk_probs': List of <IDK> probabilities at each step
            - 'generated_tokens': List of generated token strings
            - 'generated_token_ids': List of generated token IDs
    """
    # Find <IDK> token ID
    idk_token = "<IDK>"
    idk_token_id = None

    # Try to find the IDK token in the tokenizer
    try:
        idk_token_id = tokenizer.convert_tokens_to_ids(idk_token)
    except:
        # If direct conversion fails, try encoding
        encoded = tokenizer.encode(idk_token, add_special_tokens=False)
        if len(encoded) > 0:
            idk_token_id = encoded[0]

    if idk_token_id is None or len(scores) == 0:
        return {
            'idk_probs': [],
            'generated_tokens': [],
            'generated_token_ids': []
        }

    generated_ids = output_ids[0].tolist()
    idk_probs = []
    generated_tokens = []

    # Calculate <IDK> probability at each generation step
    for i, score in enumerate(scores):
        # score has shape [batch_size, vocab_size]
        logits = score[0]

        # Calculate probability using softmax
        probs = F.softmax(logits, dim=-1)
        idk_prob = probs[idk_token_id].item()
        idk_probs.append(idk_prob)

        # Get the actual generated token at this step
        if i < len(generated_ids):
            token_id = generated_ids[i]
            token_str = tokenizer.decode([token_id])
            generated_tokens.append(token_str)

    return {
        'idk_probs': idk_probs,
        'generated_tokens': generated_tokens,
        'generated_token_ids': generated_ids[:len(scores)]
    }

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
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Extract generated IDs and scores
        output_ids = outputs.sequences
        scores = outputs.scores  # List of tensors, one per generation step

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip()

        # Calculate IDK probability (original - only for last token before EOS)
        idk_prob = calculate_idk_probability(model, tokenizer, output_ids[:, inputs['input_ids'].shape[1]:], scores)

        # Calculate IDK probability sequence (for all tokens)
        idk_sequence = calculate_idk_probability_sequence(tokenizer, output_ids[:, inputs['input_ids'].shape[1]:], scores)

        # Extract first paragraph from model answer
        model_answer_full = answer
        filtered_model_answer = model_answer_full.split('\n')[0].strip() if model_answer_full else ''

        result = {
            'question': item['question'],
            'model_answer': model_answer_full,
            'filtered_model_answer': filtered_model_answer,
            'prompt': prompt,
            'idk_probability': idk_prob,
            'idk_prob_sequence': idk_sequence['idk_probs'],
            'generated_tokens': idk_sequence['generated_tokens']
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

        # Add optional fields
        if 'hallucinated_answer' in item:
            result['hallucinated_answer'] = item['hallucinated_answer']

        results.append(result)

    return results


def inference_instruct_model(model_path, data, fewshot=3, device='cuda', method="default"):
    print(f"Loading instruct model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    few_shot_examples = data[:fewshot]

    results = []

    for item in tqdm(data, desc="Instruct model inference"):
        if method == "idk":
            prompt = create_few_shot_prompt_idk(item['question'], examples=few_shot_examples)
        elif method in ["no_idk", "sft_fewshot"]:
            prompt = create_few_shot_prompt(item['question'], examples=few_shot_examples)
        elif method == "knowledge":
            knowledge = item.get('knowledge') or item.get('support')
            prompt = create_instruct_our_promt_knowledge(item['question'], knowledge=knowledge)
        else: # default
            prompt = create_instruct_our_promt(item['question'])

        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        prompt_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Extract generated IDs and scores
        output_ids = outputs.sequences
        scores = outputs.scores  # List of tensors, one per generation step

        # Decode only the generated tokens (excluding the prompt)
        generated_tokens = output_ids[0][prompt_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()

        # Calculate IDK probability (original - only for last token before EOS)
        idk_prob = calculate_idk_probability(model, tokenizer, output_ids[:, prompt_length:], scores)

        # Calculate IDK probability sequence (for all tokens)
        idk_sequence = calculate_idk_probability_sequence(tokenizer, output_ids[:, prompt_length:], scores)

        # Extract first paragraph from model answer
        model_answer_full = answer
        filtered_model_answer = model_answer_full.split('\n')[0].strip() if model_answer_full else ''

        result = {
            'question': item['question'],
            'model_answer': model_answer_full,
            'filtered_model_answer': filtered_model_answer,
            'prompt': prompt,
            'idk_probability': idk_prob,
            'idk_prob_sequence': idk_sequence['idk_probs'],
            'generated_tokens': idk_sequence['generated_tokens']
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

        # Add optional fields
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
    # python3 01_eval_advanced.py --dataname halueval --save_run_name sample_weight_reverse_smooth --sft_idk_weight 0.16
    # python3 01_eval_advanced.py --dataname halueval --save_run_name seal
    parser = argparse.ArgumentParser(description='Inference on train.jsonl using two models')
    parser.add_argument('--dataname', type=str, required=True, choices=['halueval', 'medqa', 'sciq'], help='Dataset name (halueval or medqa)')
    parser.add_argument('--base_model', type=str, default='../../model/Llama-3.2-3B', help='Path to base model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--fewshot', type=int, default=3, help='The number of fewshot samples')
    parser.add_argument('--save_run_name', type=str, default="sample_weighted", help='WandB run name (optional)')
    # sample_weighted / sample_uniform / rtuning_r / sample_weighted_reverse / no_idk, idk

    parser.add_argument('--sft_idk_weight', type=float, default=0.5, help='SFT IDK weight used in training')

    parser.add_argument('--method', type=str, default='default')
    # knowledge

    parser.add_argument('--data_eval_method', type=str, default="llm")
    parser.add_argument('--threshold', type=str, default="",
                       help='Threshold for bertscore/rouge (default: 0.7 for bertscore, 0.6 for rouge)')

    args = parser.parse_args()

    # Set data_path and output_dir based on dataname
    model_name = get_model_name(args.base_model)
    if args.dataname == 'halueval':        
        args.data_path = '../dataset/halueval/halueval_test.jsonl'
        args.output_dir = f'halueval/{model_name}'
    elif args.dataname == 'medqa':
        args.data_path = '../dataset/medqa/test.jsonl'
        args.output_dir = f'medqa/{model_name}'
    elif args.dataname == 'sciq':
        args.data_path = '../dataset/sciq/test.jsonl'
        args.output_dir = f'sciq/{model_name}'

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.data_path)
    print(f"Loaded {len(data)} examples from {args.data_path}")

    if args.data_eval_method in ["llm", "em"]:
        args.threshold = ""
    else:
        args.threshold = float(args.threshold)

    # print("\n=== Running Base Model (Few-shot) ===")
    # base_results = inference_base_model(args.base_model, data, args.fewshot, device=args.device)
    # save_results(base_results, f"{args.output_dir}/base_model.jsonl")

    # print("\n=== Running SFT Model ===")
    # args.sft_model = f'../halu_model/{args.dataname}/sft'
    # sft_results = inference_instruct_model(args.sft_model, data, fewshot=args.fewshot, device=args.device, method="sft_fewshot")
    # save_results(sft_results, f"{args.output_dir}/sft_3shot.jsonl")

    # print("\n=== Running Instruct Model ===")
    # args.inst_model = f"../../model/Llama-3.2-3B-Instruct"
    # sft_results = inference_instruct_model(args.inst_model, data, fewshot=args.fewshot, device=args.device, method=args.save_run_name)
    # save_results(sft_results, f"{args.output_dir}/instruct_{args.save_run_name}.jsonl")

    # print("\n=== Running SEAL Model ===")
    # args.seal_model = f"/mnt/frdata/rungjoo/hall/halu_model/{args.dataname}/{args.save_run_name}/seal"
    # sft_results = inference_instruct_model(args.seal_model, data, fewshot=args.fewshot, device=args.device, method=args.method)
    # save_results(sft_results, f"{args.output_dir}/{args.save_run_name}_prob.jsonl")

    # print("\n=== Running our Model ===")
    args.our_model = f"/mnt/frdata/rungjoo/hall/halu_model/{args.dataname}/{model_name}/{args.save_run_name}/sw_{args.data_eval_method}{args.threshold}_idk{args.sft_idk_weight}"
    sft_results = inference_instruct_model(args.our_model, data, fewshot=args.fewshot, device=args.device, method=args.method)
    if args.method == "default":
        save_file_name = f"{args.output_dir}/{args.save_run_name}_{args.data_eval_method}{args.threshold}_idk{args.sft_idk_weight}_prob.jsonl"
    else:
        save_file_name = f"{args.output_dir}/{args.save_run_name}_{args.data_eval_method}{args.threshold}_idk{args.sft_idk_weight}_{args.method}_prob.jsonl"
    save_results(sft_results, save_file_name)

    print("\nInference completed!")

if __name__ == "__main__":
    main()
