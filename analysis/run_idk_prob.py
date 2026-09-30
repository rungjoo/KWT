"""Test-set inference that also records the <IDK> probability at every decoding step.

Produces <dataname>/<model_name>/<result_stem>_prob.jsonl, which visualize_idk_prob.py
turns into the prepend-IDK vs. append-IDK plots (paper Figures 2 and 3).
"""
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path
import torch.nn.functional as F

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
    return f"Knowledge: {knowledge}\n\nQuestion: {question}\n\nAnswer:"

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
    parser = argparse.ArgumentParser(description='Test-set inference with per-step <IDK> probabilities')
    parser.add_argument('--dataname', type=str, required=True, choices=['halueval', 'medqa', 'sciq'])
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.2-3B',
                        help='Base model the checkpoint was trained from (only its name is used)')
    parser.add_argument('--save_run_name', type=str, default="sample_weight_reverse_smooth",
                        help='e.g. sample_weight_reverse_smooth (append-IDK), sample_weighted_reverse_ridk (prepend-IDK), seal')
    parser.add_argument('--data_eval_method', type=str, default="llm", choices=['llm', 'rouge', 'em'])
    parser.add_argument('--threshold', type=str, default=None)
    parser.add_argument('--sft_idk_weight', type=float, default=0.16)
    parser.add_argument('--method', type=str, default='default', choices=['default', 'knowledge'])
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
                                                   args.data_eval_method, threshold, args.sft_idk_weight)
    stem = result_stem(args.save_run_name, args.data_eval_method, threshold, args.sft_idk_weight)
    if args.method != "default":
        stem = f"{stem}_{args.method}"

    results = inference_trained_model(str(model_path), data, device=args.device, method=args.method)
    save_results(results, output_dir / f"{stem}_prob.jsonl")

    print("\nInference completed!")

if __name__ == "__main__":
    main()
