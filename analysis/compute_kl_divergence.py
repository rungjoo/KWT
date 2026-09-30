"""
Token-level KL divergence between the base model and fine-tuned models (paper Sec. 5.5, Table 12).

KL(M_base || M_trained) is averaged over the gold response tokens of every test question.
Reports Base-vs-SFT and Base-vs-Ours ("ours" is KWT by default, or SEAL with
--save_run_name seal), also split by whether "ours" answers with <IDK>.

Usage:
    python compute_kl_divergence.py --dataname halueval                           # SFT and KWT
    python compute_kl_divergence.py --dataname halueval --save_run_name seal      # SFT and SEAL
"""

import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CKPT_DIR, RESULT_DIR, get_model_name, split_file, resolve_threshold, checkpoint_dir, result_stem

def load_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def has_idk_response(model_answer):
    """Check if model response contains <IDK>"""
    if model_answer is None:
        return False
    return '<IDK>' in model_answer or '<idk>' in model_answer.lower()


def create_prompt(question):
    """Create prompt for the model (same as evaluate_results.py)"""
    prompt = f"Question: {question}\n\nAnswer:"
    return prompt


def get_response_text(item):
    """Get the ground truth response text"""
    return item.get('right_answer', item.get('correct_answer', item.get('answer', '')))


def generate_response(model, tokenizer, prompt, max_new_tokens=50):
    """Generate response from model"""
    device = get_model_device(model)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    prompt_length = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode only the generated tokens (excluding the prompt)
    generated_tokens = outputs[0][prompt_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()

    return answer


def get_model_device(model):
    """Get the device of the model's first parameter."""
    return next(model.parameters()).device


def compute_log_probs_and_kl(model1, model2, tokenizer, prompt, response):
    """
    Compute KL divergence between two models for a given prompt+response.
    KL(model1 || model2) - how different model1's distribution is from model2's

    Returns per-token KL and mean KL for the response portion.
    """
    # Tokenize prompt and response
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
    response_ids = tokenizer.encode(response, add_special_tokens=False, return_tensors='pt')

    if response_ids.shape[1] == 0:
        return None, None, None

    # Combine prompt + response
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    prompt_len = prompt_ids.shape[1]

    # Get device for each model and run forward pass separately
    device1 = get_model_device(model1)
    device2 = get_model_device(model2)

    with torch.no_grad():
        outputs1 = model1(full_ids.to(device1))
        outputs2 = model2(full_ids.to(device2))

        # Move logits to CPU for comparison
        logits1 = outputs1.logits.cpu()
        logits2 = outputs2.logits.cpu()

    # Get logits for response tokens (position i predicts token i+1)
    # For response starting at prompt_len, we use logits from prompt_len-1 to end-1
    response_logits1 = logits1[:, prompt_len-1:-1, :]  # [1, response_len, vocab_size]
    response_logits2 = logits2[:, prompt_len-1:-1, :]

    # Handle different vocab sizes (e.g., model with <IDK> token has +1 vocab)
    # Truncate to the smaller vocab size for comparison
    min_vocab_size = min(response_logits1.shape[-1], response_logits2.shape[-1])
    response_logits1 = response_logits1[:, :, :min_vocab_size]
    response_logits2 = response_logits2[:, :, :min_vocab_size]

    # Compute probabilities
    probs1 = F.softmax(response_logits1, dim=-1)
    log_probs1 = F.log_softmax(response_logits1, dim=-1)
    log_probs2 = F.log_softmax(response_logits2, dim=-1)

    # KL(P1 || P2) = sum over vocab(P1 * (log P1 - log P2))
    kl_per_token = (probs1 * (log_probs1 - log_probs2)).sum(dim=-1)  # [1, response_len]
    kl_per_token = kl_per_token.squeeze(0).numpy()  # [response_len]

    mean_kl = float(np.mean(kl_per_token))

    # Also get the target token log probs for analysis
    target_tokens = full_ids[0, prompt_len:].numpy()

    return kl_per_token, mean_kl, target_tokens


def analyze_models(base_model, sft_model, our_model,
                   base_tokenizer, our_tokenizer,
                   test_data, max_samples=None):
    """
    Analyze KL divergence between models.
    Runs inference on our_model to determine <IDK> responses on-the-fly.

    Args:
        base_model: Base model
        sft_model: SFT model
        our_model: Our model (with <IDK> token)
        base_tokenizer: Tokenizer for base/sft models
        our_tokenizer: Tokenizer for our model (has <IDK> token)
        test_data: Test dataset
        max_samples: Max samples to process

    Returns:
        results: List of per-sample results
        stats: Dictionary with aggregated statistics
    """
    eval_data = test_data

    if max_samples:
        eval_data = eval_data[:max_samples]

    results = []

    # KL values grouped by IDK response
    kl_base_sft_all = []
    kl_base_our_all = []
    kl_base_our_idk = []      # Our model responded with <IDK>
    kl_base_our_no_idk = []   # Our model responded without <IDK>
    kl_base_sft_idk = []      # SFT KL for samples where Our model said <IDK>
    kl_base_sft_no_idk = []   # SFT KL for samples where Our model didn't say <IDK>

    idk_count = 0
    total_count = 0

    for item in tqdm(eval_data, desc="Computing KL divergence"):
        prompt = create_prompt(item['question'])
        response = get_response_text(item)

        if not response or len(response.strip()) == 0:
            continue

        our_answer = None
        is_idk = None
        kl_mean_sft = None
        kl_mean_our = None
        error_msg = None

        try:
            # Generate response from our model to check <IDK> (use our_tokenizer)
            our_answer = generate_response(our_model, our_tokenizer, prompt)
            is_idk = has_idk_response(our_answer)

            total_count += 1
            if is_idk:
                idk_count += 1

            # Compute KL(Base || SFT) using base_tokenizer
            kl_tokens_sft, kl_mean_sft, _ = compute_log_probs_and_kl(
                base_model, sft_model, base_tokenizer, prompt, response
            )

            # Compute KL(Base || Our) using base_tokenizer for fair comparison
            kl_tokens_our, kl_mean_our, tokens = compute_log_probs_and_kl(
                base_model, our_model, base_tokenizer, prompt, response
            )

            if kl_mean_sft is None or kl_mean_our is None:
                error_msg = "KL computation returned None (empty response tokens)"
            else:
                # Collect statistics
                kl_base_sft_all.append(kl_mean_sft)
                kl_base_our_all.append(kl_mean_our)

                if is_idk:
                    kl_base_our_idk.append(kl_mean_our)
                    kl_base_sft_idk.append(kl_mean_sft)
                else:
                    kl_base_our_no_idk.append(kl_mean_our)
                    kl_base_sft_no_idk.append(kl_mean_sft)

        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"Error processing: {e}")
            traceback.print_exc()

        # Always store per-sample result (including errors)
        result = {
            'question': item['question'],
            'prompt': prompt,
            'gold_response': response,
            'our_answer': our_answer,
            'is_idk': is_idk,
            'kl_base_sft': round(kl_mean_sft, 2) if kl_mean_sft is not None else None,
            'kl_base_our': round(kl_mean_our, 2) if kl_mean_our is not None else None,
            'kl_diff': round(kl_mean_our - kl_mean_sft, 2) if (kl_mean_our is not None and kl_mean_sft is not None) else None,
        }
        if error_msg:
            result['error'] = error_msg
        results.append(result)

    if total_count > 0:
        print(f"\n<IDK> responses: {idk_count}/{total_count} ({100*idk_count/total_count:.1f}%)")
    else:
        print(f"\n<IDK> responses: {idk_count}/{total_count}")

    # Compute aggregate statistics
    def compute_stats(values, name):
        if len(values) == 0:
            return {f'{name}_count': 0}
        arr = np.array(values)
        return {
            f'{name}_count': len(arr),
            f'{name}_mean': round(float(np.mean(arr)), 2),
            f'{name}_std': round(float(np.std(arr)), 2),
            f'{name}_min': round(float(np.min(arr)), 2),
            f'{name}_max': round(float(np.max(arr)), 2),
            f'{name}_median': round(float(np.median(arr)), 2),
        }

    stats = {}
    stats.update(compute_stats(kl_base_sft_all, 'kl_base_sft_all'))
    stats.update(compute_stats(kl_base_our_all, 'kl_base_our_all'))
    stats.update(compute_stats(kl_base_our_idk, 'kl_base_our_idk'))
    stats.update(compute_stats(kl_base_our_no_idk, 'kl_base_our_no_idk'))
    stats.update(compute_stats(kl_base_sft_idk, 'kl_base_sft_idk'))
    stats.update(compute_stats(kl_base_sft_no_idk, 'kl_base_sft_no_idk'))

    # IDK statistics
    stats['idk_count'] = idk_count
    stats['total_count'] = total_count
    stats['idk_ratio'] = round(idk_count / total_count, 2) if total_count > 0 else 0

    # Compute comparison metrics
    if len(kl_base_our_idk) > 0 and len(kl_base_our_no_idk) > 0:
        stats['idk_vs_no_idk_diff'] = round(float(np.mean(kl_base_our_idk) - np.mean(kl_base_our_no_idk)), 2)

    if len(kl_base_sft_idk) > 0 and len(kl_base_our_idk) > 0:
        stats['our_vs_sft_on_idk_samples'] = round(float(np.mean(kl_base_our_idk) - np.mean(kl_base_sft_idk)), 2)

    if len(kl_base_sft_no_idk) > 0 and len(kl_base_our_no_idk) > 0:
        stats['our_vs_sft_on_no_idk_samples'] = round(float(np.mean(kl_base_our_no_idk) - np.mean(kl_base_sft_no_idk)), 2)

    return results, stats


def print_statistics(stats):
    """Print formatted statistics"""
    print("\n" + "="*70)
    print("KL DIVERGENCE ANALYSIS RESULTS")
    print("="*70)

    print(f"\nTotal samples: {stats.get('total_count', 0)}")
    print(f"<IDK> responses: {stats.get('idk_count', 0)} ({100*stats.get('idk_ratio', 0):.1f}%)")

    print("\n[1] Overall Base vs SFT:")
    print(f"    Samples: {stats.get('kl_base_sft_all_count', 0)}")
    print(f"    Mean KL: {stats.get('kl_base_sft_all_mean', 0):.4f} ± {stats.get('kl_base_sft_all_std', 0):.4f}")

    print("\n[2] Overall Base vs Our:")
    print(f"    Samples: {stats.get('kl_base_our_all_count', 0)}")
    print(f"    Mean KL: {stats.get('kl_base_our_all_mean', 0):.4f} ± {stats.get('kl_base_our_all_std', 0):.4f}")

    print("\n[3] Base vs Our Model (samples where Our model said <IDK>):")
    print(f"    Samples: {stats.get('kl_base_our_idk_count', 0)}")
    if stats.get('kl_base_our_idk_count', 0) > 0:
        print(f"    Mean KL: {stats.get('kl_base_our_idk_mean', 0):.4f} ± {stats.get('kl_base_our_idk_std', 0):.4f}")
        print(f"    (SFT on same samples: {stats.get('kl_base_sft_idk_mean', 0):.4f})")

    print("\n[4] Base vs Our Model (samples where Our model did NOT say <IDK>):")
    print(f"    Samples: {stats.get('kl_base_our_no_idk_count', 0)}")
    if stats.get('kl_base_our_no_idk_count', 0) > 0:
        print(f"    Mean KL: {stats.get('kl_base_our_no_idk_mean', 0):.4f} ± {stats.get('kl_base_our_no_idk_std', 0):.4f}")
        print(f"    (SFT on same samples: {stats.get('kl_base_sft_no_idk_mean', 0):.4f})")

    print("\n[5] Comparisons:")
    if 'idk_vs_no_idk_diff' in stats:
        diff = stats['idk_vs_no_idk_diff']
        direction = "higher" if diff > 0 else "lower"
        print(f"    Our model's KL on <IDK> samples is {abs(diff):.4f} {direction} than on non-<IDK> samples")

    if 'our_vs_sft_on_idk_samples' in stats:
        diff = stats['our_vs_sft_on_idk_samples']
        direction = "higher" if diff > 0 else "lower"
        print(f"    On <IDK> samples: Our KL is {abs(diff):.4f} {direction} than SFT KL")

    if 'our_vs_sft_on_no_idk_samples' in stats:
        diff = stats['our_vs_sft_on_no_idk_samples']
        direction = "higher" if diff > 0 else "lower"
        print(f"    On non-<IDK> samples: Our KL is {abs(diff):.4f} {direction} than SFT KL")

    print("="*70)


def save_results(results, stats, output_path):
    """Save results to JSON file"""
    output = {
        'statistics': stats,
        'samples': results
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='KL Divergence Analysis')
    parser.add_argument('--dataname', type=str, required=True, choices=['halueval', 'medqa', 'sciq'])
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-3.2-3B', help='Path or HF id of the base model')
    parser.add_argument('--save_run_name', type=str, default='sample_weight_reverse_smooth',
                        help='Training strategy of "ours" (e.g. sample_weight_reverse_smooth, seal)')
    parser.add_argument('--data_eval_method', type=str, default='llm', choices=['llm', 'rouge', 'em'])
    parser.add_argument('--threshold', type=str, default=None)
    parser.add_argument('--sft_idk_weight', type=float, default=0.16)
    parser.add_argument('--our_model_path', type=str, default=None, help='Explicit checkpoint path of "ours"')
    parser.add_argument('--sft_model_path', type=str, default=None, help='Explicit checkpoint path of SFT')
    parser.add_argument('--ckpt_root', type=str, default=str(CKPT_DIR))
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples to process (for testing)')
    parser.add_argument('--output_dir', type=str, default=str(RESULT_DIR / 'kl_divergence'))

    args = parser.parse_args()

    args.data_path = split_file(args.dataname, 'test')
    model_name = get_model_name(args.base_model)
    threshold = resolve_threshold(args.dataname, args.data_eval_method, args.threshold)
    sft_model_path = args.sft_model_path or str(checkpoint_dir(args.ckpt_root, args.dataname, model_name, 'sft'))
    our_model_path = args.our_model_path or str(checkpoint_dir(args.ckpt_root, args.dataname, model_name, args.save_run_name,
                                                               args.data_eval_method, threshold, args.sft_idk_weight))
    args.our_model_name = result_stem(args.save_run_name, args.data_eval_method, threshold, args.sft_idk_weight)

    print(f"Base model: {args.base_model}")
    print(f"SFT model: {sft_model_path}")
    print(f"Our model: {our_model_path}")

    # Create output directory
    output_dir = Path(args.output_dir) / args.dataname
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    test_data = load_dataset(args.data_path)
    print(f"\nLoaded {len(test_data)} test examples")

    # Load models
    print(f"\nLoading Base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map='auto'
    )
    base_model.eval()

    print(f"Loading SFT model: {sft_model_path}")
    sft_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path, device_map='auto'
    )
    sft_model.eval()

    print(f"Loading Our model: {our_model_path}")
    our_model = AutoModelForCausalLM.from_pretrained(
        our_model_path, device_map='auto'
    )
    our_model.eval()

    # Load tokenizers
    # Base tokenizer for base/sft models and KL computation
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_tokenizer.pad_token = base_tokenizer.eos_token

    # Our tokenizer for our model inference (has <IDK> token)
    our_tokenizer = AutoTokenizer.from_pretrained(our_model_path)
    our_tokenizer.pad_token = our_tokenizer.eos_token

    # Run analysis
    results, stats = analyze_models(
        base_model, sft_model, our_model,
        base_tokenizer, our_tokenizer,
        test_data,
        max_samples=args.max_samples
    )

    # Add metadata to stats
    stats['base_model'] = args.base_model
    stats['sft_model'] = sft_model_path
    stats['our_model'] = our_model_path
    stats['our_model_name'] = args.our_model_name
    stats['dataname'] = args.dataname

    # Print and save results
    print_statistics(stats)

    output_file = output_dir / f'kl_analysis_{args.our_model_name}.json'
    save_results(results, stats, output_file)

    print("\nDone!")


if __name__ == "__main__":
    main()
