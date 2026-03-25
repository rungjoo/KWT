import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def analyze_idk_probability_sequence(data, min_length=5):
    """
    Analyze IDK probability sequences, separating by whether <IDK> was generated or not.
    Excludes the last token (EOS) from the sequence.
    Only includes sequences with length >= min_length.
    """
    idk_generated = []  # Cases where <IDK> was in the output
    idk_not_generated = []  # Cases where <IDK> was NOT in the output
    skipped_count = 0

    for item in data:
        probs = item.get('idk_prob_sequence', [])
        tokens = item.get('generated_tokens', [])
        model_answer = item.get('model_answer', '')

        if len(probs) == 0:
            continue

        # Exclude last token (EOS)
        if len(probs) > 1:
            probs = probs[:-1]
            tokens = tokens[:-1] if len(tokens) > 1 else tokens
        else:
            continue

        # Exclude leading whitespace token
        if len(tokens) > 0 and len(probs) > 0 and tokens[0].strip() == '':
            probs = probs[1:]
            tokens = tokens[1:]

        # Skip sequences shorter than min_length
        if len(probs) < min_length:
            skipped_count += 1
            continue

        # Check if <IDK> was generated
        has_idk = '<IDK>' in model_answer

        if has_idk:
            idk_generated.append(probs)
        else:
            idk_not_generated.append(probs)

    return idk_generated, idk_not_generated, skipped_count

def normalize_sequences_no_interp(sequences, num_bins=5):
    """
    Normalize sequences to fixed length bins WITHOUT interpolation.
    Divides each sequence into num_bins segments and takes the mean of each segment.
    Only uses actual values, no interpolation.
    """
    if len(sequences) == 0:
        return np.zeros(num_bins), np.zeros(num_bins), 0

    normalized = []
    for seq in sequences:
        seq_len = len(seq)
        if seq_len < num_bins:
            continue

        # Divide sequence into num_bins segments and take mean of each
        bin_values = []
        for i in range(num_bins):
            start_idx = int(i * seq_len / num_bins)
            end_idx = int((i + 1) * seq_len / num_bins)
            # Take mean of actual values in this segment
            segment_mean = np.mean(seq[start_idx:end_idx])
            bin_values.append(segment_mean)

        normalized.append(bin_values)

    if len(normalized) == 0:
        return np.zeros(num_bins), np.zeros(num_bins), 0

    normalized = np.array(normalized)
    mean = np.mean(normalized, axis=0)
    std = np.std(normalized, axis=0)

    return mean, std, len(normalized)

def plot_idk_probability_comparison(dataname, idk_generated, idk_not_generated, output_dir, num_bins=5):
    """
    Plot IDK probability changes comparing <IDK> generated vs not generated cases.
    Only shows Position in Response plot (excludes last token/EOS).
    """
    # Normalize sequences without interpolation
    idk_mean, idk_std, idk_count = normalize_sequences_no_interp(idk_generated, num_bins)
    no_idk_mean, no_idk_std, no_idk_count = normalize_sequences_no_interp(idk_not_generated, num_bins)

    # Create figure - single plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # X-axis: categorical labels for 3 bins
    x_labels = ['Early', 'Mid', 'Late']
    x = np.arange(num_bins)  # [0, 1, 2]

    if idk_count > 0:
        ax.plot(x, idk_mean, 'r-o', linewidth=2, markersize=10, label='<IDK> Generated')
        ax.fill_between(x, idk_mean - idk_std, idk_mean + idk_std, color='red', alpha=0.2)

    if no_idk_count > 0:
        ax.plot(x, no_idk_mean, 'b-o', linewidth=2, markersize=10, label='No <IDK>')
        ax.fill_between(x, no_idk_mean - no_idk_std, no_idk_mean + no_idk_std, color='blue', alpha=0.2)

    # Format dataset name
    title_names = {'halueval': 'HaluEval', 'medqa': 'MedQA', 'sciq': 'SciQ'}
    title = title_names.get(dataname, dataname)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=22)
    ax.set_xlabel('Position in Response', fontsize=24)
    ax.set_ylabel('<IDK> Token Probability', fontsize=24)
    ax.set_title(title, fontsize=28)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0, num_bins - 1)
    ax.set_ylim(0, max(np.max(idk_mean + idk_std) if idk_count > 0 else 0,
                        np.max(no_idk_mean + no_idk_std) if no_idk_count > 0 else 0) * 1.1 + 0.01)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / f'kwt_ridk_{dataname}_idk_prob_sequence_no_interp.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

    # Print statistics
    print(f"\n{dataname.upper()} Statistics:")
    print(f"  <IDK> Generated: {idk_count} samples")
    print(f"  No <IDK>: {no_idk_count} samples")

    return {
        'idk_count': idk_count,
        'no_idk_count': no_idk_count,
        'idk_mean': idk_mean,
        'no_idk_mean': no_idk_mean
    }

def main():
    num_bins = 3  # Early, Mid, Late
    min_length = num_bins  # Minimum sequence length = num_bins

    # Define datasets and their file paths
    # datasets = {
    #     'halueval': 'halueval/llama-3.2-3b/sample_weight_reverse_smooth_llm_idk0.16_prob.jsonl',
    #     'medqa': 'medqa/llama-3.2-3b/sample_weight_reverse_smooth_llm_idk0.16_prob.jsonl',
    #     'sciq': 'sciq/llama-3.2-3b/sample_weight_reverse_smooth_llm_idk0.16_prob.jsonl'
    # }
    # datasets = {
    #     'halueval': 'halueval/llama-3.2-3b/seal_prob.jsonl',
    #     'medqa': 'medqa/llama-3.2-3b/seal_prob.jsonl',
    #     'sciq': 'sciq/llama-3.2-3b/seal_prob.jsonl'
    # }  
    datasets = {
        'halueval': 'halueval/llama-3.2-3b/sample_weighted_reverse_ridk_llm_idk0.16_prob.jsonl',
        'medqa': 'medqa/llama-3.2-3b/sample_weighted_reverse_ridk_llm_idk0.16_prob.jsonl',
        'sciq': 'sciq/llama-3.2-3b/sample_weighted_reverse_ridk_llm_idk0.16_prob.jsonl'
    }       

    output_dir = 'idk_prob_visualization'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for dataname, filepath in datasets.items():
        print(f"\n{'='*50}")
        print(f"Processing {dataname}...")
        print('='*50)

        if not Path(filepath).exists():
            print(f"  File not found: {filepath}")
            continue

        # Load data
        data = load_results(filepath)
        print(f"  Loaded {len(data)} samples")

        # Analyze sequences (excluding last token/EOS, and short sequences)
        idk_generated, idk_not_generated, skipped = analyze_idk_probability_sequence(data, min_length)
        print(f"  Skipped {skipped} samples with length < {min_length}")

        # Plot
        plot_idk_probability_comparison(dataname, idk_generated, idk_not_generated, output_dir, num_bins)

if __name__ == "__main__":
    main()
