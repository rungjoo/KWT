#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('ggplot')


def load_comparison_summary(file_path: str):
    """Load the comparison summary JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_8_cases_heatmap(summary_data):
    """Create a heatmap showing the 8 cases distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('IDK Token Analysis: 8 Cases Heatmap', fontsize=16, fontweight='bold')

    for idx, comp in enumerate(summary_data['comparisons']):
        if idx >= 2:
            break

        comp_name = comp['name'].replace('base_model_vs_', '')
        idk_stats = comp['idk_detailed_statistics']

        # Create 2x4 matrix for heatmap
        data = np.array([
            [idk_stats['base_O_comp_O_no_idk'], idk_stats['base_O_comp_O_with_idk'],
             idk_stats['base_O_comp_X_no_idk'], idk_stats['base_O_comp_X_with_idk']],
            [idk_stats['base_X_comp_O_no_idk'], idk_stats['base_X_comp_O_with_idk'],
             idk_stats['base_X_comp_X_no_idk'], idk_stats['base_X_comp_X_with_idk']]
        ])

        # Create percentage data
        data_pct = np.array([
            [idk_stats['base_O_comp_O_no_idk_pct'], idk_stats['base_O_comp_O_with_idk_pct'],
             idk_stats['base_O_comp_X_no_idk_pct'], idk_stats['base_O_comp_X_with_idk_pct']],
            [idk_stats['base_X_comp_O_no_idk_pct'], idk_stats['base_X_comp_O_with_idk_pct'],
             idk_stats['base_X_comp_X_no_idk_pct'], idk_stats['base_X_comp_X_with_idk_pct']]
        ])

        ax = axes[idx]
        im = ax.imshow(data_pct, cmap='YlOrRd', aspect='auto', vmin=0, vmax=60)

        # Set ticks and labels
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels(['Comp O\n(no IDK)', 'Comp O\n(with IDK)',
                           'Comp X\n(no IDK)', 'Comp X\n(with IDK)'])
        ax.set_yticklabels(['Base O', 'Base X'])

        # Add text annotations
        for i in range(2):
            for j in range(4):
                text = ax.text(j, i, f'{data[i, j]}\n({data_pct[i, j]:.1f}%)',
                             ha="center", va="center", color="black", fontsize=10)

        ax.set_title(f'Model: {comp_name}', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Percentage (%)')

    plt.tight_layout()
    return fig


def create_idk_impact_comparison(summary_data):
    """Create bar chart comparing IDK impact across models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('IDK Token Impact Analysis Across Different Scenarios', fontsize=16, fontweight='bold')

    models = []
    data_dict = {
        'total_idk': [],
        'idk_when_both_correct': [],
        'idk_when_both_wrong': [],
        'idk_helps_correct': [],
        'idk_causes_wrong': []
    }

    for comp in summary_data['comparisons']:
        model_name = comp['name'].replace('base_model_vs_', '')
        models.append(model_name)
        idk_stats = comp['idk_detailed_statistics']

        # Total IDK usage
        total_idk = (idk_stats['base_O_comp_O_with_idk'] +
                    idk_stats['base_O_comp_X_with_idk'] +
                    idk_stats['base_X_comp_O_with_idk'] +
                    idk_stats['base_X_comp_X_with_idk'])
        data_dict['total_idk'].append(total_idk)

        # IDK when both correct (base O, comp O with IDK)
        data_dict['idk_when_both_correct'].append(idk_stats['base_O_comp_O_with_idk'])

        # IDK when both wrong (base X, comp X with IDK)
        data_dict['idk_when_both_wrong'].append(idk_stats['base_X_comp_X_with_idk'])

        # IDK helps get correct (base X, comp O with IDK)
        data_dict['idk_helps_correct'].append(idk_stats['base_X_comp_O_with_idk'])

        # IDK causes wrong (base O, comp X with IDK)
        data_dict['idk_causes_wrong'].append(idk_stats['base_O_comp_X_with_idk'])

    # Plot 1: Total IDK Usage
    ax = axes[0, 0]
    bars = ax.bar(models, data_dict['total_idk'], color='steelblue')
    ax.set_title('Total IDK Token Usage', fontweight='bold')
    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Model')
    for bar, val in zip(bars, data_dict['total_idk']):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{val}\n({val/20:.1f}%)', ha='center', va='bottom')

    # Plot 2: IDK Usage by Correctness
    ax = axes[0, 1]
    x = np.arange(len(models))
    width = 0.35
    bars1 = ax.bar(x - width/2, data_dict['idk_when_both_correct'], width,
                   label='Both Correct with IDK', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, data_dict['idk_when_both_wrong'], width,
                   label='Both Wrong with IDK', color='red', alpha=0.7)
    ax.set_title('IDK Usage: Correct vs Wrong Cases', fontweight='bold')
    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # Plot 3: IDK Impact on Performance
    ax = axes[1, 0]
    bars1 = ax.bar(x - width/2, data_dict['idk_helps_correct'], width,
                   label='IDK Helps (Base X → Comp O)', color='darkgreen', alpha=0.7)
    bars2 = ax.bar(x + width/2, data_dict['idk_causes_wrong'], width,
                   label='IDK Hurts (Base O → Comp X)', color='darkred', alpha=0.7)
    ax.set_title('IDK Impact: Helps vs Hurts Performance', fontweight='bold')
    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # Plot 4: IDK Efficiency Ratio
    ax = axes[1, 1]
    efficiency_ratio = []
    for i in range(len(models)):
        total = data_dict['total_idk'][i]
        if total > 0:
            helps = data_dict['idk_helps_correct'][i]
            hurts = data_dict['idk_causes_wrong'][i]
            # Efficiency = (helps - hurts) / total_idk
            ratio = ((helps - hurts) / total) * 100
            efficiency_ratio.append(ratio)
        else:
            efficiency_ratio.append(0)

    bars = ax.bar(models, efficiency_ratio, color=['gray' if r == 0 else 'green' if r > 0 else 'red'
                                                   for r in efficiency_ratio])
    ax.set_title('IDK Efficiency Ratio\n(Helps - Hurts) / Total IDK × 100', fontweight='bold')
    ax.set_ylabel('Efficiency (%)')
    ax.set_xlabel('Model')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    for bar, val in zip(bars, efficiency_ratio):
        if val != 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.5 if val > 0 else bar.get_height() - 1,
                   f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top')

    plt.tight_layout()
    return fig


def create_performance_transition_chart(summary_data):
    """Create Sankey-style transition chart showing performance changes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Performance Transition Analysis: Base Model → Comparison Models',
                fontsize=16, fontweight='bold')

    for idx, comp in enumerate(summary_data['comparisons']):
        if idx >= 2:
            break

        ax = axes[idx]
        model_name = comp['name'].replace('base_model_vs_', '')
        stats = comp['statistics']
        idk_stats = comp['idk_detailed_statistics']

        # Categories
        categories = ['Both\nCorrect', 'Base O\nComp X', 'Base X\nComp O', 'Both\nWrong']
        values = [
            stats['both_correct'],
            stats['base_correct_other_incorrect'],
            stats['base_incorrect_other_correct'],
            stats['both_incorrect']
        ]

        # IDK breakdown for each category
        idk_values = [
            [idk_stats['base_O_comp_O_no_idk'], idk_stats['base_O_comp_O_with_idk']],
            [idk_stats['base_O_comp_X_no_idk'], idk_stats['base_O_comp_X_with_idk']],
            [idk_stats['base_X_comp_O_no_idk'], idk_stats['base_X_comp_O_with_idk']],
            [idk_stats['base_X_comp_X_no_idk'], idk_stats['base_X_comp_X_with_idk']]
        ]

        x = np.arange(len(categories))
        width = 0.6

        # Create stacked bars
        bars1 = ax.bar(x, [v[0] for v in idk_values], width,
                      label='No IDK', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x, [v[1] for v in idk_values], width,
                      bottom=[v[0] for v in idk_values],
                      label='With IDK', color='orange', alpha=0.8)

        ax.set_title(f'Model: {model_name}', fontweight='bold')
        ax.set_ylabel('Number of Samples')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        # Add value labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            # Label for no IDK
            if idk_values[i][0] > 0:
                ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height()/2,
                       f'{idk_values[i][0]}', ha='center', va='center', fontsize=10, fontweight='bold')
            # Label for with IDK
            if idk_values[i][1] > 0:
                ax.text(bar2.get_x() + bar2.get_width()/2,
                       bar1.get_height() + bar2.get_height()/2,
                       f'{idk_values[i][1]}', ha='center', va='center', fontsize=10, fontweight='bold')
            # Total on top
            total = values[i]
            ax.text(bar2.get_x() + bar2.get_width()/2,
                   bar1.get_height() + bar2.get_height() + 10,
                   f'{total}\n({total/20:.1f}%)', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def create_idk_distribution_pie(summary_data):
    """Create pie charts showing IDK distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('IDK Token Distribution Analysis', fontsize=16, fontweight='bold')

    for idx, comp in enumerate(summary_data['comparisons']):
        if idx >= 2:
            break

        ax = axes[idx]
        model_name = comp['name'].replace('base_model_vs_', '')
        idk_stats = comp['idk_detailed_statistics']

        # Calculate IDK vs No IDK
        total_no_idk = (idk_stats['base_O_comp_O_no_idk'] +
                       idk_stats['base_O_comp_X_no_idk'] +
                       idk_stats['base_X_comp_O_no_idk'] +
                       idk_stats['base_X_comp_X_no_idk'])

        total_with_idk = (idk_stats['base_O_comp_O_with_idk'] +
                         idk_stats['base_O_comp_X_with_idk'] +
                         idk_stats['base_X_comp_O_with_idk'] +
                         idk_stats['base_X_comp_X_with_idk'])

        if total_with_idk > 0:
            # Model with IDK tokens - show detailed breakdown
            labels = []
            sizes = []
            colors = []

            if idk_stats['base_O_comp_O_with_idk'] > 0:
                labels.append(f"O→O IDK\n({idk_stats['base_O_comp_O_with_idk']})")
                sizes.append(idk_stats['base_O_comp_O_with_idk'])
                colors.append('#2E7D32')  # Green

            if idk_stats['base_O_comp_X_with_idk'] > 0:
                labels.append(f"O→X IDK\n({idk_stats['base_O_comp_X_with_idk']})")
                sizes.append(idk_stats['base_O_comp_X_with_idk'])
                colors.append('#FFA726')  # Orange

            if idk_stats['base_X_comp_O_with_idk'] > 0:
                labels.append(f"X→O IDK\n({idk_stats['base_X_comp_O_with_idk']})")
                sizes.append(idk_stats['base_X_comp_O_with_idk'])
                colors.append('#5C6BC0')  # Blue

            if idk_stats['base_X_comp_X_with_idk'] > 0:
                labels.append(f"X→X IDK\n({idk_stats['base_X_comp_X_with_idk']})")
                sizes.append(idk_stats['base_X_comp_X_with_idk'])
                colors.append('#EF5350')  # Red

            labels.append(f"No IDK\n({total_no_idk})")
            sizes.append(total_no_idk)
            colors.append('#BDBDBD')  # Gray

            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.1f%%', startangle=90)

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
        else:
            # Model without IDK tokens
            labels = ['No IDK Usage']
            sizes = [total_no_idk]
            colors = ['#BDBDBD']

            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.0f%%', startangle=90)

        ax.set_title(f'Model: {model_name}\nTotal: {total_no_idk + total_with_idk} samples',
                    fontweight='bold')

    plt.tight_layout()
    return fig


def create_accuracy_comparison_matrix(summary_data):
    """Create matrix comparing accuracy changes with IDK usage."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Accuracy Impact Matrix: IDK Token Usage Analysis', fontsize=16, fontweight='bold')

    models = []
    metrics = []

    for comp in summary_data['comparisons']:
        model_name = comp['name'].replace('base_model_vs_', '')
        models.append(model_name)
        idk_stats = comp['idk_detailed_statistics']

        # Calculate key metrics
        total = comp['statistics']['total_compared']

        # Accuracy without IDK
        correct_no_idk = idk_stats['base_O_comp_O_no_idk'] + idk_stats['base_X_comp_O_no_idk']
        total_no_idk = (idk_stats['base_O_comp_O_no_idk'] + idk_stats['base_O_comp_X_no_idk'] +
                       idk_stats['base_X_comp_O_no_idk'] + idk_stats['base_X_comp_X_no_idk'])

        # Accuracy with IDK
        correct_with_idk = idk_stats['base_O_comp_O_with_idk'] + idk_stats['base_X_comp_O_with_idk']
        total_with_idk = (idk_stats['base_O_comp_O_with_idk'] + idk_stats['base_O_comp_X_with_idk'] +
                         idk_stats['base_X_comp_O_with_idk'] + idk_stats['base_X_comp_X_with_idk'])

        acc_no_idk = (correct_no_idk / total_no_idk * 100) if total_no_idk > 0 else 0
        acc_with_idk = (correct_with_idk / total_with_idk * 100) if total_with_idk > 0 else 0
        idk_usage_rate = (total_with_idk / total * 100) if total > 0 else 0

        metrics.append({
            'No IDK Accuracy': acc_no_idk,
            'With IDK Accuracy': acc_with_idk,
            'IDK Usage Rate': idk_usage_rate,
            'Accuracy Diff': acc_with_idk - acc_no_idk
        })

    # Create bar chart with multiple metrics
    metric_names = list(metrics[0].keys())
    x = np.arange(len(models))
    width = 0.2

    for i, metric_name in enumerate(metric_names):
        values = [m[metric_name] for m in metrics]
        offset = (i - len(metric_names)/2 + 0.5) * width

        if metric_name == 'Accuracy Diff':
            colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in values]
            bars = ax.bar(x + offset, values, width, label=metric_name)
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        else:
            bars = ax.bar(x + offset, values, width, label=metric_name, alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, values):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.5 if val > 0 else val - 1.5,
                       f'{val:.1f}', ha='center', va='bottom' if val > 0 else 'top',
                       fontsize=9)

    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='best')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    # python3 04_visualize_idk_comparison.py --summary medqa/llama_3.2-3b/comparison_results/all_comparisons_summary.json
    parser = argparse.ArgumentParser(description='Visualize IDK token comparison analysis')
    parser.add_argument('--summary', type=str, required=True,
                       help='Path to all_comparisons_summary.json file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save visualization plots')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.summary)

    # Load data
    print(f"Loading comparison summary from: {args.summary}")
    summary_data = load_comparison_summary(args.summary)

    # Create all visualizations
    print("\nGenerating visualizations...")

    # 1. Heatmap of 8 cases
    print("1. Creating 8 cases heatmap...")
    fig1 = create_8_cases_heatmap(summary_data)
    output_path1 = os.path.join(args.output_dir, '01_idk_8cases_heatmap.png')
    fig1.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_path1}")

    # 2. IDK impact comparison
    print("2. Creating IDK impact comparison...")
    fig2 = create_idk_impact_comparison(summary_data)
    output_path2 = os.path.join(args.output_dir, '02_idk_impact_analysis.png')
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_path2}")

    # 3. Performance transition chart
    print("3. Creating performance transition chart...")
    fig3 = create_performance_transition_chart(summary_data)
    output_path3 = os.path.join(args.output_dir, '03_performance_transition.png')
    fig3.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_path3}")

    # 4. IDK distribution pie charts
    print("4. Creating IDK distribution analysis...")
    fig4 = create_idk_distribution_pie(summary_data)
    output_path4 = os.path.join(args.output_dir, '04_idk_distribution_pie.png')
    fig4.savefig(output_path4, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_path4}")

    # 5. Accuracy comparison matrix
    print("5. Creating accuracy comparison matrix...")
    fig5 = create_accuracy_comparison_matrix(summary_data)
    output_path5 = os.path.join(args.output_dir, '05_accuracy_comparison_matrix.png')
    fig5.savefig(output_path5, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_path5}")

    # Show all plots
    plt.show()

    print("\n✅ All visualizations completed successfully!")
    print(f"📁 Saved to: {args.output_dir}")


if __name__ == '__main__':
    main()