#!/usr/bin/env python3
"""Generate visualization plots from benchmark results."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use('Agg')  # Non-interactive backend

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11

# Benchmark data from actual runs
MODELS = ['greedy', 'glm-5.1:cloud', 'kimi-k2.6:cloud']
MODEL_LABELS = ['Greedy\n(baseline)', 'GLM-5.1\n(Cloud)', 'Kimi-K2.6\n(Cloud)']

# Performance metrics
COVERAGE = [9.1, 32.7, 40.0]
LATENCY = [0.00, 9.38, 103.48]
FINAL_GOLD = [5, 10, 70]

# Strategic metrics
EFFICIENCY = [0.00, 0.00, 0.00]
TIMING = [0.5, 1.5, 2.0]
DIVERSITY = [50.0, 50.0, 25.0]
UTILIZATION = [95.8, 91.7, 41.7]
CONTROL = [0.15, 0.35, 0.40]
TOWERS = [2.0, 4.0, 2.0]

OUTPUT_DIR = Path(__file__).parent.parent / 'media'
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_coverage_comparison():
    """Bar chart showing coverage scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#95a5a6', '#3498db', '#e74c3c']
    bars = ax.bar(MODEL_LABELS, COVERAGE, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, COVERAGE):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Coverage Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Path Coverage by Model', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 50)
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotation
    ax.annotate('3.6× better than baseline', 
                xy=(1, 32.7), xytext=(1, 38),
                ha='center', fontsize=10, style='italic',
                arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'coverage_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_latency_comparison():
    """Bar chart showing latency."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#95a5a6', '#3498db', '#e74c3c']
    bars = ax.bar(MODEL_LABELS, LATENCY, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, LATENCY):
        label = f'{val:.2f}s' if val < 1 else f'{val:.1f}s'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                label, ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Average Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Response Latency by Model', fontsize=14, fontweight='bold', pad=20)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'latency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_strategic_metrics():
    """Grouped bar chart for strategic metrics."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metrics = ['Coverage\n(%)', 'Diversity\n(%)', 'Utilization\n(%)', 'Path Control\n(×100)']
    greedy_vals = [9.1, 50.0, 95.8, 15]
    glm_vals = [32.7, 50.0, 91.7, 35]
    kimi_vals = [40.0, 25.0, 41.7, 40]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    bars1 = ax.bar(x - width, greedy_vals, width, label='Greedy', color='#95a5a6', edgecolor='black')
    bars2 = ax.bar(x, glm_vals, width, label='GLM-5.1', color='#3498db', edgecolor='black')
    bars3 = ax.bar(x + width, kimi_vals, width, label='Kimi-K2.6', color='#e74c3c', edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Strategic Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'strategic_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_table():
    """Create a table image with all metrics."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Table data
    col_labels = ['Model', 'Coverage', 'Latency', 'Towers', 'Diversity', 'Utilization', 'Cost']
    table_data = [
        ['Greedy (baseline)', '9.1%', '0.00s', '2.0', '50.0%', '95.8%', '$0.00'],
        ['GLM-5.1:cloud', '32.7%', '9.38s', '4.0', '50.0%', '91.7%', '$0.00'],
        ['Kimi-K2.6:cloud', '40.0%', '103.48s', '2.0', '25.0%', '41.7%', '$0.00'],
    ]
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.15, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(color='white', fontweight='bold')
    
    # Style rows
    for i in range(1, 4):
        for j in range(len(col_labels)):
            cell = table[(i, j)]
            if i == 1:
                cell.set_facecolor('#ecf0f1')
            elif i == 2:
                cell.set_facecolor('#d5e8f0')
            else:
                cell.set_facecolor('#fadbd8')
    
    plt.title('Benchmark Results Summary (5 steps, 1 episode)', 
              fontsize=13, fontweight='bold', pad=30)
    
    plt.savefig(OUTPUT_DIR / 'benchmark_summary.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Generate all plots."""
    print("Generating benchmark visualizations...")
    plot_coverage_comparison()
    print("- Coverage comparison plot")

    plot_latency_comparison()
    print("- Latency comparison plot")

    plot_strategic_metrics()
    print("- Strategic metrics plot")

    plot_summary_table()
    print("- Summary table plot")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
