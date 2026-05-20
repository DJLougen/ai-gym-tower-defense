#!/usr/bin/env python3
"""
Benchmark results visualization and reporting.

Generates:
- Markdown leaderboard table
- Optional matplotlib charts (win rate, waves, cost, latency)
- JSON summary
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_results(path: str) -> Dict:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        return json.load(f)


def generate_markdown_table(results: Dict) -> str:
    """Generate a markdown leaderboard table from benchmark results."""
    models = results.get("models", {})
    if not models:
        return "No results to display."

    # Sort by win rate, then by waves survived
    sorted_models = sorted(
        models.items(),
        key=lambda x: (x[1]["summary"]["win_rate"], x[1]["summary"]["avg_waves_survived"]),
        reverse=True,
    )

    lines = [
        "| Rank | Model | Win Rate | Avg Waves | Avg Lives | Avg Gold | Avg Latency | Total Cost |",
        "|------|-------|----------|-----------|-----------|----------|-------------|------------|",
    ]

    for rank, (model_name, data) in enumerate(sorted_models, 1):
        s = data["summary"]
        win_rate = f"{s['win_rate']:.0%}"
        waves = f"{s['avg_waves_survived']:.1f}"
        lives = f"{s['avg_final_lives']:.1f}"
        gold = f"{s['avg_final_gold']:.0f}"
        latency = f"{s.get('avg_latency', 0):.2f}s"
        cost = f"${s.get('total_cost', 0):.4f}"
        lines.append(f"| {rank} | **{model_name}** | {win_rate} | {waves} | {lives} | {gold} | {latency} | {cost} |")

    return "\n".join(lines)


def generate_text_leaderboard(results: Dict, width: int = 80) -> str:
    """Generate a plain-text leaderboard (same format as benchmark.py output)."""
    models = results.get("models", {})
    if not models:
        return "No results to display."

    sorted_models = sorted(
        models.items(),
        key=lambda x: (x[1]["summary"]["win_rate"], x[1]["summary"]["avg_waves_survived"]),
        reverse=True,
    )

    sep = "=" * width
    lines = [
        sep,
        "BENCHMARK LEADERBOARD",
        sep,
        "",
        f"{'Model':<25} {'Win%':>6} {'Waves':>7} {'Lives':>7} {'Gold':>7} {'Latency':>9} {'Cost':>8}",
        "-" * width,
    ]

    for model_name, data in sorted_models:
        s = data["summary"]
        lines.append(
            f"{model_name:<25} "
            f"{s['win_rate']:>5.1%} "
            f"{s['avg_waves_survived']:>7.1f} "
            f"{s['avg_final_lives']:>7.1f} "
            f"{s['avg_final_gold']:>7.0f} "
            f"{s.get('avg_latency', 0):>8.2f}s "
            f"${s.get('total_cost', 0):>7.4f}"
        )

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def generate_charts(results: Dict, output_dir: str) -> List[str]:
    """Generate matplotlib charts from benchmark results.

    Returns list of generated file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not installed — skipping chart generation. Install with: pip install matplotlib")
        return []

    models = results.get("models", {})
    if not models:
        return []

    os.makedirs(output_dir, exist_ok=True)
    sorted_models = sorted(
        models.items(),
        key=lambda x: (x[1]["summary"]["win_rate"], x[1]["summary"]["avg_waves_survived"]),
        reverse=True,
    )

    names = [m[0] for m in sorted_models]
    summaries = [m[1]["summary"] for m in sorted_models]

    win_rates = [s["win_rate"] * 100 for s in summaries]
    waves = [s["avg_waves_survived"] for s in summaries]
    lives = [s["avg_final_lives"] for s in summaries]
    gold = [s["avg_final_gold"] for s in summaries]
    latencies = [s.get("avg_latency", 0) for s in summaries]
    costs = [s.get("total_cost", 0) for s in summaries]

    n = len(names)
    x = range(n)

    fig_paths = []

    # Chart 1: Win Rate + Waves Survived (dual axis)
    fig, ax1 = plt.subplots(figsize=(max(8, n * 1.5), 5))
    bar_width = 0.35

    bars1 = ax1.bar([i - bar_width / 2 for i in x], win_rates, bar_width, label="Win Rate (%)", color="#4CAF50", alpha=0.85)
    bars2 = ax1.bar([i + bar_width / 2 for i in x], waves, bar_width, label="Avg Waves Survived", color="#2196F3", alpha=0.85)

    ax1.set_xlabel("Model")
    ax1.set_ylabel("Win Rate (%) / Waves Survived")
    ax1.set_title("Tower Defense Benchmark: Win Rate & Waves Survived")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.legend()
    ax1.set_ylim(0, max(max(win_rates), max(waves)) * 1.15 + 1)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "benchmark_win_rate_waves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    fig_paths.append(path)

    # Chart 2: Cost + Latency
    fig, ax1 = plt.subplots(figsize=(max(8, n * 1.5), 5))

    color_cost = "#FF5722"
    color_lat = "#9C27B0"

    ax1.bar([i - bar_width / 2 for i in x], costs, bar_width, label="Total Cost ($)", color=color_cost, alpha=0.85)
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Total Cost ($)", color=color_cost)
    ax1.tick_params(axis="y", labelcolor=color_cost)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.bar([i + bar_width / 2 for i in x], latencies, bar_width, label="Avg Latency (s)", color=color_lat, alpha=0.85)
    ax2.set_ylabel("Avg Latency (s)", color=color_lat)
    ax2.tick_params(axis="y", labelcolor=color_lat)

    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
    plt.title("Tower Defense Benchmark: Cost & Latency")
    plt.tight_layout()
    path = os.path.join(output_dir, "benchmark_cost_latency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    fig_paths.append(path)

    # Chart 3: Per-episode detail (if multiple episodes)
    has_episodes = any(len(d.get("episodes", [])) > 1 for d in models.values())
    if has_episodes:
        fig, axes = plt.subplots(1, 2, figsize=(max(12, n * 2), 5))

        # Waves per episode
        ax = axes[0]
        for model_name, data in sorted_models:
            ep_waves = [ep["waves_survived"] for ep in data.get("episodes", [])]
            ax.plot(range(1, len(ep_waves) + 1), ep_waves, marker="o", label=model_name, markersize=4)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Waves Survived")
        ax.set_title("Waves Survived per Episode")
        ax.legend(fontsize=7)

        # Gold per episode
        ax = axes[1]
        for model_name, data in sorted_models:
            ep_gold = [ep["final_gold"] for ep in data.get("episodes", [])]
            ax.plot(range(1, len(ep_gold) + 1), ep_gold, marker="s", label=model_name, markersize=4)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Final Gold")
        ax.set_title("Final Gold per Episode")
        ax.legend(fontsize=7)

        plt.tight_layout()
        path = os.path.join(output_dir, "benchmark_episodes.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        fig_paths.append(path)

    return fig_paths


def generate_report(results: Dict, output_dir: str, charts: bool = True) -> str:
    """Generate a full markdown report.

    Returns path to generated markdown file.
    """
    os.makedirs(output_dir, exist_ok=True)

    config = results.get("config", {})
    timestamp = results.get("timestamp", "unknown")

    sections = [
        "# Tower Defense LLM Benchmark Report",
        "",
        f"**Run timestamp**: {timestamp}",
        f"**Episodes per model**: {config.get('episodes_per_model', '?')}",
        f"**Max steps per episode**: {config.get('max_steps', '?')}",
        "",
        "## Leaderboard",
        "",
        generate_markdown_table(results),
        "",
        "## Summary",
        "",
        generate_text_leaderboard(results),
        "",
    ]

    # Add chart references if generated
    if charts:
        chart_paths = generate_charts(results, output_dir)
        if chart_paths:
            sections.append("## Charts")
            sections.append("")
            for cp in chart_paths:
                rel = os.path.relpath(cp, output_dir)
                name = os.path.splitext(os.path.basename(cp))[0].replace("_", " ").title()
                sections.append(f"### {name}")
                sections.append("")
                sections.append(f"![{name}]({rel})")
                sections.append("")

    # Per-model details
    sections.append("## Per-Model Details")
    sections.append("")

    models = results.get("models", {})
    sorted_models = sorted(
        models.items(),
        key=lambda x: (x[1]["summary"]["win_rate"], x[1]["summary"]["avg_waves_survived"]),
        reverse=True,
    )

    for model_name, data in sorted_models:
        sections.append(f"### {model_name}")
        sections.append("")
        s = data["summary"]
        sections.append(f"- **Win rate**: {s['win_rate']:.0%}")
        sections.append(f"- **Avg waves**: {s['avg_waves_survived']:.1f}")
        sections.append(f"- **Avg final lives**: {s['avg_final_lives']:.1f}")
        sections.append(f"- **Avg final gold**: {s['avg_final_gold']:.0f}")
        sections.append(f"- **Avg latency**: {s.get('avg_latency', 0):.2f}s")
        sections.append(f"- **Total cost**: ${s.get('total_cost', 0):.4f}")
        sections.append(f"- **Total errors**: {s.get('total_errors', 0)}")
        sections.append("")

        episodes = data.get("episodes", [])
        if episodes:
            sections.append("| Episode | Waves | Won | Lives | Gold | Steps | Errors |")
            sections.append("|---------|-------|-----|-------|------|-------|--------|")
            for i, ep in enumerate(episodes, 1):
                won = "Yes" if ep.get("won") else "No"
                sections.append(
                    f"| {i} | {ep['waves_survived']} | {won} | "
                    f"{ep['final_lives']:.0f} | {ep['final_gold']} | "
                    f"{ep['steps']} | {len(ep.get('errors', []))} |"
                )
            sections.append("")

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(sections))

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to benchmark JSON results file (default: latest in benchmark_results/)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for charts and report (default: same as input)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "text", "charts", "all"],
        default="all",
        help="Output format (default: all)",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation (useful without matplotlib)",
    )

    args = parser.parse_args()

    # Find input file
    if args.input:
        input_path = args.input
    else:
        # Find latest benchmark result
        results_dir = "benchmark_results"
        if not os.path.isdir(results_dir):
            print(f"No {results_dir}/ directory found. Run benchmark.py first.")
            sys.exit(1)
        files = sorted(Path(results_dir).glob("benchmark_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            print(f"No benchmark JSON files found in {results_dir}/")
            sys.exit(1)
        input_path = str(files[0])
        print(f"Using latest results: {input_path}")

    results = load_results(input_path)

    # Determine output directory
    output_dir = args.output_dir or os.path.dirname(input_path) or "."

    fmt = args.format
    make_charts = fmt in ("charts", "all") and not args.no_charts

    if fmt in ("markdown", "all"):
        table = generate_markdown_table(results)
        print("\n## Markdown Table\n")
        print(table)

    if fmt in ("text", "all"):
        leaderboard = generate_text_leaderboard(results)
        print("\n## Text Leaderboard\n")
        print(leaderboard)

    if make_charts:
        chart_paths = generate_charts(results, output_dir)
        if chart_paths:
            print("\n## Charts Generated\n")
            for p in chart_paths:
                print(f"  {p}")

    if fmt in ("markdown", "all"):
        report_path = generate_report(results, output_dir, charts=make_charts)
        print(f"\n## Full Report\n  {report_path}")


if __name__ == "__main__":
    main()
