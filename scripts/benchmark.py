#!/usr/bin/env python3
"""
Benchmark harness for LLM agents on Tower Defense.

Runs multiple models across episodes and collects:
- Win rate
- Average waves survived
- Average gold efficiency
- Token usage and cost
- Latency statistics
- Action validity rate
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_gym_td.env import TowerDefenseEnv
from ai_gym_td.llm_agents import OpenAIAgent, AnthropicAgent, GoogleAgent, OllamaAgent
from ai_gym_td.agents import GreedyAgent, RandomAgent


# Model configurations
MODEL_CONFIGS = {
    # OpenAI models
    "gpt-4o": {"class": OpenAIAgent, "provider": "openai", "temperature": 0.7},
    "gpt-4o-mini": {"class": OpenAIAgent, "provider": "openai", "temperature": 0.7},
    "o1-mini": {"class": OpenAIAgent, "provider": "openai", "temperature": 1.0},
    "o1-preview": {"class": OpenAIAgent, "provider": "openai", "temperature": 1.0},
    
    # Anthropic models
    "claude-3-5-sonnet": {"class": AnthropicAgent, "provider": "anthropic", "temperature": 0.7},
    "claude-3-5-haiku": {"class": AnthropicAgent, "provider": "anthropic", "temperature": 0.7},
    "claude-3-opus": {"class": AnthropicAgent, "provider": "anthropic", "temperature": 0.7},
    
    # Google models
    "gemini-1.5-pro": {"class": GoogleAgent, "provider": "google", "temperature": 0.7},
    "gemini-1.5-flash": {"class": GoogleAgent, "provider": "google", "temperature": 0.7},
    "gemini-2.0-flash": {"class": GoogleAgent, "provider": "google", "temperature": 0.7},

    # Ollama local/cloud models (OpenAI-compatible API)
    "qwen3-coder-next:cloud": {"class": OllamaAgent, "provider": "ollama", "temperature": 0.7},
    "minimax-m2.7:cloud": {"class": OllamaAgent, "provider": "ollama", "temperature": 0.7},
    "glm-5.1:cloud": {"class": OllamaAgent, "provider": "ollama", "temperature": 0.7},
    "qwen3.5:397b-cloud": {"class": OllamaAgent, "provider": "ollama", "temperature": 0.7},
    "gemma4:31b-cloud": {"class": OllamaAgent, "provider": "ollama", "temperature": 0.7},
    "deepseek-v4-pro:cloud": {"class": OllamaAgent, "provider": "ollama", "temperature": 0.7},
    "deepseek-v4-flash:cloud": {"class": OllamaAgent, "provider": "ollama", "temperature": 0.7},
    "kimi-k2.6:cloud": {"class": OllamaAgent, "provider": "ollama", "temperature": 0.7},
    
    # Baselines (no API needed)
    "greedy": {"class": GreedyAgent, "provider": "baseline", "temperature": None},
    "random": {"class": RandomAgent, "provider": "baseline", "temperature": None},
}


def create_agent(model_name: str, env: TowerDefenseEnv, verbose: bool = False):
    """Create agent from model name."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    agent_class = config["class"]
    
    if config["provider"] == "baseline":
        return agent_class()
    else:
        # Check for API key
        api_key = None
        if config["provider"] == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
        elif config["provider"] == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
        elif config["provider"] == "google":
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")

        kwargs = dict(
            model=model_name,
            env=env,
            api_key=api_key,
            temperature=config["temperature"],
            verbose=verbose,
        )
        # Ollama models get a per-call timeout to prevent benchmark hangs
        if config["provider"] == "ollama":
            kwargs["timeout"] = 120.0
        return agent_class(**kwargs)


def compute_strategic_metrics(env: TowerDefenseEnv, episode_data: Dict) -> Dict:
    """Compute strategic metrics from game state after episode."""
    game = env.game
    metrics = {}

    # Get tower stats
    towers = list(game.towers.values())
    num_towers = len(towers)

    if num_towers == 0:
        # No towers built — return minimal metrics
        return {
            "tower_efficiency": 0.0,
            "coverage_score": 0.0,
            "build_timing": float("inf"),
            "tower_diversity": 0.0,
            "gold_utilization": 0.0,
            "path_control": 0.0,
            "num_towers_built": 0,
        }

    # 1. Tower efficiency: total damage / gold spent
    total_damage = sum(t.total_damage for t in towers)
    gold_spent = sum(t.spec.cost for t in towers)
    metrics["tower_efficiency"] = total_damage / max(1, gold_spent)
    metrics["num_towers_built"] = num_towers

    # 2. Coverage score: % of path cells within tower range
    path_cells = game.path
    covered = 0
    for px, py in path_cells:
        for t in towers:
            dist = ((t.x - px) ** 2 + (t.y - py) ** 2) ** 0.5
            if dist <= t.spec.range:
                covered += 1
                break
    metrics["coverage_score"] = covered / max(1, len(path_cells))

    # 3. Build timing: average step when towers were placed
    # We track this in episode_data["tower_build_steps"]
    build_steps = episode_data.get("tower_build_steps", [])
    if build_steps:
        metrics["build_timing"] = np.mean(build_steps)
    else:
        metrics["build_timing"] = float("inf")

    # 4. Tower diversity: unique tower types / total available
    unique_types = len(set(t.spec.name for t in towers))
    total_types = len(game.cfg.towers)
    metrics["tower_diversity"] = unique_types / max(1, total_types)

    # 5. Gold utilization: gold spent / (starting gold + gold earned)
    starting_gold = game.cfg.starting_gold
    # Estimate gold earned from kills and waves (rough approximation)
    final_gold = episode_data.get("final_gold", 0)
    gold_earned = final_gold + gold_spent - starting_gold
    total_gold_available = starting_gold + max(0, gold_earned)
    metrics["gold_utilization"] = gold_spent / max(1, total_gold_available)

    # 6. Path control: average number of towers covering each path cell
    total_coverage = 0
    for px, py in path_cells:
        for t in towers:
            dist = ((t.x - px) ** 2 + (t.y - py) ** 2) ** 0.5
            if dist <= t.spec.range:
                total_coverage += 1
    metrics["path_control"] = total_coverage / max(1, len(path_cells))

    return metrics



def run_episode(agent, env: TowerDefenseEnv, max_steps: int = 4000, render: bool = False) -> Dict:
    """Run a single episode and collect metrics."""
    obs, info = env.reset()

    episode_data = {
        "steps": 0,
        "waves_survived": 0,
        "won": False,
        "final_gold": 0,
        "final_lives": 0,
        "total_reward": 0.0,
        "valid_actions": 0,
        "invalid_actions": 0,
        "latencies": [],
        "errors": [],
        "tower_build_steps": [],  # Track when towers are built
    }

    if hasattr(agent, "reset"):
        agent.reset()


    # Track tower count to detect builds
    prev_tower_count = env.game.tower_count()

    for step in range(max_steps):
        # Get action
        t0 = time.time()
        try:
            action = agent.act(obs, info)
            latency = time.time() - t0
            episode_data["latencies"].append(latency)
        except Exception as e:
            error_msg = f"Step {step}: {type(e).__name__}: {str(e)}"
            episode_data["errors"].append(error_msg)
            print(f"  Error: {error_msg}")
            # Use pass action on error
            action = np.array([0, 0, 0], dtype=np.int64)

        # Check if action is valid (within bounds)
        if action.shape == (3,):
            episode_data["valid_actions"] += 1
        else:
            episode_data["invalid_actions"] += 1
            action = np.array([0, 0, 0], dtype=np.int64)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_data["total_reward"] += reward
        episode_data["steps"] += 1

        # Track tower builds
        curr_tower_count = env.game.tower_count()
        if curr_tower_count > prev_tower_count:
            episode_data["tower_build_steps"].append(step)
        prev_tower_count = curr_tower_count

        if render:
            env.render()

        if terminated or truncated:
            break

    # Collect final metrics
    episode_data["waves_survived"] = info.get("wave", 0)
    episode_data["won"] = info.get("phase") == "won"
    episode_data["final_gold"] = info.get("gold", 0)
    episode_data["final_lives"] = info.get("lives", 0)

    # Compute strategic metrics
    episode_data["strategic"] = compute_strategic_metrics(env, episode_data)

    # Get agent stats if available
    if hasattr(agent, "get_stats"):
        episode_data["agent_stats"] = agent.get_stats()

    return episode_data


def run_benchmark(
    models: List[str],
    episodes_per_model: int = 3,
    max_steps: int = 4000,
    render: bool = False,
    verbose: bool = False,
    maps: Optional[List[str]] = None,
) -> Dict:
    """Run benchmark across multiple models and maps.

    Args:
        models: List of model names to test
        episodes_per_model: Number of episodes per model per map
        max_steps: Maximum steps per episode
        render: Whether to render visually
        verbose: Print detailed agent output
        maps: List of map names (e.g., ["s_curve", "straight", "zigzag"]).
              If None, uses only s_curve.
    """
    if maps is None:
        maps = ["s_curve"]

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "episodes_per_model": episodes_per_model,
            "max_steps": max_steps,
            "maps": maps,
        },
        "models": {},
    }

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name}")
        print(f"{'='*60}")

        # Create a single env for all maps (will reset with different configs)
        from ai_gym_td.config import GameConfig
        cfg = GameConfig.default_20x12()  # Default config
        env = TowerDefenseEnv(config=cfg)
        agent = create_agent(model_name, env, verbose=verbose)

        model_results = {
            "episodes": [],
            "summary": {},
        }
        
        for map_name in maps:
            print(f"\n--- Map: {map_name} ---")

            # Create config for this map
            if map_name == "s_curve":
                cfg = GameConfig.default_20x12()
            elif hasattr(GameConfig, f"{map_name}_path"):
                cfg = getattr(GameConfig, f"{map_name}_path")()
            else:
                print(f"Warning: Unknown map '{map_name}', using s_curve")
                cfg = GameConfig.default_20x12()

            # Recreate env with new config
            env.close()
            env = TowerDefenseEnv(config=cfg)

            for ep in range(episodes_per_model):
                print(f"\nEpisode {ep + 1}/{episodes_per_model}")

                episode_data = run_episode(agent, env, max_steps=max_steps, render=render)
                episode_data["map"] = map_name  # Track which map this was
                model_results["episodes"].append(episode_data)

                print(f"  Waves: {episode_data['waves_survived']}/20")
                print(f"  Won: {episode_data['won']}")
                print(f"  Final lives: {episode_data['final_lives']}")
                print(f"  Final gold: {episode_data['final_gold']}")
                print(f"  Steps: {episode_data['steps']}")

                if episode_data["latencies"]:
                    print(f"  Avg latency: {np.mean(episode_data['latencies']):.2f}s")

                if episode_data["errors"]:
                    print(f"  Errors: {len(episode_data['errors'])}")
        # Compute summary statistics
        episodes = model_results["episodes"]
        model_results["summary"] = {
            "win_rate": np.mean([ep["won"] for ep in episodes]),
            "avg_waves_survived": np.mean([ep["waves_survived"] for ep in episodes]),
            "avg_final_lives": np.mean([ep["final_lives"] for ep in episodes]),
            "avg_final_gold": np.mean([ep["final_gold"] for ep in episodes]),
            "avg_steps": np.mean([ep["steps"] for ep in episodes]),
            "total_errors": sum(len(ep["errors"]) for ep in episodes),
        }

        # Aggregate strategic metrics
        strategic_keys = ["tower_efficiency", "coverage_score", "build_timing",
                          "tower_diversity", "gold_utilization", "path_control", "num_towers_built"]
        for key in strategic_keys:
            values = [ep["strategic"].get(key, 0) for ep in episodes]
            # Filter out inf values for build_timing
            finite_values = [v for v in values if not (isinstance(v, float) and v == float("inf"))]
            if finite_values:
                model_results["summary"][f"avg_{key}"] = np.mean(finite_values)
            else:
                model_results["summary"][f"avg_{key}"] = 0.0
        
        # Add agent stats if available
        if hasattr(agent, "get_stats"):
            agent_stats = agent.get_stats()
            model_results["summary"]["agent_stats"] = agent_stats
            
            if "avg_latency" in agent_stats:
                model_results["summary"]["avg_latency"] = agent_stats["avg_latency"]
            if "total_cost" in agent_stats:
                model_results["summary"]["total_cost"] = agent_stats["total_cost"]
            if "total_tokens" in agent_stats:
                model_results["summary"]["total_tokens"] = agent_stats["total_tokens"]
        
        results["models"][model_name] = model_results
        env.close()
    
    return results


def print_leaderboard(results: Dict):
    """Print formatted leaderboard."""
    print(f"\n{'='*80}")
    print("BENCHMARK LEADERBOARD")
    print(f"{'='*80}\n")
    
    # Sort by win rate, then by waves survived
    sorted_models = sorted(
        results["models"].items(),
        key=lambda x: (x[1]["summary"]["win_rate"], x[1]["summary"]["avg_waves_survived"]),
        reverse=True,
    )
    
    # Header
    print(f"{'Model':<20} {'Win%':>6} {'Waves':>7} {'Lives':>7} {'Gold':>7} {'Latency':>9} {'Cost':>8}")
    print("-" * 80)
    
    for model_name, data in sorted_models:
        summary = data["summary"]
        win_rate = summary["win_rate"]
        waves = summary["avg_waves_survived"]
        lives = summary["avg_final_lives"]
        gold = summary["avg_final_gold"]
        latency = summary.get("avg_latency", 0)
        cost = summary.get("total_cost", 0)
        
        print(f"{model_name:<20} {win_rate:>5.1%} {waves:>7.1f} {lives:>7.1f} {gold:>7.0f} {latency:>8.2f}s ${cost:>7.4f}")
    
    print(f"\n{'='*80}\n")

    # Strategic metrics leaderboard
    print(f"\n{'='*100}")
    print("STRATEGIC PERFORMANCE METRICS")
    print(f"{'='*100}\n")

    print(f"{'Model':<20} {'Efficiency':>11} {'Coverage':>10} {'Timing':>8} {'Diversity':>10} {'Util':>6} {'Control':>9} {'Towers':>7}")
    print("-" * 100)

    for model_name, data in sorted_models:
        summary = data["summary"]
        efficiency = summary.get("avg_tower_efficiency", 0)
        coverage = summary.get("avg_coverage_score", 0)
        timing = summary.get("avg_build_timing", 0)
        diversity = summary.get("avg_tower_diversity", 0)
        util = summary.get("avg_gold_utilization", 0)
        control = summary.get("avg_path_control", 0)
        towers = summary.get("avg_num_towers_built", 0)

        print(f"{model_name:<20} {efficiency:>11.2f} {coverage:>9.1%} {timing:>8.1f} {diversity:>9.1%} {util:>5.1%} {control:>9.2f} {towers:>7.1f}")

    print(f"\n{'='*100}")
    print("Legend: Efficiency=dmg/gold, Coverage=%path covered, Timing=avg build step, Diversity=unique types, Util=gold spent, Control=avg towers/path cell")
    print(f"{'='*100}\n")


def save_results(results: Dict, output_dir: str = "benchmark_results"):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM agents on Tower Defense")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["greedy", "random"],
        help="Models to test (default: greedy random)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes per model (default: 3)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4000,
        help="Maximum steps per episode (default: 4000)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render game visually",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed agent output",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory to save results (default: benchmark_results)",
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        default=["s_curve"],
        help="Maps to test (default: s_curve). Available: s_curve, straight, zigzag, chokepoint, spiral",
    )
    
    args = parser.parse_args()
    
    print("Tower Defense LLM Benchmark")
    print(f"Models: {', '.join(args.models)}")
    print(f"Maps: {', '.join(args.maps)}")
    print(f"Episodes per model per map: {args.episodes}")
    print(f"Max steps: {args.max_steps}")
    print()

    # Run benchmark
    results = run_benchmark(
        models=args.models,
        episodes_per_model=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        verbose=args.verbose,
        maps=args.maps,
    )
    
    # Print leaderboard
    print_leaderboard(results)
    
    # Save results
    save_results(results, args.output_dir)
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
