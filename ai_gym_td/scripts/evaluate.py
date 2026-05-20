"""CLI: evaluate agents and render GIFs.

Examples
--------
# Run the greedy baseline and dump stats
python -m ai_gym_td.scripts.evaluate --agent greedy --episodes 5

# Render a GIF of the rule-based agent playing 1 episode
python -m ai_gym_td.scripts.evaluate --agent rule --episodes 1 --gif out.gif --render-steps 2

# Evaluate a trained PPO checkpoint
python -m ai_gym_td.scripts.evaluate --agent ppo --ckpt agent.pt --episodes 10
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np

from ai_gym_td.agents import GreedyAgent, RandomAgent, RuleBasedAgent
from ai_gym_td.env import TowerDefenseEnv
from ai_gym_td.render import PILRenderer, save_frames_as_gif


def _build_agent(name: str, ckpt: Optional[str]):
    if name == "random":
        return RandomAgent()
    if name == "greedy":
        return GreedyAgent()
    if name == "rule":
        return RuleBasedAgent()
    if name == "ppo":
        if not ckpt:
            raise ValueError("--ckpt is required for --agent ppo")
        import torch
        from ai_gym_td.ppo import TowerDefensePPOAgent
        env = TowerDefenseEnv()
        env.reset()
        agent = TowerDefensePPOAgent(env)
        agent.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        agent.eval()
        return _PPOAgentWrapper(agent, env)
    raise ValueError(f"Unknown agent: {name}")


class _PPOAgentWrapper:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def act(self, obs, info):
        import torch
        from ai_gym_td.ppo import batch_obs
        with torch.no_grad():
            batched, masks = batch_obs([obs])
            act, _, _, _ = self.agent.get_action_and_value(batched, action_mask=masks)
        return act[0].cpu().numpy()


def evaluate(
    agent_name: str,
    episodes: int,
    max_steps: int,
    render_every: int,
    gif_path: Optional[str],
    ckpt: Optional[str],
    seed: int,
) -> dict:
    env = TowerDefenseEnv(max_env_steps=max_steps)
    agent = _build_agent(agent_name, ckpt)
    renderer = PILRenderer(env.cfg) if (gif_path or render_every > 0) else None

    results = []
    frames: List[np.ndarray] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        if hasattr(agent, "reset"):
            agent.reset()
        total_reward = 0.0
        steps = 0
        while True:
            if renderer and (gif_path or (render_every and ep % render_every == 0)):
                frames.append(renderer.render_frame(env.game))
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        results.append({
            "episode": ep,
            "steps": steps,
            "reward": total_reward,
            "gold": env.game.gold,
            "lives": env.game.lives,
            "wave": env.game.wave_number,
            "phase": env.game.phase,
            "towers": env.game.tower_count(),
        })
        print(
            f"ep {ep}: steps={steps} reward={total_reward:.2f} "
            f"wave={env.game.wave_number}/{env.cfg.max_waves} "
            f"lives={int(env.game.lives)} phase={env.game.phase}"
        )
    env.close()

    if gif_path and frames:
        os.makedirs(os.path.dirname(os.path.abspath(gif_path)) or ".", exist_ok=True)
        save_frames_as_gif(frames, gif_path, fps=20)
        print(f"Saved GIF -> {gif_path} ({len(frames)} frames)")

    summary = {
        "episodes": len(results),
        "mean_reward": float(np.mean([r["reward"] for r in results])),
        "mean_waves": float(np.mean([r["wave"] for r in results])),
        "wins": sum(1 for r in results if r["phase"] == "won"),
        "losses": sum(1 for r in results if r["phase"] == "lost"),
    }
    print(f"\nSummary: {summary}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["random", "greedy", "rule", "ppo"], default="greedy")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--render-steps", type=int, default=0,
                   help="Render every N episodes (0 = off unless --gif).")
    p.add_argument("--gif", type=str, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    evaluate(
        agent_name=args.agent,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render_every=args.render_steps,
        gif_path=args.gif,
        ckpt=args.ckpt,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
