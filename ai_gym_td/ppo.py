"""PPO for Tower Defense — CleanRL-style, single-file, minimal deps.

Usage:
    python -m ai_gym_td.scripts.train --total-timesteps 500_000

The agent uses a small CNN over the spatial grid plus an MLP over the
global features. Action space is MultiDiscrete([T+1, H, W]) with optional
masking from `obs["action_mask"]`.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions.categorical import Categorical
except ImportError as e:
    raise ImportError(
        "PPO requires PyTorch. Install with `pip install ai-gym-tower-defense[rl]`."
    ) from e

import gymnasium as gym

from ai_gym_td.env import TowerDefenseEnv


# ---- Agent ------------------------------------------------------------------


class TowerDefensePPOAgent(nn.Module):
    """CNN + MLP actor-critic with action masking."""

    def __init__(self, env: TowerDefenseEnv):
        super().__init__()
        obs_space = env.observation_space
        grid_shape = obs_space["grid"].shape   # (H, W, C)
        H, W, C = grid_shape
        global_dim = obs_space["global"].shape[0]
        md = env.action_space.nvec  # [T+1, H, W]
        self.action_dims = tuple(int(d) for d in md)

        # CNN over spatial grid. Channels-first for PyTorch.
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Infer the flattened size with a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            cnn_out = self.cnn(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out + global_dim, 256),
            nn.ReLU(),
        )
        # One head per action dimension.
        self.policy_heads = nn.ModuleList(
            [nn.Linear(256, d) for d in self.action_dims]
        )
        self.value_head = nn.Linear(256, 1)

    def _features(self, obs: dict) -> torch.Tensor:
        grid = obs["grid"]          # (B, H, W, C) or (H, W, C)
        glob = obs["global"]        # (B, G) or (G,)
        if grid.ndim == 3:
            grid = grid.unsqueeze(0)
            glob = glob.unsqueeze(0)
        # Channels-last -> channels-first
        grid = grid.permute(0, 3, 1, 2).contiguous()
        cnn = self.cnn(grid)
        x = torch.cat([cnn, glob], dim=1)
        return self.fc(x)

    def get_value(self, obs: dict) -> torch.Tensor:
        return self.value_head(self._features(obs)).squeeze(-1)

    def get_action_and_value(
        self,
        obs: dict,
        action: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ):
        feats = self._features(obs)
        logits = [head(feats) for head in self.policy_heads]  # list of (B, D_i)

        # Apply per-dim masks if provided.
        if action_mask is not None:
            # action_mask shape: (B, T+1, H, W) — one sub-mask per action dim.
            # We assume it's pre-split into per-dim masks by the caller.
            for i, mask_i in enumerate(action_mask):
                # mask_i: (B, D_i)
                big_neg = torch.finfo(logits[i].dtype).min
                logits[i] = logits[i] + torch.where(mask_i > 0, torch.zeros_like(logits[i]), big_neg)

        dists = [Categorical(logits=lg) for lg in logits]
        if action is None:
            action = torch.stack([d.sample() for d in dists], dim=-1)  # (B, K)
        log_prob = sum(d.log_prob(action[:, i]) for i, d in enumerate(dists))
        entropy = sum(d.entropy() for d in dists)
        value = self.value_head(feats).squeeze(-1)
        return action, log_prob, entropy, value


# ---- Observation / mask batching --------------------------------------------


def batch_obs(obs_list):
    """Stack a list of Dict observations into a single batched dict."""
    grid = np.stack([o["grid"] for o in obs_list], axis=0).astype(np.float32)
    glob = np.stack([o["global"] for o in obs_list], axis=0).astype(np.float32)
    # MultiDiscrete action mask: env gives (T+1, H, W); split per-dim.
    raw_mask = np.stack([o["action_mask"] for o in obs_list], axis=0)  # (B, T+1, H, W)
    B, Tp1, H, W = raw_mask.shape
    T = Tp1 - 1
    # Build per-dim masks:
    #   dim 0 (tower type): any (y, x) legal for this type -> legal.
    #   dim 1 (y): exists tower type t and column x s.t. mask[t, y, x] = 1 OR t=0 (pass).
    #   dim 2 (x): symmetric.
    m_t = raw_mask.any(axis=(2, 3))                  # (B, T+1)
    m_y_pass = np.ones((B, H), dtype=np.int8)
    m_x_pass = np.ones((B, W), dtype=np.int8)
    m_y_build = raw_mask[:, 1:].any(axis=(1, 3))     # (B, H)
    m_x_build = raw_mask[:, 1:].any(axis=(1, 2))     # (B, W)
    m_y = np.maximum(m_y_pass, m_y_build)
    m_x = np.maximum(m_x_pass, m_x_build)

    return {
        "grid": torch.from_numpy(grid),
        "global": torch.from_numpy(glob),
    }, [
        torch.from_numpy(m_t),
        torch.from_numpy(m_y),
        torch.from_numpy(m_x),
    ]


# ---- Training loop ----------------------------------------------------------


@dataclass
class PPOConfig:
    total_timesteps: int = 500_000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    num_steps: int = 256
    gamma: float = 0.995
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    seed: int = 1
    cuda: bool = True
    log_dir: str = "runs"
    run_name: str = "td_ppo"


def train(cfg: PPOConfig) -> TowerDefensePPOAgent:
    run_name = f"{cfg.run_name}__{cfg.seed}__{int(time.time())}"
    # Optional tensorboard.
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(cfg.log_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(cfg.log_dir, run_name))
    except Exception:
        writer = None

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = TowerDefenseEnv()
    env.reset(seed=cfg.seed)
    agent = TowerDefensePPOAgent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # Storage
    num_steps = cfg.num_steps
    obs_grid = torch.zeros((num_steps, *env.observation_space["grid"].shape)).to(device)
    obs_glob = torch.zeros((num_steps, *env.observation_space["global"].shape)).to(device)
    actions = torch.zeros((num_steps, 3), dtype=torch.int64).to(device)
    logprobs = torch.zeros(num_steps).to(device)
    rewards = torch.zeros(num_steps).to(device)
    dones = torch.zeros(num_steps).to(device)
    values = torch.zeros(num_steps).to(device)
    masks_t = torch.zeros((num_steps, env.action_space.nvec[0]), dtype=torch.int8).to(device)
    masks_y = torch.zeros((num_steps, env.action_space.nvec[1]), dtype=torch.int8).to(device)
    masks_x = torch.zeros((num_steps, env.action_space.nvec[2]), dtype=torch.int8).to(device)

    obs, info = env.reset(seed=cfg.seed)
    next_obs = obs
    next_done = False
    num_updates = cfg.total_timesteps // num_steps
    global_step = 0
    episode_returns = []
    episode_return = 0.0

    start = time.time()
    for update in range(1, num_updates + 1):
        for step in range(num_steps):
            global_step += 1
            batched_obs, batched_masks = batch_obs([next_obs])
            batched_obs = {k: v.to(device) for k, v in batched_obs.items()}
            batched_masks = [m.to(device) for m in batched_masks]

            with torch.no_grad():
                act, lg, _, val = agent.get_action_and_value(batched_obs, action_mask=batched_masks)
            a = act[0].cpu().numpy()
            obs_grid[step] = torch.from_numpy(obs["grid"])
            obs_glob[step] = torch.from_numpy(obs["global"])
            actions[step] = torch.from_numpy(a)
            logprobs[step] = lg[0]
            values[step] = val[0]
            masks_t[step] = batched_masks[0][0]
            masks_y[step] = batched_masks[1][0]
            masks_x[step] = batched_masks[2][0]

            obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            episode_return += reward
            rewards[step] = reward
            dones[step] = float(done)
            next_obs = obs
            next_done = done
            if done:
                episode_returns.append(episode_return)
                if writer:
                    writer.add_scalar("charts/episodic_return", episode_return, global_step)
                episode_return = 0.0
                obs, info = env.reset(seed=cfg.seed + len(episode_returns))
                next_obs = obs

        # GAE
        with torch.no_grad():
            b_obs = {"grid": obs_grid, "global": obs_glob}
            next_obs_batched, _ = batch_obs([next_obs])
            next_obs_batched = {k: v.to(device) for k, v in next_obs_batched.items()}
            next_value = agent.get_value(next_obs_batched)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - float(next_done)
                    nextvalues = next_value[0]
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values

        # Flatten
        b_obs_grid = obs_grid.reshape(-1, *obs_grid.shape[1:])
        b_obs_glob = obs_glob.reshape(-1, *obs_glob.shape[1:])
        b_actions = actions.reshape(-1, 3)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_masks = [masks_t.reshape(-1, masks_t.shape[-1]),
                   masks_y.reshape(-1, masks_y.shape[-1]),
                   masks_x.reshape(-1, masks_x.shape[-1])]

        b_inds = np.arange(num_steps)
        batch_size = num_steps
        minibatch_size = batch_size // cfg.num_minibatches
        for _ in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = {"grid": b_obs_grid[mb_inds], "global": b_obs_glob[mb_inds]}
                mb_masks = [m[mb_inds] for m in b_masks]
                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    mb_obs, action=b_actions[mb_inds], action_mask=mb_masks,
                )
                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean().item()
                    approx_ent = entropy.mean().item()
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()
                e_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * e_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        elapsed = max(1e-6, time.time() - start)
        sps = int(global_step / elapsed)
        if writer:
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/entropy", e_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", old_approx_kl, global_step)
        if update % 5 == 0 or update == num_updates:
            ret_str = f"{np.mean(episode_returns[-20:]) if episode_returns else 0.0:.2f}"
            print(
                f"[update {update}/{num_updates}] SPS={sps} "
                f"pg={pg_loss.item():.3f} v={v_loss.item():.3f} "
                f"ent={e_loss.item():.3f} kl={old_approx_kl:.3f} "
                f"ep_ret(20)={ret_str}"
            )

    if writer:
        writer.close()
    env.close()
    return agent


# ---- CLI --------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--num-steps", type=int, default=256)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", action="store_true", default=False)
    p.add_argument("--run-name", type=str, default="td_ppo")
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--save-path", type=str, default=None,
                   help="If set, saves the final agent state_dict to this path.")
    args = p.parse_args()

    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        seed=args.seed,
        cuda=args.cuda,
        run_name=args.run_name,
        log_dir=args.log_dir,
    )
    agent = train(cfg)
    if args.save_path:
        torch.save(agent.state_dict(), args.save_path)
        print(f"Saved agent to {args.save_path}")


if __name__ == "__main__":
    main()


__all__ = ["TowerDefensePPOAgent", "PPOConfig", "train", "batch_obs"]
