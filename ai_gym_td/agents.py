"""Baseline agents: random, greedy, and a small rule-based policy.

Every agent here implements the minimal `Agent` protocol used by `evaluate.py`
and `play.py`:

    class Agent:
        def act(self, obs, info) -> np.ndarray:
            ...

No training, no state — just a callable. This keeps tournaments and smoke
tests trivial.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class Agent:
    """Abstract base (not a real ABC — we don't want to force inheritance)."""

    def act(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        """Optional hook called between episodes."""


class RandomAgent(Agent):
    """Uniformly samples from the current action mask."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, info):
        mask = obs["action_mask"]  # (T+1, H, W) int8
        flat = mask.reshape(-1).astype(np.int32)
        if flat.sum() == 0:
            return np.array([0, 0, 0], dtype=np.int64)
        probs = flat / flat.sum()
        idx = self.rng.choice(flat.size, p=probs)
        T, H, W = mask.shape
        t = idx // (H * W)
        rem = idx % (H * W)
        y = rem // W
        x = rem % W
        return np.array([t, y, x], dtype=np.int64)


class GreedyAgent(Agent):
    """Places the most expensive affordable tower next to the path.

    This is the "does my env reward sensible play?" sanity check. If a
    trained RL agent can't beat this on average, something is wrong with
    the reward or observation.
    """

    def __init__(self, pass_prob: float = 0.1, seed: int = 0):
        self.pass_prob = pass_prob
        self.rng = np.random.default_rng(seed)

    def act(self, obs, info):
        mask = obs["action_mask"]
        T, H, W = mask.shape
        if self.rng.random() < self.pass_prob:
            return np.array([0, 0, 0], dtype=np.int64)

        # Prefer the tower with the highest cost among legal placements.
        # Towers are sorted by name in the env; we recover specs via info.
        gold = info.get("gold", 0)
        # Iterate tower types from most to least expensive using channel order.
        tower_priority = list(range(T - 1, 0, -1))  # T-1, T-2, ..., 1
        path_dist = obs["grid"][..., 7]  # normalized distance-to-path channel
        for ti in tower_priority:
            legal = mask[ti]  # (H, W)
            if legal.sum() == 0:
                continue
            # Among legal cells, pick the one closest to the path (smallest dist).
            candidates = np.argwhere(legal > 0)
            # candidates is (N, 2) with columns (y, x).
            dists = np.array([path_dist[y, x] for y, x in candidates])
            best = int(np.argmin(dists))
            y, x = candidates[best]
            return np.array([ti, int(y), int(x)], dtype=np.int64)
        # Nothing affordable / placeable — pass.
        return np.array([0, 0, 0], dtype=np.int64)


class RuleBasedAgent(Agent):
    """A slightly smarter heuristic:

    - Keep a small cash reserve on early waves.
    - Build a mix of archer + ice near path bends, cannon on long straights,
      tesla in the late game.
    - Always pass during the first 0.5 s of a build phase to let enemies
      commit (approximated by the global phase flag).
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.steps_since_reset = 0

    def reset(self):
        self.steps_since_reset = 0

    def act(self, obs, info):
        self.steps_since_reset += 1
        mask = obs["action_mask"]
        T, H, W = mask.shape
        global_vec = obs["global"]
        gold_norm, lives_norm, wave_norm = global_vec[0], global_vec[1], global_vec[2]
        phase_build = global_vec[3]

        # Hold cash early.
        if wave_norm < 0.15 and gold_norm < 0.6:
            return np.array([0, 0, 0], dtype=np.int64)

        path_dist = obs["grid"][..., 7]
        enemy_dens = obs["grid"][..., 5]

        # Find cells within ~2 cells of the path (the "kill zone").
        near_path = path_dist < (2.0 / 10.0)  # path_dist is normalized to [0,1]
        # Choose tower type by wave progress.
        if wave_norm > 0.6 and T >= 5 and mask[4].sum() > 0:
            ti = 4  # tesla
        elif wave_norm > 0.3 and T >= 3 and mask[3].sum() > 0:
            ti = 3  # ice
        elif T >= 2 and mask[2].sum() > 0:
            ti = 2  # cannon
        elif mask[1].sum() > 0:
            ti = 1  # archer
        else:
            return np.array([0, 0, 0], dtype=np.int64)

        legal = mask[ti].astype(bool) & near_path
        if not legal.any():
            # Fall back to any legal placement closest to path.
            legal = mask[ti].astype(bool)
            if not legal.any():
                return np.array([0, 0, 0], dtype=np.int64)
        candidates = np.argwhere(legal)
        # Score by: proximity to path + proximity to current enemies.
        scores = []
        for y, x in candidates:
            s = -path_dist[y, x] + 0.5 * enemy_dens[y, x]
            scores.append(s)
        best = int(np.argmax(scores))
        y, x = candidates[best]
        return np.array([ti, int(y), int(x)], dtype=np.int64)


__all__ = ["Agent", "RandomAgent", "GreedyAgent", "RuleBasedAgent"]
