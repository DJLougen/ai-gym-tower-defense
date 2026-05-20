"""Gymnasium environment wrapper around TowerDefenseGame.

Conventions
-----------
- One env `step` = one decision: the agent may place a tower (or pass), then
  the engine advances by a fixed amount of simulation time.
- Action space is `MultiDiscrete([num_towers+1, H, W])`:
    - dim 0: 0 = pass, 1..num_towers = build that tower type (sorted by name)
    - dim 1: row (y)
    - dim 2: column (x)
  Invalid builds (occupied cell, not enough gold, not grass, out of bounds,
  during terminal state) are rejected and incur `rewards.invalid_action`.
- Observation space is a `Dict` with a spatial grid stack and a compact
  global vector. See `OBS_CHANNELS` for channel semantics.
- Reward is accumulated over the post-action simulation sub-steps.
- Episode ends on win, loss, or after `max_waves` waves have been resolved.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai_gym_td.config import GameConfig, RewardConfig, TILE_BASE, TILE_BLOCKED, TILE_GRASS, TILE_PATH, TILE_SPAWN
from ai_gym_td.engine import BuildEvent, KillEvent, LeakEvent, TowerDefenseGame
from ai_gym_td.pathfinding import distance_to_path_cells


# Observation channels for the spatial (H, W) grid.
OBS_CHANNELS = (
    "terrain",             # 0: grass=0, path=1, spawn=2, base=3, blocked=4
    "tower_archer",        # 1: 1.0 where archer is built
    "tower_cannon",        # 2: 1.0 where cannon is built
    "tower_ice",           # 3
    "tower_tesla",         # 4
    "enemy_density",       # 5: count of enemies per cell (soft)
    "enemy_hp_frac",       # 6: sum(hp/max_hp) of enemies per cell
    "path_distance",       # 7: min Euclidean distance to the path, normalized
)
NUM_CHANNELS = len(OBS_CHANNELS)

# Global feature vector layout. See `env._global_vec()` for the ordering.
NUM_GLOBAL_FEATURES = 6


def _tower_channel_index(tower_name: str) -> int:
    # Channel indices are hardcoded to match sorted default roster: archer, cannon, ice, tesla.
    order = ("archer", "cannon", "ice", "tesla")
    return 1 + order.index(tower_name)


class TowerDefenseEnv(gym.Env):
    """Gymnasium-compatible tower defense.

    Parameters
    ----------
    config
        Game configuration. Defaults to the canonical 20x12 S-curve map.
    steps_per_action
        How many engine sub-steps to run after each agent decision. At the
        default 30 Hz tick rate and `steps_per_action=4`, one env step
        corresponds to ~133 ms of sim time.
    max_env_steps
        Safety cap; episode truncates if hit. Set `None` for no cap.
    auto_start_waves
        If True (default), waves advance on their own via the build timer.
        If False, the agent must call `info["manual_start_wave"] = True`
        via a wrapper or use the helper env `skip_build_phase()`.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        steps_per_action: int = 4,
        max_env_steps: int = 4000,
        auto_start_waves: bool = True,
        render_mode: Optional[str] = "rgb_array",
    ):
        super().__init__()
        self.cfg = config or GameConfig.default_20x12()
        self.steps_per_action = steps_per_action
        self.max_env_steps = max_env_steps
        self.auto_start_waves = auto_start_waves
        self.render_mode = render_mode

        self.game = TowerDefenseGame(self.cfg)
        self._tower_names: Tuple[str, ...] = tuple(sorted(self.cfg.towers.keys()))
        self._path_dist = self._precompute_path_distance()

        # Spaces.
        self.action_space = spaces.MultiDiscrete(
            [len(self._tower_names) + 1, self.cfg.height, self.cfg.width]
        )
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.cfg.height, self.cfg.width, NUM_CHANNELS),
                dtype=np.float32,
            ),
            "global": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(NUM_GLOBAL_FEATURES,),
                dtype=np.float32,
            ),
            "action_mask": spaces.Box(
                low=0, high=1,
                shape=(len(self._tower_names) + 1, self.cfg.height, self.cfg.width),
                dtype=np.int8,
            ),
        })

        self._renderer = None
        self._step_count = 0

    # ---- gym API --------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.game.reset(seed=seed)
        self._step_count = 0
        obs = self._observe()
        info = self._base_info()
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape[0] != 3:
            raise ValueError(f"Action must have shape (3,), got {action}")
        tower_idx, y, x = int(action[0]), int(action[1]), int(action[2])

        reward = 0.0
        # 1. Try to apply the action (only during non-terminal states).
        placed = False
        invalid = False
        if not self.game.is_terminal:
            if tower_idx == 0:
                # Explicit pass.
                pass
            elif 1 <= tower_idx <= len(self._tower_names):
                name = self._tower_names[tower_idx - 1]
                tid = self.game.build_tower(x, y, name)
                if tid is None:
                    invalid = True
                    reward += self.cfg.rewards.invalid_action
                else:
                    placed = True
            else:
                invalid = True
                reward += self.cfg.rewards.invalid_action
        self.game.invalid_action_this_step = invalid

        # 2. Advance the engine.
        dt = 1.0 / self.cfg.tick_rate
        events_accum: List = []
        for _ in range(self.steps_per_action):
            if self.game.is_terminal:
                break
            self.game.step(dt)
            events_accum.extend(self.game.events_this_step)

        # 3. Shape reward from events.
        rew_cfg = self.cfg.rewards
        for ev in events_accum:
            if isinstance(ev, KillEvent):
                reward += rew_cfg.scale * rew_cfg.kill
            elif isinstance(ev, LeakEvent):
                reward += rew_cfg.scale * rew_cfg.leak
        reward += rew_cfg.step

        # 4. Terminal bonuses.
        terminated = False
        truncated = False
        if self.game.phase == TowerDefenseGame.WON:
            reward += rew_cfg.scale * rew_cfg.win
            terminated = True
        elif self.game.phase == TowerDefenseGame.LOST:
            reward += rew_cfg.scale * rew_cfg.lose
            terminated = True

        self._step_count += 1
        if not terminated and self.max_env_steps and self._step_count >= self.max_env_steps:
            truncated = True

        obs = self._observe()
        info = self._base_info(events=events_accum, placed=placed, invalid=invalid)
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from ai_gym_td.render import PILRenderer
            if self.render_mode == "human":
                try:
                    from ai_gym_td.viewer import PygameViewer
                    self._renderer = PygameViewer(self.cfg)
                except Exception:
                    # Pygame not available; fall back to returning frames.
                    self._renderer = PILRenderer(self.cfg)
                    self.render_mode = "rgb_array"
            else:
                self._renderer = PILRenderer(self.cfg)
        from ai_gym_td.render import PILRenderer as _PIL
        if isinstance(self._renderer, _PIL):
            return self._renderer.render_frame(self.game)
        # Pygame path
        self._renderer.draw(self.game)
        return None

    def close(self):
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    # ---- helpers --------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Boolean mask: True where the action is currently legal."""
        mask = np.zeros(
            (len(self._tower_names) + 1, self.cfg.height, self.cfg.width),
            dtype=np.int8,
        )
        # Pass action is always valid.
        mask[0, :, :] = 1
        if self.game.is_terminal:
            return mask
        for ti, name in enumerate(self._tower_names, start=1):
            spec = self.cfg.towers[name]
            affordable = self.game.gold >= spec.cost
            if not affordable:
                continue
            for y in range(self.cfg.height):
                for x in range(self.cfg.width):
                    if self.game.grid[y][x] != TILE_GRASS:
                        continue
                    if self.game._tower_at(x, y) is not None:
                        continue
                    mask[ti, y, x] = 1
        return mask

    def skip_build_phase(self) -> None:
        """Fast-forward through build downtime. Useful for evaluation / training."""
        if self.game.phase != TowerDefenseGame.BUILD:
            return
        # Force the timer to expire so the next step() transitions into WAVE.
        self.game._build_timer = self.cfg.wave_build_seconds

    # ---- observation assembly ------------------------------------------

    def _precompute_path_distance(self) -> np.ndarray:
        d = distance_to_path_cells(self.cfg.width, self.cfg.height, self.game.path)
        arr = np.asarray(d, dtype=np.float32)
        maxd = float(arr.max()) if arr.size else 1.0
        if maxd > 0:
            arr = arr / maxd
        return arr

    def _observe(self) -> Dict[str, np.ndarray]:
        H, W = self.cfg.height, self.cfg.width
        grid = np.zeros((H, W, NUM_CHANNELS), dtype=np.float32)

        # Channel 0: terrain.
        terrain = np.asarray(self.game.grid, dtype=np.float32) / 4.0
        grid[..., 0] = terrain

        # Channels 1..4: tower one-hot per archetype (only for known names).
        tgrid = self.game.tower_grid
        for ti, name in enumerate(self._tower_names, start=1):
            layer = np.zeros((H, W), dtype=np.float32)
            for y in range(H):
                for x in range(W):
                    if tgrid[y][x] == name:
                        layer[y, x] = 1.0
            grid[..., ti] = layer

        # Channels 5..6: enemy density + hp fraction, smeared onto their cell.
        dens = np.zeros((H, W), dtype=np.float32)
        hp_frac = np.zeros((H, W), dtype=np.float32)
        for e in self.game.enemies.values():
            cx = int(max(0, min(W - 1, math_floor(e.x))))
            cy = int(max(0, min(H - 1, math_floor(e.y))))
            dens[cy, cx] += 1.0
            if e.max_hp > 0:
                hp_frac[cy, cx] += e.hp / e.max_hp
        # Normalize density so channel stays in [0, 1] for typical wave sizes.
        max_d = max(8.0, float(dens.max())) if dens.size else 1.0
        grid[..., 5] = dens / max_d
        grid[..., 6] = np.clip(hp_frac, 0.0, 1.0)

        # Channel 7: path distance (precomputed).
        grid[..., 7] = self._path_dist

        return {
            "grid": grid,
            "global": self._global_vec(),
            "action_mask": self.action_masks(),
        }

    def _global_vec(self) -> np.ndarray:
        # [gold_norm, lives_norm, wave_norm, phase_onehot(2)]
        gold_norm = float(self.game.gold) / max(1, self.cfg.starting_gold * 2)
        lives_norm = float(self.game.lives) / max(1, self.cfg.starting_lives)
        wave_norm = float(self.game.wave_number) / max(1, self.cfg.max_waves)
        phase_build = 1.0 if self.game.phase == TowerDefenseGame.BUILD else 0.0
        phase_wave = 1.0 if self.game.phase == TowerDefenseGame.WAVE else 0.0
        return np.array(
            [gold_norm, lives_norm, wave_norm, phase_build, phase_wave,
             float(self.game.active_enemy_count()) / 16.0],
            dtype=np.float32,
        )

    def _base_info(self, events=None, placed=False, invalid=False) -> Dict[str, Any]:
        return {
            "phase": self.game.phase,
            "gold": self.game.gold,
            "lives": self.game.lives,
            "wave": self.game.wave_number,
            "tick": self.game.tick,
            "placed_tower": placed,
            "invalid_action": invalid,
            "events": events or [],
            "action_mask": self.action_masks(),
            "snapshot": self.game.snapshot(),
        }


def math_floor(v: float) -> int:
    # Avoid pulling in the math module for a single call.
    return int(v) if v >= 0 else int(v) - 1


# ---- Convenience constructors -----------------------------------------------

def make_env(config: Optional[GameConfig] = None, render_mode: Optional[str] = None, **kwargs) -> TowerDefenseEnv:
    """Tiny helper so callers can do `from ai_gym_td.env import make_env`."""
    return TowerDefenseEnv(config=config, render_mode=render_mode, **kwargs)


__all__ = ["TowerDefenseEnv", "make_env", "OBS_CHANNELS", "NUM_CHANNELS", "NUM_GLOBAL_FEATURES"]
