<div align="center">

# AI Gym Tower Defense 🏰

**A tower defense game with a [Gymnasium](https://gymnasium.farama.org/) API for training AI agents.**

Train RL agents, pit heuristic bots against each other, or play by hand — all on the same deterministic engine.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![Gymnasium](https://img.shields.io/badge/gymnasium-0.29%2B-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Installation](#installation) • [Quick Start](#quick-start) • [Environment API](#environment-api) • [Training](#training) • [Contributing](#contributing)

</div>

---

## Why?

Tower defense is a surprisingly rich sandbox for AI research:
- **Long horizon**: decisions now affect waves minutes later.
- **Spatial reasoning**: tower placement is a 2-D combinatorial problem.
- **Resource management**: gold, lives, and cooldowns must be balanced.
- **Multi-agent potential**: tournaments between learned agents are one `evaluate.py` away.

This package gives you a **small, fast, pure-Python engine** plus a **Gymnasium wrapper** so you can plug it into Stable-Baselines3, CleanRL, TorchRL, or your own loop.

## Features

- 🎮 **Deterministic engine** — pure Python + NumPy, no GPU required.
- 🧭 **4 tower archetypes** (archer, cannon, ice, tesla) and **4 enemy types** (scout, grunt, brute, swarm).
- 🤖 **Gymnasium env** with spatial grid + global feature observations and action masking.
- 🧠 **Built-in PPO** (CleanRL-style) plus random / greedy / rule-based baselines.
- 🖼️ **Headless PIL renderer** for GIF / MP4 export — great for notebooks and CI.
- 🎨 **Optional pygame viewer** for live play.
- 🔁 **Registered as `TowerDefense-v0`** via `gymnasium.make`.

## Installation

```bash
# Minimal install (engine + gym + renderer)
pip install -e .

# With RL training extras
pip install -e ".[rl]"

# With pygame viewer
pip install -e ".[viewer]"

# Everything (rl + viewer + dev tools)
pip install -e ".[all]"
```

## Quick Start

```python
import ai_gym_td
from ai_gym_td.env import TowerDefenseEnv
from ai_gym_td.agents import GreedyAgent

env = TowerDefenseEnv()
agent = GreedyAgent()

obs, info = env.reset(seed=42)
total = 0.0
for _ in range(2000):
    action = agent.act(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    if terminated or truncated:
        break
print(f"reward={total:.1f}, wave={info['wave']}, lives={info['lives']}")
```

Or via `gymnasium.make`:

```python
import gymnasium as gym
import ai_gym_td  # registers TowerDefense-v0 on import

env = gym.make("TowerDefense-v0")
obs, info = env.reset()
```

## Environment API

| Item | Shape / Type | Meaning |
|------|--------------|---------|
| **Observation** | `Dict` with `grid`, `global`, `action_mask` | 8-channel `(H, W, 8)` spatial grid + 6-dim global vector + per-action legality mask |
| **Action** | `MultiDiscrete([T+1, H, W])` | `[tower_type, y, x]`. `tower_type=0` = pass; `1..T` = build that tower type. |
| **Reward** | `float` | `+kill -leak ±terminal`, scaled by `RewardConfig`. |
| **Termination** | `bool` | Win (all waves cleared) or loss (lives ≤ 0). |
| **Truncation** | `bool` | Safety cap (`max_env_steps`). |

### Observation channels

| # | Channel | Meaning |
|---|---------|---------|
| 0 | `terrain` | 0=grass, 0.25=path, 0.5=spawn, 0.75=base, 1.0=blocked |
| 1 | `tower_archer` | 1.0 where an archer is built |
| 2 | `tower_cannon` | 1.0 where a cannon is built |
| 3 | `tower_ice` | 1.0 where ice is built |
| 4 | `tower_tesla` | 1.0 where tesla is built |
| 5 | `enemy_density` | Count of enemies in each cell, normalized |
| 6 | `enemy_hp_frac` | Sum of `hp/max_hp` per cell |
| 7 | `path_distance` | Euclidean distance to nearest path cell, normalized |

Global features (length-6 vector): `gold_norm`, `lives_norm`, `wave_norm`, `phase_build`, `phase_wave`, `enemy_count_norm`.

## Training

### PPO

```bash
python -m ai_gym_td.scripts.train \
    --total-timesteps 500_000 \
    --num-steps 256 \
    --lr 2.5e-4 \
    --run-name td_ppo \
    --save-path agent.pt
```

Logs to TensorBoard under `runs/`.

### Evaluate and render

```bash
# Greedy baseline
python -m ai_gym_td.scripts.evaluate --agent greedy --episodes 5

# Rule-based agent, dump a GIF
python -m ai_gym_td.scripts.evaluate --agent rule --episodes 1 --gif greedy.gif

# Your trained agent
python -m ai_gym_td.scripts.evaluate --agent ppo --ckpt agent.pt --episodes 10 --gif ppo.gif
```

### Play manually (requires pygame)

```bash
python -m ai_gym_td.scripts.play
```

Keys `1..4` select tower type, click a grass tile to build, `SPACE` skips the build phase, `ESC` quits.

## Project layout

```
ai-gym-tower-defense/
├── ai_gym_td/
│   ├── __init__.py        # package + gym registration
│   ├── config.py          # TowerSpec, EnemySpec, GameConfig, RewardConfig
│   ├── engine.py          # deterministic simulation
│   ├── pathfinding.py     # A* + distance-to-path helpers
│   ├── env.py             # Gymnasium env
│   ├── render.py          # PIL renderer + GIF/MP4 export
│   ├── viewer.py          # optional pygame viewer
│   ├── agents.py          # random / greedy / rule-based baselines
│   ├── ppo.py             # CleanRL-style PPO
│   ├── gym_register.py    # TowerDefense-v0 registration
│   └── scripts/
│       ├── train.py       # CLI: train PPO
│       ├── evaluate.py    # CLI: eval + GIF
│       └── play.py        # CLI: manual pygame play
├── examples/
│   ├── quickstart.py
│   └── tournament.py
├── tests/                 # pytest smoke tests
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

## Customization

All knobs live in `ai_gym_td/config.py`. Swap the tower roster, change the map, or reshape the reward:

```python
from ai_gym_td import GameConfig
from ai_gym_td.config import TOWERS, RewardConfig

cfg = GameConfig.default_20x12()
cfg = GameConfig(
    width=24, height=14,
    starting_gold=200, starting_lives=30, max_waves=30,
    towers=dict(TOWERS),
    rewards=RewardConfig(kill=2.0, leak=-5.0, win=100.0, lose=-100.0, scale=0.1),
)
```

## Contributing

PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Roughly:

1. Fork & branch.
2. `pip install -e ".[dev]"`.
3. `ruff check .` and `pytest` before pushing.
4. Open a PR with a clear description and, for gameplay changes, a GIF.

## License

MIT — see [LICENSE](LICENSE).
