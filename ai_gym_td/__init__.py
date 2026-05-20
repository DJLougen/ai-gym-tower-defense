"""AI Gym Tower Defense — a tower defense game with a Gymnasium API for AI research."""

from ai_gym_td.config import GameConfig, TowerSpec, EnemySpec
from ai_gym_td.engine import TowerDefenseGame
from ai_gym_td.env import TowerDefenseEnv
from ai_gym_td import gym_register  # noqa: F401  (registers TowerDefense-v0)

__version__ = "0.1.0"
__all__ = ["GameConfig", "TowerSpec", "EnemySpec", "TowerDefenseGame", "TowerDefenseEnv"]
