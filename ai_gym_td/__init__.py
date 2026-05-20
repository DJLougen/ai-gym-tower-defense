"""AI Gym Tower Defense — a tower defense game with a Gymnasium API for AI research."""

from ai_gym_td.config import GameConfig, TowerSpec, EnemySpec
from ai_gym_td.engine import TowerDefenseGame
from ai_gym_td.env import TowerDefenseEnv
from ai_gym_td import gym_register  # noqa: F401  (registers TowerDefense-v0)

# Optional LLM agent imports (require API keys and external packages)
try:
    from ai_gym_td.llm_agents import (
        LLMAgent,
        OpenAIAgent,
        AnthropicAgent,
        GoogleAgent,
        OllamaAgent,
        create_llm_agent,
    )
    from ai_gym_td.obs_format import format_obs_for_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

__version__ = "0.2.0"
__all__ = [
    "GameConfig", "TowerSpec", "EnemySpec", "TowerDefenseGame", "TowerDefenseEnv",
    "LLMAgent", "OpenAIAgent", "AnthropicAgent", "GoogleAgent", "OllamaAgent",
    "create_llm_agent", "format_obs_for_llm", "LLM_AVAILABLE"
]
