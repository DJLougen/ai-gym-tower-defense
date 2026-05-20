"""Register TowerDefenseEnv with gymnasium.make so users can do:

    import gymnasium as gym
    import ai_gym_td  # registers on import
    env = gym.make("TowerDefense-v0")
"""

from gymnasium.envs.registration import register


register(
    id="TowerDefense-v0",
    entry_point="ai_gym_td.env:TowerDefenseEnv",
    max_episode_steps=4000,
    kwargs={},
)
