"""CLI: train a PPO agent.

Forwards to `ai_gym_td.ppo.main`. Exists so `ai-td-train` and
`python -m ai_gym_td.scripts.train` both work.
"""

from ai_gym_td.ppo import main

if __name__ == "__main__":
    main()
