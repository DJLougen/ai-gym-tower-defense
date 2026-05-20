"""Quick-start example — run a few episodes with a greedy baseline."""

from ai_gym_td.agents import GreedyAgent
from ai_gym_td.env import TowerDefenseEnv


def main():
    env = TowerDefenseEnv(max_env_steps=2000)
    agent = GreedyAgent()

    for episode in range(3):
        obs, info = env.reset(seed=episode)
        total = 0.0
        steps = 0
        while True:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            if terminated or truncated:
                break
        print(
            f"episode {episode}: steps={steps}, reward={total:.2f}, "
            f"wave={info['wave']}/{env.cfg.max_waves}, lives={int(info['lives'])}"
        )
    env.close()


if __name__ == "__main__":
    main()
