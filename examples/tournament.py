"""Tournament — pit the shipped baseline agents against each other."""

from ai_gym_td.agents import GreedyAgent, RandomAgent, RuleBasedAgent
from ai_gym_td.env import TowerDefenseEnv


AGENTS = {
    "random": lambda: RandomAgent(seed=0),
    "greedy": lambda: GreedyAgent(seed=0),
    "rule": lambda: RuleBasedAgent(seed=0),
}


def run_one(agent_name, agent, env, seed):
    obs, info = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    total = 0.0
    steps = 0
    while True:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if terminated or truncated:
            break
    return {
        "agent": agent_name,
        "seed": seed,
        "steps": steps,
        "reward": total,
        "wave": info["wave"],
        "lives": int(info["lives"]),
        "phase": info["phase"],
    }


def main(episodes: int = 5, base_seed: int = 42):
    env = TowerDefenseEnv(max_env_steps=2000)
    results = []
    for name, factory in AGENTS.items():
        agent = factory()
        for ep in range(episodes):
            results.append(run_one(name, agent, env, base_seed + ep))
    env.close()

    # Print a compact leaderboard.
    from collections import defaultdict
    by_agent = defaultdict(list)
    for r in results:
        by_agent[r["agent"]].append(r)
    print(f"{'agent':<10} {'ep':>4} {'mean reward':>12} {'mean wave':>10} {'wins':>5} {'loss':>5}")
    for name, runs in sorted(by_agent.items()):
        mean_r = sum(r["reward"] for r in runs) / len(runs)
        mean_w = sum(r["wave"] for r in runs) / len(runs)
        wins = sum(1 for r in runs if r["phase"] == "won")
        losses = sum(1 for r in runs if r["phase"] == "lost")
        print(f"{name:<10} {len(runs):>4} {mean_r:>12.2f} {mean_w:>10.2f} {wins:>5} {losses:>5}")


if __name__ == "__main__":
    main()
