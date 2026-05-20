"""Smoke tests — these run in <5s and catch engine / env / render regressions."""

import numpy as np
import pytest


def test_engine_runs_a_full_game():
    from ai_gym_td.config import GameConfig
    from ai_gym_td.engine import TowerDefenseGame

    cfg = GameConfig.default_20x12()
    game = TowerDefenseGame(cfg, seed=0)

    # Build some towers along the path's edges.
    placed = 0
    for y in range(cfg.height):
        for x in range(cfg.width):
            if placed >= 4:
                break
            if game.grid[y][x] == 0 and game.build_tower(x, y, "archer") is not None:
                placed += 1
    assert placed == 4, "Should be able to place 4 archers with starting gold"

    # Run the simulation until terminal or a safe cap.
    dt = 1.0 / cfg.tick_rate
    for _ in range(5000):
        if game.is_terminal:
            break
        game.step(dt)

    assert game.is_terminal, "Game should terminate"
    assert game.phase in (TowerDefenseGame.WON, TowerDefenseGame.LOST)


def test_env_resets_and_steps():
    from ai_gym_td.env import TowerDefenseEnv

    env = TowerDefenseEnv()
    obs, info = env.reset(seed=1)
    assert "grid" in obs and "global" in obs and "action_mask" in obs
    assert obs["grid"].shape[2] == 8

    # Step with a pass action.
    action = np.array([0, 0, 0])
    obs2, r, term, trunc, info2 = env.step(action)
    assert isinstance(r, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)

    # Step with an invalid build (on spawn).
    spawn_x, spawn_y = env.game.spawn
    action = np.array([1, spawn_y, spawn_x])
    obs3, r, term, trunc, info3 = env.step(action)
    assert info3["invalid_action"] is True

    env.close()


def test_action_mask_is_consistent_with_legality():
    from ai_gym_td.env import TowerDefenseEnv

    env = TowerDefenseEnv()
    env.reset(seed=2)
    mask = env.action_masks()

    # Pass (dim 0 == 0) is always legal.
    assert mask[0].all()

    # For any cell marked 1 in a tower mask, build_tower should succeed.
    for ti in range(1, mask.shape[0]):
        name = env._tower_names[ti - 1]
        ys, xs = np.where(mask[ti] > 0)
        for y, x in zip(ys, xs):
            # Simulate a fresh game state per build test so placements don't interact.
            fresh = type(env)(max_env_steps=100)
            fresh.reset(seed=2)
            # Grant enough gold in case the tower is expensive.
            fresh.game._gold = max(fresh.game._gold, fresh.cfg.towers[name].cost)
            tid = fresh.game.build_tower(int(x), int(y), name)
            assert tid is not None, f"mask claimed ({x},{y}) was legal for {name}"
            fresh.close()

    env.close()


def test_pathfinding_finds_s_curve():
    from ai_gym_td.config import GameConfig
    from ai_gym_td.pathfinding import a_star, path_is_clear

    cfg = GameConfig.default_20x12()
    grid = cfg.make_grid()
    spawn = None
    base = None
    for y in range(cfg.height):
        for x in range(cfg.width):
            if grid[y][x] == 2:
                spawn = (x, y)
            elif grid[y][x] == 3:
                base = (x, y)
    assert spawn and base
    assert path_is_clear(grid, spawn, base)
    path = a_star(grid, spawn, base)
    assert path and path[0] == spawn and path[-1] == base


def test_pil_renderer_produces_frames():
    from ai_gym_td.config import GameConfig
    from ai_gym_td.engine import TowerDefenseGame
    from ai_gym_td.render import PILRenderer

    cfg = GameConfig.default_20x12()
    game = TowerDefenseGame(cfg, seed=0)
    renderer = PILRenderer(cfg, cell_px=16, hud_height=32)
    frame = renderer.render_frame(game)
    expected_h = cfg.height * 16 + 32
    expected_w = cfg.width * 16
    assert frame.shape == (expected_h, expected_w, 3)
    assert frame.dtype == np.uint8


def test_greedy_agent_beats_nothing():
    from ai_gym_td.agents import GreedyAgent
    from ai_gym_td.env import TowerDefenseEnv

    env = TowerDefenseEnv(max_env_steps=1500)
    agent = GreedyAgent(seed=0)
    obs, info = env.reset(seed=7)
    total = 0.0
    for _ in range(1500):
        a = agent.act(obs, info)
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    env.close()
    # Greedy should reliably survive at least a few waves.
    assert info["wave"] >= 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
