"""CLI: play tower defense manually via pygame.

Controls:
  - Click on a grass tile to build a tower.
  - Keys 1..4 select the tower type.
  - SPACE skips the build phase.
  - ESC quits.
"""

from __future__ import annotations

import argparse
import sys


def play(cell_px: int = 36, target_fps: int = 30, seed: int = 0) -> int:
    try:
        import pygame
    except ImportError:
        print("pygame is required for interactive play. Install with `pip install pygame`.")
        return 1

    from ai_gym_td.config import GameConfig, TILE_GRASS
    from ai_gym_td.engine import TowerDefenseGame
    from ai_gym_td.viewer import PygameViewer

    cfg = GameConfig.default_20x12()
    game = TowerDefenseGame(cfg, seed=seed)
    viewer = PygameViewer(cfg, cell_px=cell_px)
    tower_names = tuple(sorted(cfg.towers.keys()))
    selected = 0  # index into tower_names
    dt = 1.0 / cfg.tick_rate
    clock = pygame.time.Clock()

    running = True
    print(f"Playing with {len(tower_names)} towers: {tower_names}")
    print("Keys 1..4 select tower type; SPACE skips build; ESC quits.")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    game.start_next_wave()
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if idx < len(tower_names):
                        selected = idx
                        print(f"Selected: {tower_names[selected]} (cost {cfg.towers[tower_names[selected]].cost})")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                x = mx // cell_px
                y = my // cell_px
                if 0 <= y < cfg.height and 0 <= x < cfg.width and game.grid[y][x] == TILE_GRASS:
                    tid = game.build_tower(x, y, tower_names[selected])
                    if tid is None:
                        print(f"Cannot build {tower_names[selected]} at ({x},{y}) — insufficient gold or occupied.")

        # Step simulation several sub-steps per frame for smooth motion.
        for _ in range(4):
            if game.is_terminal:
                break
            game.step(dt)
        viewer.draw(game, fps=target_fps)
        clock.tick(target_fps)

    viewer.close()
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell-px", type=int, default=36)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    sys.exit(play(cell_px=args.cell_px, target_fps=args.fps, seed=args.seed))


if __name__ == "__main__":
    main()
