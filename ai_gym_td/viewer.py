"""Optional Pygame viewer for interactive play and live training visualization.

Importing this module is safe even if pygame is not installed — it raises a
clear ImportError only when `PygameViewer` is instantiated.
"""

from __future__ import annotations

from typing import Optional, Tuple

from ai_gym_td.config import GameConfig, TILE_BASE, TILE_BLOCKED, TILE_GRASS, TILE_PATH, TILE_SPAWN
from ai_gym_td.engine import TowerDefenseGame


PALETTE = {
    TILE_GRASS: (52, 96, 58),
    TILE_PATH: (170, 150, 110),
    TILE_SPAWN: (200, 110, 80),
    TILE_BASE: (110, 150, 220),
    TILE_BLOCKED: (70, 70, 80),
}


class PygameViewer:
    """Interactive pygame window for `render_mode='human'` or manual play."""

    def __init__(self, config: GameConfig, cell_px: int = 36, hud_height: int = 56, caption: str = "AI Gym Tower Defense"):
        try:
            import pygame  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PygameViewer requires pygame. Install with `pip install pygame`."
            ) from e
        import pygame

        self.pygame = pygame
        self.cfg = config
        self.cell = cell_px
        self.W = config.width * cell_px
        self.H = config.height * cell_px + hud_height
        self.hud_h = hud_height

        pygame.init()
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", max(12, cell_px // 3))

    def draw(self, game: TowerDefenseGame, fps: int = 30) -> None:
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((20, 22, 28))

        # Grid
        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                code = game.grid[y][x]
                color = PALETTE.get(code, (0, 0, 0))
                if code == TILE_GRASS and (x + y) % 2 == 1:
                    color = tuple(max(0, c - 10) for c in color)
                rect = pygame.Rect(x * self.cell, y * self.cell, self.cell, self.cell)
                pygame.draw.rect(self.screen, color, rect)

        # Path arrows
        if len(game.path) >= 2:
            for i in range(0, len(game.path) - 1, 2):
                ax, ay = game.path[i]
                bx, by = game.path[i + 1]
                p1 = (ax * self.cell + self.cell // 2, ay * self.cell + self.cell // 2)
                p2 = (bx * self.cell + self.cell // 2, by * self.cell + self.cell // 2)
                pygame.draw.line(self.screen, (220, 210, 180), p1, p2, 2)

        # Towers
        for t in game.towers.values():
            cx = t.x * self.cell + self.cell // 2
            cy = t.y * self.cell + self.cell // 2
            r = self.cell // 2 - 2
            pygame.draw.circle(self.screen, t.spec.color, (cx, cy), r)
            pygame.draw.circle(self.screen, (10, 10, 12), (cx, cy), r, 2)
            if t.target_eid is not None:
                rr = int(t.spec.range * self.cell)
                pygame.draw.circle(self.screen, t.spec.color, (cx, cy), rr, 1)
            label = self.font.render(t.spec.symbol, True, (20, 20, 24))
            self.screen.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))

        # Enemies
        for e in game.enemies.values():
            cx = int((e.x + 0.5) * self.cell)
            cy = int((e.y + 0.5) * self.cell)
            r = int(e.spec.radius * self.cell)
            pygame.draw.circle(self.screen, e.spec.color, (cx, cy), r)
            # HP bar
            bar_w = self.cell - 4
            bar_h = max(2, self.cell // 10)
            bx = cx - bar_w // 2
            by = cy - r - bar_h - 2
            pygame.draw.rect(self.screen, (30, 30, 34), (bx, by, bar_w, bar_h))
            frac = max(0.0, min(1.0, e.hp / e.max_hp if e.max_hp else 0.0))
            color = (120, 220, 120) if frac > 0.5 else (230, 80, 80)
            pygame.draw.rect(self.screen, color, (bx, by, int(bar_w * frac), bar_h))

        # Projectiles
        for p in game.projectiles.values():
            cx = int(p.x * self.cell)
            cy = int(p.y * self.cell)
            pygame.draw.circle(self.screen, (255, 230, 120), (cx, cy), 3)

        # HUD
        hud_y = self.cfg.height * self.cell
        pygame.draw.rect(self.screen, (14, 16, 20), (0, hud_y, self.W, self.hud_h))
        hud_text = (
            f"Wave {game.wave_number}/{self.cfg.max_waves}   "
            f"Gold {game.gold}   "
            f"Lives {int(game.lives)}   "
            f"Enemies {game.active_enemy_count()}   "
            f"Towers {game.tower_count()}   "
            f"Phase {game.phase}"
        )
        label = self.font.render(hud_text, True, (240, 240, 240))
        self.screen.blit(label, (8, hud_y + 8))
        if game.is_terminal:
            msg = "VICTORY" if game.phase == TowerDefenseGame.WON else "DEFEAT"
            big = self.font.render(msg, True, (255, 220, 80))
            self.screen.blit(big, ((self.W - big.get_width()) // 2, hud_y + 28))

        pygame.display.flip()
        self.clock.tick(fps)

    def close(self) -> None:
        try:
            self.pygame.quit()
        except Exception:
            pass


__all__ = ["PygameViewer"]
