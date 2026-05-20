"""Headless PIL renderer — produces RGB numpy frames for video / GIF export.

Design goals:
- Pure Python + PIL + numpy, no GPU or display required.
- Deterministic output given the same game state.
- Readable at 16px per cell (the default), but scales up cleanly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ai_gym_td.config import GameConfig, TILE_BASE, TILE_BLOCKED, TILE_GRASS, TILE_PATH, TILE_SPAWN
from ai_gym_td.engine import TowerDefenseGame


# Palette (R, G, B)
COLOR_BG = (20, 22, 28)
COLOR_GRASS = (52, 96, 58)
COLOR_GRASS_DARK = (42, 80, 48)
COLOR_PATH = (170, 150, 110)
COLOR_SPAWN = (200, 110, 80)
COLOR_BASE = (110, 150, 220)
COLOR_BLOCKED = (70, 70, 80)
COLOR_HP_GOOD = (120, 220, 120)
COLOR_HP_BAD = (230, 80, 80)
COLOR_PROJECTILE = (255, 230, 120)
COLOR_TEXT = (240, 240, 240)


class PILRenderer:
    """Render game state to RGB numpy arrays using Pillow."""

    def __init__(self, config: GameConfig, cell_px: int = 28, hud_height: int = 48):
        try:
            from PIL import Image, ImageDraw, ImageFont  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PILRenderer requires Pillow. Install with `pip install Pillow`."
            ) from e
        self.cfg = config
        self.cell = cell_px
        self.hud_h = hud_height
        self.W = config.width * cell_px
        self.H = config.height * cell_px + hud_height
        self._font = self._load_font(size=max(10, cell_px // 2))

    def _load_font(self, size: int):
        from PIL import ImageFont

        # Try a built-in that ships with Pillow, then fall back to the default bitmap.
        for name in ("DejaVuSans.ttf", "arial.ttf", "FreeSans.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def render_frame(self, game: TowerDefenseGame) -> np.ndarray:
        """Return an (H, W, 3) uint8 numpy array representing the current frame."""
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (self.W, self.H), COLOR_BG)
        draw = ImageDraw.Draw(img)
        self._draw_grid(draw, game)
        self._draw_towers(draw, game)
        self._draw_enemies(draw, game)
        self._draw_projectiles(draw, game)
        self._draw_hud(draw, game)
        return np.asarray(img, dtype=np.uint8)

    # ---- internals -----------------------------------------------------

    def _cell_xy(self, x: int, y: int) -> Tuple[int, int]:
        return x * self.cell, y * self.cell

    def _draw_grid(self, draw, game: TowerDefenseGame) -> None:
        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                code = game.grid[y][x]
                if code == TILE_GRASS:
                    color = COLOR_GRASS if (x + y) % 2 == 0 else COLOR_GRASS_DARK
                elif code == TILE_PATH:
                    color = COLOR_PATH
                elif code == TILE_SPAWN:
                    color = COLOR_SPAWN
                elif code == TILE_BASE:
                    color = COLOR_BASE
                elif code == TILE_BLOCKED:
                    color = COLOR_BLOCKED
                else:
                    color = COLOR_BG
                px, py = self._cell_xy(x, y)
                draw.rectangle((px, py, px + self.cell - 1, py + self.cell - 1), fill=color)
        # Path arrows for readability.
        if len(game.path) >= 2:
            for i in range(0, len(game.path) - 1, 2):
                ax, ay = game.path[i]
                bx, by = game.path[i + 1]
                axp, ayp = ax * self.cell + self.cell // 2, ay * self.cell + self.cell // 2
                bxp, byp = bx * self.cell + self.cell // 2, by * self.cell + self.cell // 2
                draw.line((axp, ayp, bxp, byp), fill=(220, 210, 180), width=2)

    def _draw_towers(self, draw, game: TowerDefenseGame) -> None:
        for t in game.towers.values():
            px, py = self._cell_xy(t.x, t.y)
            cx, cy = px + self.cell // 2, py + self.cell // 2
            r = self.cell // 2 - 2
            # Base plate
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=t.spec.color, outline=(10, 10, 12))
            # Letter
            bbox = self._text_bbox(draw, t.spec.symbol)
            tw, th = bbox
            draw.text((cx - tw // 2, cy - th // 2), t.spec.symbol, fill=(20, 20, 24), font=self._font)
            # Range ring if currently targeting.
            if t.target_eid is not None:
                rr = int(t.spec.range * self.cell)
                draw.ellipse(
                    (cx - rr, cy - rr, cx + rr, cy + rr),
                    outline=(*t.spec.color, 128),
                    width=1,
                )

    def _draw_enemies(self, draw, game: TowerDefenseGame) -> None:
        for e in game.enemies.values():
            cx = int((e.x + 0.5) * self.cell)
            cy = int((e.y + 0.5) * self.cell)
            r = int(e.spec.radius * self.cell)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=e.spec.color, outline=(10, 10, 12))
            # HP bar
            bar_w = self.cell - 4
            bar_h = max(2, self.cell // 10)
            bx = cx - bar_w // 2
            by = cy - r - bar_h - 2
            draw.rectangle((bx, by, bx + bar_w, by + bar_h), fill=(30, 30, 34))
            frac = max(0.0, min(1.0, e.hp / e.max_hp if e.max_hp else 0.0))
            color = COLOR_HP_GOOD if frac > 0.5 else COLOR_HP_BAD
            draw.rectangle((bx, by, bx + int(bar_w * frac), by + bar_h), fill=color)
            # Slow indicator
            if e.slow_mult < 1.0:
                draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=(150, 200, 255))

    def _draw_projectiles(self, draw, game: TowerDefenseGame) -> None:
        for p in game.projectiles.values():
            cx = int(p.x * self.cell)
            cy = int(p.y * self.cell)
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=COLOR_PROJECTILE)

    def _draw_hud(self, draw, game: TowerDefenseGame) -> None:
        y_base = self.cfg.height * self.cell + 4
        draw.rectangle((0, self.cfg.height * self.cell, self.W, self.H), fill=(14, 16, 20))
        text = (
            f"Wave {game.wave_number}/{self.cfg.max_waves}   "
            f"Gold {game.gold}   "
            f"Lives {int(game.lives)}   "
            f"Enemies {game.active_enemy_count()}   "
            f"Towers {game.tower_count()}   "
            f"Phase {game.phase}"
        )
        draw.text((8, y_base), text, fill=COLOR_TEXT, font=self._font)
        if game.is_terminal:
            msg = "VICTORY" if game.phase == TowerDefenseGame.WON else "DEFEAT"
            bbox = self._text_bbox(draw, msg)
            tx = (self.W - bbox[0]) // 2
            draw.text((tx, y_base + 18), msg, fill=(255, 220, 80), font=self._font)

    def _text_bbox(self, draw, text: str) -> Tuple[int, int]:
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=self._font)
            return right - left, bottom - top
        except Exception:
            # Fallback for older Pillow versions without textbbox.
            return draw.textsize(text, font=self._font)


def save_frames_as_gif(frames, path: str, fps: int = 15, loop: int = 0) -> str:
    """Write a list of (H, W, 3) uint8 arrays to an animated GIF."""
    from PIL import Image

    if not frames:
        raise ValueError("No frames to save.")
    images = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / fps)
    images[0].save(
        path, save_all=True, append_images=images[1:],
        duration=duration_ms, loop=loop, optimize=False,
    )
    return path


def save_frames_as_mp4(frames, path: str, fps: int = 30) -> str:
    """Write frames to an MP4 via imageio (requires imageio-ffmpeg)."""
    import imageio.v2 as imageio

    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()
    return path


__all__ = ["PILRenderer", "save_frames_as_gif", "save_frames_as_mp4"]
