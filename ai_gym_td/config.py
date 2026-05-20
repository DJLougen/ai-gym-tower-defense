"""Game configuration: grid size, towers, enemies, waves, and reward shaping.

Everything tunable lives here. Configs are frozen dataclasses so an experiment
can be reproduced from a single serialized blob.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple
import json

# Grid tile codes — these appear in the map array and in observations.
TILE_GRASS = 0   # buildable
TILE_PATH = 1    # enemy walkway, not buildable
TILE_SPAWN = 2   # enemy entry
TILE_BASE = 3    # enemy exit (the thing you defend)
TILE_BLOCKED = 4 # non-buildable decoration (rocks, water, etc.)


@dataclass(frozen=True)
class TowerSpec:
    """Static stats for a tower archetype."""

    name: str
    cost: int
    damage: float
    range: float         # in grid cells (Euclidean)
    fire_rate: float     # shots per second
    projectile_speed: float  # cells per second; 0 = hitscan
    splash: float = 0.0  # splash radius (0 = single target)
    slow: float = 0.0    # multiplicative speed factor applied to target (0 = none)
    slow_duration: float = 0.0
    chain: int = 0       # extra targets the projectile bounces to
    color: Tuple[int, int, int] = (220, 220, 220)
    symbol: str = "T"


@dataclass(frozen=True)
class EnemySpec:
    """Static stats for an enemy archetype."""

    name: str
    hp: float
    speed: float         # cells per second
    bounty: int          # gold dropped on kill
    damage: float        # damage dealt to base on leak
    armor: float = 0.0   # flat damage reduction (never below 1)
    color: Tuple[int, int, int] = (180, 40, 40)
    radius: float = 0.28 # fraction of a cell used for rendering / collision


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping knobs. Defaults are sparse — tweak for your algorithm."""

    kill: float = 1.0
    leak: float = -10.0
    win: float = 50.0
    lose: float = -50.0
    step: float = 0.0        # per-step living cost/reward
    invalid_action: float = -0.1
    economy: float = 0.0     # optional: reward for holding gold (discourages hoarding when negative)
    scale: float = 0.1       # global multiplier applied to all non-step rewards


# ---- Canonical tower / enemy rosters -----------------------------------------

TOWERS: Dict[str, TowerSpec] = {
    "archer": TowerSpec(
        name="archer", cost=25, damage=8, range=3.2, fire_rate=1.4,
        projectile_speed=9.0, color=(90, 170, 90), symbol="A",
    ),
    "cannon": TowerSpec(
        name="cannon", cost=55, damage=28, range=2.6, fire_rate=0.55,
        projectile_speed=6.0, splash=1.1, color=(180, 110, 60), symbol="C",
    ),
    "ice": TowerSpec(
        name="ice", cost=35, damage=3, range=2.8, fire_rate=1.0,
        projectile_speed=8.0, slow=0.5, slow_duration=1.5,
        color=(110, 190, 230), symbol="I",
    ),
    "tesla": TowerSpec(
        name="tesla", cost=80, damage=14, range=3.0, fire_rate=0.9,
        projectile_speed=0.0, chain=2, color=(210, 190, 60), symbol="Z",
    ),
}

ENEMIES: Dict[str, EnemySpec] = {
    "scout": EnemySpec(
        name="scout", hp=28, speed=2.2, bounty=5, damage=1,
        color=(200, 90, 90),
    ),
    "grunt": EnemySpec(
        name="grunt", hp=60, speed=1.4, bounty=8, damage=1,
        color=(180, 60, 60),
    ),
    "brute": EnemySpec(
        name="brute", hp=180, speed=0.9, bounty=20, damage=3,
        armor=3.0, color=(110, 40, 40), radius=0.36,
    ),
    "swarm": EnemySpec(
        name="swarm", hp=14, speed=2.6, bounty=3, damage=1,
        color=(230, 140, 120), radius=0.22,
    ),
}


@dataclass
class GameConfig:
    """All the knobs for one run of the game. Hashable via to_json()."""

    width: int = 20
    height: int = 12
    starting_gold: int = 120
    starting_lives: int = 20
    max_waves: int = 20
    wave_build_seconds: float = 8.0   # player gets this much downtime between waves
    tick_rate: float = 30.0           # simulation Hz (used by the env wrapper)
    path: List[Tuple[int, int]] = field(default_factory=list)
    blocked_cells: List[Tuple[int, int]] = field(default_factory=list)
    towers: Dict[str, TowerSpec] = field(default_factory=lambda: dict(TOWERS))
    enemies: Dict[str, EnemySpec] = field(default_factory=lambda: dict(ENEMIES))
    rewards: RewardConfig = field(default_factory=RewardConfig)
    map_name: str = "s_curve"  # identifier for benchmarking

    # ---- map helpers -----------------------------------------------------

    def make_grid(self) -> List[List[int]]:
        """Build a fresh WxH grid from the configured path and blocked cells."""
        grid = [[TILE_GRASS for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.blocked_cells:
            grid[y][x] = TILE_BLOCKED
        if not self.path:
            # Default S-curve so a config without an explicit path still works.
            grid = _default_s_curve(grid, self.width, self.height)
        else:
            for x, y in self.path:
                grid[y][x] = TILE_PATH
            grid[self.path[0][1]][self.path[0][0]] = TILE_SPAWN
            grid[self.path[-1][1]][self.path[-1][0]] = TILE_BASE
        return grid

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def default_20x12(cls) -> "GameConfig":
        """The canonical starter map used by tournaments and the README."""
        cfg = cls()
        cfg.path = _s_curve_path(cfg.width, cfg.height)
        cfg.blocked_cells = [(1, 1), (10, 4), (17, 7)]
        cfg.map_name = "s_curve"
        return cfg

    @classmethod
    def straight_path(cls) -> "GameConfig":
        """Simple horizontal path — easy baseline for testing."""
        cfg = cls(width=20, height=12)
        cfg.path = _straight_path(cfg.width, cfg.height)
        cfg.blocked_cells = []
        cfg.map_name = "straight"
        return cfg

    @classmethod
    def zigzag_path(cls) -> "GameConfig":
        """Sharp zigzag path — tests spatial reasoning."""
        cfg = cls(width=20, height=12)
        cfg.path = _zigzag_path(cfg.width, cfg.height)
        cfg.blocked_cells = []
        cfg.map_name = "zigzag"
        return cfg

    @classmethod
    def chokepoint_path(cls) -> "GameConfig":
        """Narrow chokepoints with open areas — tests strategic placement."""
        cfg = cls(width=20, height=12)
        cfg.path = _chokepoint_path(cfg.width, cfg.height)
        cfg.blocked_cells = [(5, 3), (5, 4), (5, 5), (14, 6), (14, 7), (14, 8)]
        cfg.map_name = "chokepoint"
        return cfg

    @classmethod
    def spiral_path(cls) -> "GameConfig":
        """Long winding spiral path — tests long-term planning."""
        cfg = cls(width=20, height=12)
        cfg.path = _spiral_path(cfg.width, cfg.height)
        cfg.blocked_cells = []
        cfg.map_name = "spiral"
        return cfg

# ---- Default map geometry ----------------------------------------------------

def _s_curve_path(w: int, h: int) -> List[Tuple[int, int]]:
    """An S-shaped path that snakes through a w x h grid.

    Path enters at (0, 2), goes right, dips down, goes left, dips down,
    goes right, exits at (w-1, h-3). Deterministic and visually readable.
    """
    y_top = max(2, h // 6)
    y_mid = h // 2
    y_bot = h - 1 - y_top
    path: List[Tuple[int, int]] = []
    # Enter from the left at y_top. Top row goes all the way to x = w-3
    # (inclusive) so it shares a corner cell with the right-side vertical.
    for x in range(0, w - 2):
        path.append((x, y_top))
    # Down the right side (corner cell already covered by the top row).
    for y in range(y_top + 1, y_mid + 1):
        path.append((w - 3, y))
    # Left across the middle
    for x in range(w - 4, 2, -1):
        path.append((x, y_mid))
    # Down the left side
    for y in range(y_mid + 1, y_bot + 1):
        path.append((3, y))
    # Right to the exit
    for x in range(4, w):
        path.append((x, y_bot))
    return path


def _default_s_curve(grid: List[List[int]], w: int, h: int) -> List[List[int]]:
    path = _s_curve_path(w, h)
    for x, y in path:
        grid[y][x] = TILE_PATH
    grid[path[0][1]][path[0][0]] = TILE_SPAWN
    grid[path[-1][1]][path[-1][0]] = TILE_BASE
    return grid


def _straight_path(w: int, h: int) -> List[Tuple[int, int]]:
    """Simple horizontal path across the middle of the grid."""
    y = h // 2
    path = [(x, y) for x in range(0, w)]
    return path


def _zigzag_path(w: int, h: int) -> List[Tuple[int, int]]:
    """Sharp zigzag path with multiple vertical and horizontal segments."""
    path: List[Tuple[int, int]] = []
    y = 2
    direction = 1  # 1 = right, -1 = left

    for segment in range(4):
        # Horizontal segment
        if direction == 1:
            for x in range(2, w - 2):
                path.append((x, y))
        else:
            for x in range(w - 3, 1, -1):
                path.append((x, y))

        # Vertical segment
        if segment < 3:
            y_next = y + 3 if segment % 2 == 0 else y - 3
            y_next = max(2, min(h - 3, y_next))
            step = 1 if y_next > y else -1
            for y_step in range(y + step, y_next + step, step):
                x_pos = w - 3 if direction == 1 else 2
                path.append((x_pos, y_step))
            y = y_next
            direction *= -1

    return path


def _chokepoint_path(w: int, h: int) -> List[Tuple[int, int]]:
    """Path with narrow chokepoints and open areas."""
    path: List[Tuple[int, int]] = []

    # Start at left
    y = 2
    for x in range(0, 7):
        path.append((x, y))

    # Chokepoint 1: narrow vertical section
    for y_step in range(3, 6):
        path.append((6, y_step))

    # Open middle area
    y = 5
    for x in range(7, 13):
        path.append((x, y))

    # Chokepoint 2: narrow vertical section
    for y_step in range(6, 10):
        path.append((12, y_step))

    # Final stretch
    y = 9
    for x in range(13, w):
        path.append((x, y))

    return path


def _spiral_path(w: int, h: int) -> List[Tuple[int, int]]:
    """Long winding spiral path that spirals inward."""
    path: List[Tuple[int, int]] = []

    # Start from outside and spiral inward
    y_top = 1
    y_bot = h - 2
    x_left = 1
    x_right = w - 2

    while y_top <= y_bot and x_left <= x_right:
        # Top row: left to right
        for x in range(x_left, x_right + 1):
            path.append((x, y_top))
        y_top += 1

        if y_top > y_bot:
            break

        # Right column: top to bottom
        for y in range(y_top, y_bot + 1):
            path.append((x_right, y))
        x_right -= 1

        if x_left > x_right:
            break

        # Bottom row: right to left
        for x in range(x_right, x_left - 1, -1):
            path.append((x, y_bot))
        y_bot -= 1

        if y_top > y_bot:
            break

        # Left column: bottom to top
        for y in range(y_bot, y_top - 1, -1):
            path.append((x_left, y))
        x_left += 1

    return path
