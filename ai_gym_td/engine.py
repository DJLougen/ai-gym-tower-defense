"""Core game engine — deterministic, headless, pure-Python + NumPy.

The engine is a single-step state machine: call `step(dt)` to advance the
simulation by `dt` seconds. It does not know about rendering, RL, or input.
Everything it produces is plain Python/NumPy so it can be serialized,
replayed, or wrapped by any training framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ai_gym_td.config import (
    EnemySpec,
    GameConfig,
    TowerSpec,
    TILE_BASE,
    TILE_BLOCKED,
    TILE_GRASS,
    TILE_PATH,
    TILE_SPAWN,
)
from ai_gym_td.pathfinding import a_star


# ---- Entity payloads (flat dataclasses — no inheritance, no virtuals) --------

@dataclass
class Enemy:
    eid: int
    spec: EnemySpec
    path_index: int            # index of the LAST waypoint fully passed
    progress: float            # fraction along segment [path_index, path_index+1]
    x: float
    y: float
    hp: float
    max_hp: float
    slow_mult: float = 1.0     # 1.0 = normal, <1.0 = slowed
    slow_until: float = 0.0    # simulation time when slow wears off
    dead: bool = False
    leaked: bool = False


@dataclass
class Tower:
    tid: int
    spec: TowerSpec
    x: int                     # grid col
    y: int                     # grid row
    cooldown: float = 0.0      # seconds until next shot
    target_eid: Optional[int] = None
    shots_fired: int = 0
    total_damage: float = 0.0


@dataclass
class Projectile:
    pid: int
    kind: str                  # tower spec name (for splash/slow/chain lookups)
    x: float
    y: float
    target_eid: int
    speed: float               # 0 = hitscan, already resolved at fire time
    damage: float
    splash: float
    slow: float
    slow_duration: float
    chain: int
    already_hit: List[int] = field(default_factory=list)
    dead: bool = False


# ---- Event log entries (used for reward shaping and replay) ------------------

@dataclass
class KillEvent:
    eid: int
    enemy: str
    bounty: int
    x: float
    y: float


@dataclass
class LeakEvent:
    eid: int
    enemy: str
    damage: float


@dataclass
class BuildEvent:
    tid: int
    tower: str
    x: int
    y: int
    cost: int


# ---- Game --------------------------------------------------------------------


class TowerDefenseGame:
    """The simulation. No I/O, no rendering, fully deterministic given a seed."""

    # Phase constants
    BUILD = "build"
    WAVE = "wave"
    WON = "won"
    LOST = "lost"

    def __init__(self, config: Optional[GameConfig] = None, seed: Optional[int] = None):
        self.cfg = config or GameConfig.default_20x12()
        self.grid: List[List[int]] = self.cfg.make_grid()
        self.seed = seed if seed is not None else 0
        self._rng_seed = self.seed

        # Locate spawn/base from the grid.
        self.spawn: Optional[Tuple[int, int]] = None
        self.base: Optional[Tuple[int, int]] = None
        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                if self.grid[y][x] == TILE_SPAWN:
                    self.spawn = (x, y)
                elif self.grid[y][x] == TILE_BASE:
                    self.base = (x, y)
        if self.spawn is None or self.base is None:
            raise ValueError("Map is missing SPAWN or BASE tile.")

        # Reconstruct path by walking from spawn to base over PATH tiles.
        self.path: List[Tuple[int, int]] = self._extract_path()

        # Runtime state
        self.reset()

    # ---- public state accessors -----------------------------------------

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def gold(self) -> int:
        return self._gold

    @property
    def lives(self) -> float:
        return self._lives

    @property
    def wave_number(self) -> int:
        return self._wave_number

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def towers(self) -> Dict[int, Tower]:
        return self._towers

    @property
    def enemies(self) -> Dict[int, Enemy]:
        return self._enemies

    @property
    def projectiles(self) -> Dict[int, Projectile]:
        return self._projectiles

    @property
    def is_terminal(self) -> bool:
        return self._phase in (self.WON, self.LOST)

    @property
    def tower_grid(self) -> List[List[Optional[str]]]:
        """For rendering / observation: tower name (or None) at each cell."""
        g: List[List[Optional[str]]] = [[None] * self.cfg.width for _ in range(self.cfg.height)]
        for t in self._towers.values():
            g[t.y][t.x] = t.spec.name
        return g

    # ---- lifecycle ------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
            self._rng_seed = seed
        self._gold = self.cfg.starting_gold
        self._lives = float(self.cfg.starting_lives)
        self._wave_number = 0
        self._tick = 0
        self._sim_time = 0.0
        self._phase = self.BUILD
        self._build_timer = 0.0
        self._towers = {}
        self._enemies = {}
        self._projectiles = {}
        self._next_eid = 1
        self._next_tid = 1
        self._next_pid = 1
        self._spawn_queue: List[Tuple[float, str]] = []  # (time_offset, enemy_name)
        self._wave_elapsed = 0.0
        self.events_this_step: List = []
        self.invalid_action_this_step = False

    # ---- player actions -------------------------------------------------

    def build_tower(self, x: int, y: int, tower_name: str) -> Optional[int]:
        """Place a tower. Returns tower id on success, None on failure."""
        if self._phase in (self.WON, self.LOST):
            return None
        if not (0 <= x < self.cfg.width and 0 <= y < self.cfg.height):
            return None
        if self.grid[y][x] not in (TILE_GRASS,):
            return None
        if self._tower_at(x, y) is not None:
            return None
        spec = self.cfg.towers.get(tower_name)
        if spec is None:
            return None
        if self._gold < spec.cost:
            return None
        # Safety: ensure the tower does not block the enemy path. Our grid
        # treats grass as buildable but not walkable, so towers never block
        # the path — but guard against misconfigured maps.
        self._gold -= spec.cost
        tid = self._next_tid
        self._next_tid += 1
        tower = Tower(tid=tid, spec=spec, x=x, y=y)
        self._towers[tid] = tower
        self.events_this_step.append(BuildEvent(tid, tower_name, x, y, spec.cost))
        return tid

    def sell_tower(self, tid: int, refund_frac: float = 0.6) -> bool:
        t = self._towers.get(tid)
        if t is None:
            return False
        refund = int(t.spec.cost * refund_frac)
        self._gold += refund
        del self._towers[tid]
        return True

    def start_next_wave(self) -> bool:
        """Manually trigger the next wave. Used by the env to skip build time."""
        if self._phase != self.BUILD:
            return False
        if self._wave_number >= self.cfg.max_waves:
            return False
        self._begin_wave(self._wave_number + 1)
        return True

    # ---- simulation -----------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance the simulation by dt seconds. Collects events in
        `self.events_this_step` (reset at the start of each call).
        """
        self.events_this_step = []
        self.invalid_action_this_step = False
        if self.is_terminal:
            return
        self._tick += 1
        self._sim_time += dt

        if self._phase == self.BUILD:
            self._build_timer += dt
            if self._build_timer >= self.cfg.wave_build_seconds:
                if self._wave_number >= self.cfg.max_waves:
                    self._phase = self.WON
                    return
                self._begin_wave(self._wave_number + 1)
            return

        # WAVE phase
        self._wave_elapsed += dt
        self._spawn_enemies(dt)
        self._update_enemies(dt)
        self._update_towers(dt)
        self._update_projectiles(dt)
        self._cleanup()
        self._check_wave_complete()
        if self._lives <= 0:
            self._phase = self.LOST

    # ---- internals ------------------------------------------------------

    def _tower_at(self, x: int, y: int) -> Optional[int]:
        for tid, t in self._towers.items():
            if t.x == x and t.y == y:
                return tid
        return None

    def _extract_path(self) -> List[Tuple[int, int]]:
        """Use A* over the PATH tiles to produce an ordered spawn->base path."""
        p = a_star(self.grid, self.spawn, self.base, passable_codes=(TILE_PATH, TILE_SPAWN, TILE_BASE))
        if p is None:
            raise ValueError("No walkable path from spawn to base.")
        return p

    def _begin_wave(self, wave_number: int) -> None:
        self._wave_number = wave_number
        self._phase = self.WAVE
        self._wave_elapsed = 0.0
        self._spawn_queue = list(self._wave_roster(wave_number))
        # Convert offsets to absolute sim times.
        base = self._sim_time
        self._spawn_queue = [(base + offset, name) for offset, name in self._spawn_queue]

    def _wave_roster(self, wave_number: int) -> List[Tuple[float, str]]:
        """Generate (time_offset, enemy_name) pairs for a wave.

        Difficulty curve: scouts early, grunts mid, brutes late, swarms on
        multiples of 3, a boss brute every 5 waves.
        """
        import random

        rng = random.Random(self._rng_seed + wave_number)
        n = wave_number
        roster: List[Tuple[float, str]] = []
        # Scouts: present every wave, scaling count.
        scout_count = 3 + n
        for i in range(scout_count):
            roster.append((i * 0.7, "scout"))
        # Grunts: from wave 2 onward.
        if n >= 2:
            grunt_count = max(1, n - 1)
            for i in range(grunt_count):
                roster.append((scout_count * 0.7 + i * 0.9, "grunt"))
        # Swarms: every 3rd wave.
        if n % 3 == 0:
            swarm_count = 8 + n
            base_time = len(roster) * 0.3
            for i in range(swarm_count):
                roster.append((base_time + i * 0.25, "swarm"))
        # Brutes: from wave 4 onward.
        if n >= 4:
            brute_count = max(1, n // 3)
            base_time = max(t for t, _ in roster) + 1.0 if roster else 0.0
            for i in range(brute_count):
                roster.append((base_time + i * 1.4, "brute"))
        # Boss: every 5th wave — one extra-brick brute.
        if n % 5 == 0:
            roster.append((max(t for t, _ in roster) + 2.0 if roster else 0.0, "brute"))
        rng.shuffle(roster)  # mild variety while preserving composition
        return roster

    def _spawn_enemies(self, dt: float) -> None:
        now = self._sim_time
        while self._spawn_queue and self._spawn_queue[0][0] <= now:
            _, name = self._spawn_queue.pop(0)
            spec = self.cfg.enemies[name]
            sx, sy = self.spawn
            # Scale enemy HP slightly with wave number for long games.
            hp_scale = 1.0 + 0.05 * max(0, self._wave_number - 1)
            hp = spec.hp * hp_scale
            eid = self._next_eid
            self._next_eid += 1
            self._enemies[eid] = Enemy(
                eid=eid,
                spec=spec,
                path_index=0,
                progress=0.0,
                x=float(sx),
                y=float(sy),
                hp=hp,
                max_hp=hp,
            )

    def _update_enemies(self, dt: float) -> None:
        for e in list(self._enemies.values()):
            if e.dead or e.leaked:
                continue
            # Refresh slow.
            if e.slow_until < self._sim_time:
                e.slow_mult = 1.0
            speed = e.spec.speed * e.slow_mult
            remaining = speed * dt
            while remaining > 0.0 and e.path_index < len(self.path) - 1:
                ax, ay = self.path[e.path_index]
                bx, by = self.path[e.path_index + 1]
                seg_len = math.hypot(bx - ax, by - ay) or 1.0
                dist_to_end = seg_len * (1.0 - e.progress)
                if remaining >= dist_to_end:
                    remaining -= dist_to_end
                    e.path_index += 1
                    e.progress = 0.0
                    e.x, e.y = float(self.path[e.path_index][0]), float(self.path[e.path_index][1])
                else:
                    e.progress += remaining / seg_len
                    e.x = ax + (bx - ax) * e.progress
                    e.y = ay + (by - ay) * e.progress
                    remaining = 0.0
            if e.path_index >= len(self.path) - 1:
                # Reached the base.
                e.leaked = True
                self._lives -= e.spec.damage
                self.events_this_step.append(LeakEvent(e.eid, e.spec.name, e.spec.damage))

    def _update_towers(self, dt: float) -> None:
        for t in self._towers.values():
            t.cooldown = max(0.0, t.cooldown - dt)
            target = self._acquire_target(t)
            t.target_eid = target.eid if target is not None else None
            if target is None or t.cooldown > 0.0:
                continue
            t.cooldown = 1.0 / t.spec.fire_rate
            t.shots_fired += 1
            self._fire(t, target)

    def _acquire_target(self, tower: Tower) -> Optional[Enemy]:
        """First policy: enemy furthest along the path that is in range."""
        best: Optional[Enemy] = None
        best_progress = -1.0
        tx, ty = tower.x + 0.5, tower.y + 0.5
        r2 = tower.spec.range * tower.spec.range
        for e in self._enemies.values():
            if e.dead or e.leaked:
                continue
            dx = (e.x + 0.5) - tx
            dy = (e.y + 0.5) - ty
            if dx * dx + dy * dy > r2:
                continue
            prog = e.path_index + e.progress
            if prog > best_progress:
                best_progress = prog
                best = e
        return best

    def _fire(self, tower: Tower, target: Enemy) -> None:
        spec = tower.spec
        if spec.projectile_speed <= 0.0:
            # Hitscan: resolve damage immediately (used by tesla / lightning).
            self._apply_hit(tower, target, chain_left=spec.chain, already_hit=[])
            return
        pid = self._next_pid
        self._next_pid += 1
        self._projectiles[pid] = Projectile(
            pid=pid,
            kind=spec.name,
            x=tower.x + 0.5,
            y=tower.y + 0.5,
            target_eid=target.eid,
            speed=spec.projectile_speed,
            damage=spec.damage,
            splash=spec.splash,
            slow=spec.slow,
            slow_duration=spec.slow_duration,
            chain=spec.chain,
        )

    def _update_projectiles(self, dt: float) -> None:
        for p in list(self._projectiles.values()):
            if p.dead:
                continue
            target = self._enemies.get(p.target_eid)
            if target is None or target.dead or target.leaked:
                # Lost its target; just disappear.
                p.dead = True
                continue
            tx, ty = target.x + 0.5, target.y + 0.5
            dx, dy = tx - p.x, ty - p.y
            dist = math.hypot(dx, dy)
            travel = p.speed * dt
            if travel >= dist:
                # Impact.
                self._apply_impact(p, target)
                p.dead = True
            else:
                p.x += dx * (travel / dist)
                p.y += dy * (travel / dist)

    def _apply_impact(self, proj: Projectile, target: Enemy) -> None:
        # Direct damage on the target.
        dmg = max(1.0, proj.damage - target.spec.armor)
        target.hp -= dmg
        if proj.slow > 0.0:
            target.slow_mult = min(target.slow_mult, proj.slow)
            target.slow_until = max(target.slow_until, self._sim_time + proj.slow_duration)
        # Splash.
        if proj.splash > 0.0:
            r2 = proj.splash * proj.splash
            for e in self._enemies.values():
                if e is target or e.dead or e.leaked:
                    continue
                dx, dy = (e.x + 0.5) - (target.x + 0.5), (e.y + 0.5) - (target.y + 0.5)
                if dx * dx + dy * dy <= r2:
                    e.hp -= max(1.0, proj.damage * 0.6 - e.spec.armor)
        # Chain bounces (lightning-style).
        if proj.chain > 0:
            already = proj.already_hit + [target.eid]
            # Find nearest other enemy to bounce to.
            bounce = self._nearest_enemy(target.x, target.y, exclude=already, max_range=proj.splash + 2.0)
            if bounce is not None:
                # Create a mini "child" projectile that resolves instantly.
                child = Projectile(
                    pid=self._next_pid, kind=proj.kind,
                    x=target.x + 0.5, y=target.y + 0.5,
                    target_eid=bounce.eid, speed=0.0,
                    damage=proj.damage * 0.7, splash=0.0,
                    slow=proj.slow, slow_duration=proj.slow_duration,
                    chain=proj.chain - 1, already_hit=already,
                )
                self._next_pid += 1
                self._apply_hit_for_child(child, bounce)

    def _apply_hit_for_child(self, proj: Projectile, target: Enemy) -> None:
        """Resolve a zero-travel chain bounce immediately."""
        dmg = max(1.0, proj.damage - target.spec.armor)
        target.hp -= dmg
        if proj.slow > 0.0:
            target.slow_mult = min(target.slow_mult, proj.slow)
            target.slow_until = max(target.slow_until, self._sim_time + proj.slow_duration)
        if proj.chain > 0:
            already = proj.already_hit + [target.eid]
            bounce = self._nearest_enemy(target.x, target.y, exclude=already, max_range=3.0)
            if bounce is not None:
                child = Projectile(
                    pid=self._next_pid, kind=proj.kind,
                    x=target.x + 0.5, y=target.y + 0.5,
                    target_eid=bounce.eid, speed=0.0,
                    damage=proj.damage * 0.7, splash=0.0,
                    slow=proj.slow, slow_duration=proj.slow_duration,
                    chain=proj.chain - 1, already_hit=already,
                )
                self._next_pid += 1
                self._apply_hit_for_child(child, bounce)

    def _apply_hit(self, tower: Tower, target: Enemy, chain_left: int, already_hit: List[int]) -> None:
        """Hitscan (tesla) damage resolution with chain bounces."""
        spec = tower.spec
        dmg = max(1.0, spec.damage - target.spec.armor)
        target.hp -= dmg
        tower.total_damage += dmg
        if spec.slow > 0.0:
            target.slow_mult = min(target.slow_mult, spec.slow)
            target.slow_until = max(target.slow_until, self._sim_time + spec.slow_duration)
        if chain_left > 0:
            already = already_hit + [target.eid]
            bounce = self._nearest_enemy(target.x, target.y, exclude=already, max_range=spec.range)
            if bounce is not None:
                # Synthesize a virtual tower for the child hit so stats stay consistent.
                self._apply_hit(tower, bounce, chain_left - 1, already)

    def _nearest_enemy(
        self, x: float, y: float, exclude: List[int], max_range: float
    ) -> Optional[Enemy]:
        best: Optional[Enemy] = None
        best_d2 = max_range * max_range
        for e in self._enemies.values():
            if e.dead or e.leaked or e.eid in exclude:
                continue
            dx, dy = (e.x + 0.5) - (x + 0.5), (e.y + 0.5) - (y + 0.5)
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = e
        return best

    def _cleanup(self) -> None:
        for eid in list(self._enemies.keys()):
            e = self._enemies[eid]
            if e.hp <= 0 and not e.dead and not e.leaked:
                e.dead = True
                self._gold += e.spec.bounty
                self.events_this_step.append(KillEvent(e.eid, e.spec.name, e.spec.bounty, e.x, e.y))
        # Prune dead / leaked enemies and dead projectiles.
        self._enemies = {k: v for k, v in self._enemies.items() if not (v.dead or v.leaked)}
        self._projectiles = {k: v for k, v in self._projectiles.items() if not v.dead}

    def _check_wave_complete(self) -> None:
        if self._phase != self.WAVE:
            return
        if self._spawn_queue:
            return
        if self._enemies:
            return
        # Wave cleared.
        if self._wave_number >= self.cfg.max_waves:
            self._phase = self.WON
        else:
            self._phase = self.BUILD
            self._build_timer = 0.0
            # Small wave-clear bonus.
            self._gold += 20 + 5 * self._wave_number

    # ---- introspection --------------------------------------------------

    def active_enemy_count(self) -> int:
        return len(self._enemies)

    def tower_count(self) -> int:
        return len(self._towers)

    def snapshot(self) -> Dict:
        """Plain-dict snapshot for replay / logging."""
        return {
            "tick": self._tick,
            "sim_time": self._sim_time,
            "phase": self._phase,
            "gold": self._gold,
            "lives": self._lives,
            "wave": self._wave_number,
            "enemies": [
                {"eid": e.eid, "name": e.spec.name, "x": e.x, "y": e.y,
                 "hp": e.hp, "max_hp": e.max_hp, "slow": e.slow_mult}
                for e in self._enemies.values()
            ],
            "towers": [
                {"tid": t.tid, "name": t.spec.name, "x": t.x, "y": t.y,
                 "cooldown": t.cooldown, "target": t.target_eid,
                 "shots": t.shots_fired, "dmg": t.total_damage}
                for t in self._towers.values()
            ],
        }


# Re-export event dataclasses at module level so callers can do `from ai_gym_td.engine import KillEvent`.
__all__ = [
    "TowerDefenseGame",
    "Enemy", "Tower", "Projectile",
    "KillEvent", "LeakEvent", "BuildEvent",
]
