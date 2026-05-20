"""Pathfinding helpers.

The engine uses a fixed waypoint path (fastest, deterministic) but we expose
A* so maps can be generated / validated at config time. If the path is broken
(no route from spawn to base) the config is rejected before training starts.
"""

from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

DIRECTIONS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def a_star(
    grid: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    passable_codes: Tuple[int, ...] = (0, 1, 2, 3),
) -> Optional[List[Tuple[int, int]]]:
    """Grid A* over integer tile codes.

    `passable_codes` is the set of tile values the path may traverse. For our
    default map this includes GRASS, PATH, SPAWN, and BASE. Returns None if
    no path exists.
    """
    if not grid:
        return None
    h = len(grid)
    w = len(grid[0])
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if grid[sy][sx] not in passable_codes or grid[gy][gx] not in passable_codes:
        return None

    def h_(x: int, y: int) -> int:
        return abs(x - gx) + abs(y - gy)

    open_heap: List[Tuple[int, int, int, int]] = [(h_(sx, sy), 0, sx, sy)]
    came: dict = {}
    g_score = {(sx, sy): 0}
    closed = set()

    while open_heap:
        _, g, x, y = heapq.heappop(open_heap)
        if (x, y) == (gx, gy):
            # Reconstruct.
            path = [(x, y)]
            while (x, y) in came:
                x, y = came[(x, y)]
                path.append((x, y))
            path.reverse()
            return path
        if (x, y) in closed:
            continue
        closed.add((x, y))
        for dx, dy in DIRECTIONS_4:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if grid[ny][nx] not in passable_codes:
                continue
            if (nx, ny) in closed:
                continue
            tentative = g + 1
            if tentative < g_score.get((nx, ny), tentative + 1):
                g_score[(nx, ny)] = tentative
                came[(nx, ny)] = (x, y)
                heapq.heappush(open_heap, (tentative + h_(nx, ny), tentative, nx, ny))
    return None


def path_is_clear(grid: List[List[int]], spawn: Tuple[int, int], base: Tuple[int, int]) -> bool:
    """True iff enemies could walk from spawn to base on the current grid."""
    return a_star(grid, spawn, base) is not None


def distance_to_path_cells(
    width: int, height: int, path: List[Tuple[int, int]]
) -> List[List[float]]:
    """Precompute Manhattan-ish distance from every cell to the nearest path cell.

    Used as an observation channel — it tells the agent where the action is.
    """
    import math

    INF = float("inf")
    dist = [[INF] * width for _ in range(height)]
    # Multi-source BFS on integer grid, then sqrt only at the end (not really —
    # we store Euclidean min distance, so we do a proper O(WH) BFS on squared
    # distance candidates).
    # Simplest correct approach: for each cell compute min Euclidean to any path cell.
    for y in range(height):
        for x in range(width):
            best = INF
            for px, py in path:
                d = math.hypot(px - x, py - y)
                if d < best:
                    best = d
            dist[y][x] = best
    return dist
