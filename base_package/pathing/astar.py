"""A* pathfinding with Manhattan distance heuristic."""
from __future__ import annotations
import heapq
from typing import Optional

from . import PathResult, Pos, Grid, CostMap


def manhattan(a: Pos, b: Pos) -> float:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: Grid, start: Pos, goal: Pos, cost_map: CostMap = None) -> PathResult:
    """A* finds shortest path with heuristic guidance."""
    if cost_map is None:
        cost_map = {}

    if not grid.passable(start) or not grid.passable(goal):
        return PathResult(path=None, explored=0, cost=float('inf'))

    # Priority queue: (f_score, g_score, position)
    pq: list[tuple[float, float, Pos]] = [(manhattan(start, goal), 0.0, start)]
    # Best known g_score (cost from start)
    g_score: dict[Pos, float] = {start: 0.0}
    parent: dict[Pos, Optional[Pos]] = {start: None}
    explored = 0

    while pq:
        f_score, current_g, current = heapq.heappop(pq)
        explored += 1

        # Skip if we found a better path already
        if current_g > g_score.get(current, float('inf')):
            continue

        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return PathResult(path=path, explored=explored, cost=current_g)

        for neighbor in grid.neighbors(current):
            step_cost = 1.0 + cost_map.get(neighbor, 0.0)
            tentative_g = current_g + step_cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                parent[neighbor] = current
                f = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(pq, (f, tentative_g, neighbor))

    return PathResult(path=None, explored=explored, cost=float('inf'))