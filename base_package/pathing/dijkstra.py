"""Dijkstra's algorithm pathfinding."""
from __future__ import annotations
import heapq
from typing import Optional

from . import PathResult, Pos, Grid, CostMap


def dijkstra(grid: Grid, start: Pos, goal: Pos, cost_map: CostMap = None) -> PathResult:
    """Dijkstra finds shortest path with optional cost map."""
    if cost_map is None:
        cost_map = {}

    if not grid.passable(start) or not grid.passable(goal):
        return PathResult(path=None, explored=0, cost=float('inf'))

    # Priority queue: (cost, position)
    pq: list[tuple[float, Pos]] = [(0.0, start)]
    # Best known cost to reach each position
    dist: dict[Pos, float] = {start: 0.0}
    parent: dict[Pos, Optional[Pos]] = {start: None}
    explored = 0

    while pq:
        current_cost, current = heapq.heappop(pq)
        explored += 1

        # Skip if we found a better path already
        if current_cost > dist.get(current, float('inf')):
            continue

        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return PathResult(path=path, explored=explored, cost=current_cost)

        for neighbor in grid.neighbors(current):
            step_cost = 1.0 + cost_map.get(neighbor, 0.0)
            new_cost = current_cost + step_cost

            if new_cost < dist.get(neighbor, float('inf')):
                dist[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))

    return PathResult(path=None, explored=explored, cost=float('inf'))