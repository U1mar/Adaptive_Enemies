"""Breadth-First Search pathfinding."""
from __future__ import annotations
from collections import deque
from typing import Optional

from . import PathResult, Pos, Grid, CostMap


def bfs(grid: Grid, start: Pos, goal: Pos) -> PathResult:
    """BFS finds shortest path in unweighted grid."""
    if not grid.passable(start) or not grid.passable(goal):
        return PathResult(path=None, explored=0, cost=float('inf'))

    queue = deque([start])
    visited = {start}
    parent = {start: None}
    explored = 0

    while queue:
        current = queue.popleft()
        explored += 1

        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return PathResult(path=path, explored=explored, cost=len(path) - 1 if path else 0)

        for neighbor in grid.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    return PathResult(path=None, explored=explored, cost=float('inf'))