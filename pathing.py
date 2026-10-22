"""Pathfinding algorithms: BFS, Dijkstra, A*."""
from __future__ import annotations
import heapq
from collections import deque
from typing import Dict, List, Tuple, Optional, Set, Callable

Position = Tuple[int, int]
CostMap = Dict[Position, float]

# Import Grid for type hints
from grid import Grid


def _reconstruct_path(came_from: Dict[Position, Position], start: Position, goal: Position) -> List[Position]:
    """Reconstruct path from came_from dict."""
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


def bfs(grid: Grid, start: Position, goal: Position, cost_map: CostMap = None) -> Tuple[List[Position], int]:
    """Breadth-first search. Returns (path, explored_count)."""
    if start == goal:
        return [start], 1

    queue = deque([start])
    came_from = {start: None}
    explored = 0

    while queue:
        current = queue.popleft()
        explored += 1

        if current == goal:
            return _reconstruct_path(came_from, start, goal), explored

        for neighbor in grid.neighbors(current):
            if neighbor not in came_from:
                came_from[neighbor] = current
                queue.append(neighbor)

    return [], explored


def dijkstra(grid: Grid, start: Position, goal: Position, cost_map: CostMap = None) -> Tuple[List[Position], int]:
    """Dijkstra's algorithm. Returns (path, explored_count)."""
    if start == goal:
        return [start], 1

    cost_map = cost_map or {}
    pq = [(0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}
    explored = 0

    while pq:
        current_cost, current = heapq.heappop(pq)
        explored += 1

        if current == goal:
            return _reconstruct_path(came_from, start, goal), explored

        if current_cost > cost_so_far.get(current, float('inf')):
            continue

        for neighbor in grid.neighbors(current):
            extra_cost = cost_map.get(neighbor, 0.0)
            new_cost = current_cost + 1.0 + extra_cost
            if new_cost < cost_so_far.get(neighbor, float('inf')):
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))

    return [], explored


def heuristic(a: Position, b: Position) -> float:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: Grid, start: Position, goal: Position, cost_map: CostMap = None) -> Tuple[List[Position], int]:
    """A* search with Manhattan heuristic. Returns (path, explored_count)."""
    if start == goal:
        return [start], 1

    cost_map = cost_map or {}
    pq = [(heuristic(start, goal), 0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}
    explored = 0

    while pq:
        _, current_cost, current = heapq.heappop(pq)
        explored += 1

        if current == goal:
            return _reconstruct_path(came_from, start, goal), explored

        if current_cost > cost_so_far.get(current, float('inf')):
            continue

        for neighbor in grid.neighbors(current):
            extra_cost = cost_map.get(neighbor, 0.0)
            new_cost = current_cost + 1.0 + extra_cost
            if new_cost < cost_so_far.get(neighbor, float('inf')):
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                came_from[neighbor] = current
                heapq.heappush(pq, (priority, new_cost, neighbor))

    return [], explored