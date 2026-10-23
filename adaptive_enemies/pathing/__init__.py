"""Pathing algorithms for adaptive_enemies."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

from game.grid import Pos, Grid


@dataclass
class PathResult:
    path: Optional[List[Pos]]
    explored: int
    cost: float


CostMap = Dict[Pos, float]


def bfs(grid: Grid, start: Pos, goal: Pos) -> PathResult:
    raise NotImplementedError


def dijkstra(grid: Grid, start: Pos, goal: Pos, cost_map: CostMap = None) -> PathResult:
    raise NotImplementedError


def astar(grid: Grid, start: Pos, goal: Pos, cost_map: CostMap = None) -> PathResult:
    raise NotImplementedError