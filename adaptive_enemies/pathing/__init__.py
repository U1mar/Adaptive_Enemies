"""Pathing algorithms for adaptive_enemies."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

from adaptive_enemies.game.grid import Pos, Grid


@dataclass
class PathResult:
    path: Optional[List[Pos]]
    explored: int
    cost: float


CostMap = Dict[Pos, float]

from .bfs import bfs
from .dijkstra import dijkstra
from .astar import astar

__all__ = ['PathResult', 'CostMap', 'bfs', 'dijkstra', 'astar']