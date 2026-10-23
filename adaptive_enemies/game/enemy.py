"""Enemy classes for adaptive_enemies."""
from __future__ import annotations
from typing import List, Dict

from game.grid import Pos, Grid


class Enemy:
    def __init__(self, pos: Pos, role: str, patrol_route: List[Pos]):
        raise NotImplementedError

    def step(self, grid: Grid, player_pos: Pos, cost_map: Dict[Pos, float]) -> Pos:
        raise NotImplementedError

    def sees(self, grid: Grid, player_pos: Pos, radius: int = 5) -> bool:
        raise NotImplementedError