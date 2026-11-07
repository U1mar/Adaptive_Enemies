"""Cost map builder for adaptive_enemies."""
from __future__ import annotations
from typing import Dict

from game.grid import Pos, Grid


def build_cost_map(grid: Grid, prob_map: Dict[Pos, float], alpha: float = 5.0) -> Dict[Pos, float]:
    raise NotImplementedError