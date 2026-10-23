"""Tracker for adaptive_enemies."""
from __future__ import annotations
from typing import Dict

from game.grid import Pos


class Tracker:
    def __init__(self, decay: float = 0.95, history_len: int = 50):
        raise NotImplementedError

    def record(self, pos: Pos) -> None:
        raise NotImplementedError

    def probability_map(self) -> Dict[Pos, float]:
        raise NotImplementedError