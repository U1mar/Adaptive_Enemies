"""Tracker for adaptive_enemies."""
from __future__ import annotations
from collections import deque
from typing import Dict

from adaptive_enemies.game.grid import Pos


class Tracker:
    def __init__(self, decay: float = 0.95, history_len: int = 50):
        self.decay = decay
        self.history_len = history_len
        self.counts: Dict[Pos, float] = {}
        self.recent: deque[Pos] = deque(maxlen=history_len)

    def record(self, pos: Pos) -> None:
        # Decay all existing counts
        for k in self.counts:
            self.counts[k] *= self.decay
        # Add 1 to the new position
        self.counts[pos] = self.counts.get(pos, 0.0) + 1.0
        # Track recent positions
        self.recent.append(pos)

    def probability_map(self) -> Dict[Pos, float]:
        if not self.counts:
            return {}
        total = sum(self.counts.values())
        return {pos: count / total for pos, count in self.counts.items()}