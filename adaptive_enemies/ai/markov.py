"""ZoneModel (Markov chain) for adaptive_enemies."""
from __future__ import annotations
from collections import defaultdict
from typing import Dict


class ZoneModel:
    def __init__(self):
        self.zones = ["west", "middle", "east"]
        # transition_counts[prev_zone][next_zone] = count
        self.transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def observe(self, prev_zone: str, next_zone: str) -> None:
        self.transition_counts[prev_zone][next_zone] += 1

    def predict_next(self, zone: str) -> Dict[str, float]:
        # Laplace smoothing: add 1 to each possible next zone
        counts = self.transition_counts[zone]
        total = sum(counts.values()) + len(self.zones)  # +1 for each zone
        return {z: (counts.get(z, 0) + 1) / total for z in self.zones}