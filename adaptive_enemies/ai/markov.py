"""ZoneModel (Markov chain) for adaptive_enemies."""
from __future__ import annotations
from typing import Dict


class ZoneModel:
    def observe(self, prev_zone: str, next_zone: str) -> None:
        raise NotImplementedError

    def predict_next(self, zone: str) -> Dict[str, float]:
        raise NotImplementedError