"""Game engine for adaptive_enemies."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable

from game.grid import Pos, Grid
from game.enemy import Enemy
from ai.tracker import Tracker
from ai.markov import ZoneModel


@dataclass
class RoundResult:
    caught: bool
    got_key: bool
    escaped: bool
    turns: int
    path: List[Pos]


class Game:
    def __init__(self, grid: Grid, enemies: List[Enemy], tracker: Tracker, model: ZoneModel):
        raise NotImplementedError

    def play_round(self, player_policy: Callable, max_turns: int = 200) -> RoundResult:
        raise NotImplementedError