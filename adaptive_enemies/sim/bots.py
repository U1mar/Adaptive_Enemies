"""Bot policies for adaptive_enemies."""
from __future__ import annotations
from typing import Any

from game.grid import Pos


def random_bot(state: Any) -> Pos:
    raise NotImplementedError


def fixed_route_bot(state: Any) -> Pos:
    raise NotImplementedError


def switching_bot(state: Any) -> Pos:
    raise NotImplementedError