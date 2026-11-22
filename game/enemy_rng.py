"""game/enemy_rng.py - per-enemy random generators (task 2.1).

The seed string is built from (level seed, round, role, index), so every enemy gets its own stream
that does not depend on creation order or on the global `random` module.
"""
from __future__ import annotations
import random


def enemy_rng(level_seed: int, round_no: int, role: str, index: int = 0) -> random.Random:
    # str seeds are hashed with SHA-512 by random.seed(): stable across runs and Python versions.
    return random.Random(f"adaptive-enemies:{level_seed}:{round_no}:{role}:{index}")
