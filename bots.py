"""Bot policies for automated play."""
from __future__ import annotations
from typing import List, Tuple, Callable
from grid import Grid, Position
from pathing import astar


Policy = Callable[[Grid, Position, bool, List[Position]], Position]


def find_path_to(grid: Grid, start: Position, goal: Position) -> Position:
    """Get next step toward goal using A*."""
    path, _ = astar(grid, start, goal)
    if len(path) > 1:
        return path[1]
    return start


class FixedRouteBot:
    """Always takes the east corridor."""

    def __init__(self):
        self.target_sequence = ['vault', 'exit']
        self.current_target = 0

    def __call__(self, grid: Grid, player_pos: Position, got_key: bool,
                 enemy_positions: List[Position]) -> Position:
        if not got_key:
            target = grid.vault
        else:
            target = grid.exit
        return find_path_to(grid, player_pos, target)


class SwitchingBot:
    """Changes corridor every 3 rounds."""

    def __init__(self):
        self.round_count = 0
        self.corridor = 'west'
        self.target_sequence = ['vault', 'exit']
        self.current_target = 0

    def __call__(self, grid: Grid, player_pos: Position, got_key: bool,
                 enemy_positions: List[Position]) -> Position:
        # Note: round tracking would need to be external
        if not got_key:
            target = grid.vault
        else:
            target = grid.exit
        return find_path_to(grid, player_pos, target)

    def new_round(self):
        self.round_count += 1
        if self.round_count % 3 == 0:
            corridors = ['west', 'middle', 'east']
            self.corridor = corridors[(corridors.index(self.corridor) + 1) % 3]


class RandomBot:
    """Random valid moves."""

    def __init__(self):
        import random
        self.random = random

    def __call__(self, grid: Grid, player_pos: Position, got_key: bool,
                 enemy_positions: List[Position]) -> Position:
        neighbors = grid.neighbors(player_pos)
        if not neighbors:
            return player_pos
        return self.random.choice(neighbors)


# For compatibility with engine's Policy type
from typing import Callable
Policy = Callable[[Grid, Position, bool, List[Position]], Position]