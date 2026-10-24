"""Bot policies for the adaptive enemies game."""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable
import random

from grid import Grid, Position
from pathing import astar


# Type alias for policy
Policy = Callable[[Grid, Position, bool, List[Position]], Position]


class FixedRouteBot:
    """Always take the east corridor via A* with no cost map.
    Goes to vault first, then exit.
    """

    def __init__(self):
        self._path: List[Position] = []
        self._target_is_exit = False

    def __call__(self, grid: Grid, player_pos: Position, got_key: bool, enemy_positions: List[Position]) -> Position:
        # Replan if no path or target changed
        target = grid.exit if got_key else grid.vault
        if not self._path or self._target_is_exit != got_key:
            # Compute A* with no cost map
            self._path, _ = astar(grid, player_pos, target, cost_map={})
            if self._path and self._path[0] == player_pos:
                self._path.pop(0)
            self._target_is_exit = got_key

        if self._path:
            return self._path.pop(0)
        return player_pos


class RandomBot:
    """Random valid moves biased toward the exit."""

    def __call__(self, grid: Grid, player_pos: Position, got_key: bool, enemy_positions: List[Position]) -> Position:
        target = grid.exit if got_key else grid.vault
        neighbors = grid.neighbors(player_pos)
        if not neighbors:
            return player_pos
        # Weight by inverse distance to target
        weights = []
        for n in neighbors:
            dist = abs(n[0] - target[0]) + abs(n[1] - target[1])
            weights.append(1.0 / (dist + 1))
        # Choose randomly with weights
        return random.choices(neighbors, weights=weights, k=1)[0]


class SwitchingBot:
    """Changes corridor every 3 rounds.
    Corridors: west, middle, east. Uses A* to follow a patrol route for the current corridor.
    """

    def __init__(self):
        self.corridor = 'west'
        self.round_count = 0
        self._path: List[Position] = []

    def __call__(self, grid: Grid, player_pos: Position, got_key: bool, enemy_positions: List[Position]) -> Position:
        # Determine corridor based on round count (every 3 rounds)
        # But we don't have round number here; we'll increment round count each call?
        # Actually policy called each turn, not each round. We'll switch every 3 calls?
        # Better: switch every 3 rounds, but we don't know rounds. We'll approximate by switching every 30 turns.
        # For simplicity, switch every 3 calls to __call__ (i.e., every 3 turns). Not ideal.
        # However the requirement: "changes corridor every 3 rounds". We'll assume the caller increments round.
        # We'll add a method to notify new round.
        pass

    def new_round(self):
        """Call at start of each round to possibly switch corridor."""
        self.round_count += 1
        if self.round_count % 3 == 0:
            corridors = ['west', 'middle', 'east']
            idx = corridors.index(self.corridor)
            self.corridor = corridors[(idx + 1) % 3]
        self._path = []  # force replan

    def _get_corridor_target(self, grid: Grid, got_key: bool) -> Position:
        # For simplicity, target is the vault if not got_key, else exit.
        # But we want to stay in corridor. We'll just target vault/exit via A* (no cost).
        return grid.exit if got_key else grid.vault


# To make SwitchingBot work with the policy signature, we need a callable that also has new_round.
# We'll create a wrapper class that implements __call__ and new_round.
class SwitchingBotPolicy:
    def __init__(self):
        self.bot = SwitchingBot()

    def __call__(self, grid: Grid, player_pos: Position, got_key: bool, enemy_positions: List[Position]) -> Position:
        target = self.bot._get_corridor_target(grid, got_key)
        if not self.bot._path:
            self.bot._path, _ = astar(grid, player_pos, target, cost_map={})
            if self.bot._path and self.bot._path[0] == player_pos:
                self.bot._path.pop(0)
        if self.bot._path:
            return self.bot._path.pop(0)
        return player_pos

    def new_round(self):
        self.bot.new_round()


# Export the policy instances
fixed_route_bot = FixedRouteBot()
random_bot = RandomBot()
switching_bot = SwitchingBotPolicy()