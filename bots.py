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
    """Always takes the east corridor from S to V to E."""

    def __init__(self):
        self.target_sequence = ['vault', 'exit']
        self.current_target = 0
        # Waypoints for east corridor: S -> east corridor top -> east corridor at V level -> V
        # then V -> east corridor at V level -> east corridor at E level -> E
        self.to_vault_waypoints = [(21, 1), (21, 7)]  # Enter east corridor, go down to V row
        self.to_exit_waypoints = [(21, 7), (21, 13)]   # From V to east corridor, down to E row
        self.waypoint_index = 0

    def __call__(self, grid: Grid, player_pos: Position, got_key: bool,
                 enemy_positions: List[Position]) -> Position:
        if not got_key:
            # Heading to vault via east corridor
            waypoints = self.to_vault_waypoints
            final_target = grid.vault
        else:
            # Heading to exit via east corridor
            waypoints = self.to_exit_waypoints
            final_target = grid.exit

        # Determine current target waypoint
        if self.waypoint_index < len(waypoints):
            target = waypoints[self.waypoint_index]
            # Check if reached current waypoint
            if player_pos == target:
                self.waypoint_index += 1
        else:
            target = final_target

        return find_path_to(grid, player_pos, target)

    def new_round(self):
        """Reset for new round."""
        self.waypoint_index = 0


class SwitchingBot:
    """Changes corridor every 3 rounds. Follows the corridor named in self.corridor."""

    def __init__(self):
        self.round_count = 0
        self.corridor = 'west'
        self.waypoint_index = 0

        # Define waypoints for each corridor
        # Corridors: west (x=1-9), middle (x=11-19), east (x=21-28)
        # S at (1,1), V at (14,7), E at (28,13)
        # Openings at rows 1, 7, 13 at x=10 and x=20
        self.corridor_waypoints = {
            'west': {
                'to_vault': [(1, 7)],           # Go down west corridor to V row
                'to_exit': [(1, 7), (1, 13)],   # Back to west, down to E row
            },
            'middle': {
                'to_vault': [(11, 1), (11, 7)],  # Enter middle at top, down to V
                'to_exit': [(11, 7), (11, 13)],  # From V down middle to E row
            },
            'east': {
                'to_vault': [(21, 1), (21, 7)],   # Enter east at top, down to V
                'to_exit': [(21, 7), (21, 13)],   # From V down east to E row
            },
        }

    def __call__(self, grid: Grid, player_pos: Position, got_key: bool,
                 enemy_positions: List[Position]) -> Position:
        corridor = self.corridor
        waypoints = self.corridor_waypoints[corridor]

        if not got_key:
            wp_list = waypoints['to_vault']
            final_target = grid.vault
        else:
            wp_list = waypoints['to_exit']
            final_target = grid.exit

        # Determine current target waypoint
        if self.waypoint_index < len(wp_list):
            target = wp_list[self.waypoint_index]
            # Check if reached current waypoint
            if player_pos == target:
                self.waypoint_index += 1
        else:
            target = final_target

        return find_path_to(grid, player_pos, target)

    def new_round(self):
        self.round_count += 1
        if self.round_count % 3 == 0:
            corridors = ['west', 'middle', 'east']
            self.corridor = corridors[(corridors.index(self.corridor) + 1) % 3]
        self.waypoint_index = 0


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