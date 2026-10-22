"""Enemy classes: Patroller and Flanker."""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
import random

from grid import Grid, Position
from pathing import astar
from learning import Tracker, ZoneModel


@dataclass
class Enemy:
    """Base enemy class."""
    position: Position
    patrol_route: List[Position]
    route_index: int = 0
    chase_mode: bool = False
    target: Position = None
    path: List[Position] = field(default_factory=list)
    replan_counter: int = 0
    replan_interval: int = 3

    def patrol_step(self) -> Position:
        """Move to next position in patrol route."""
        if not self.patrol_route:
            return self.position
        self.route_index = (self.route_index + 1) % len(self.patrol_route)
        return self.patrol_route[self.route_index]

    def update(self, grid: Grid, player_pos: Position, tracker: Tracker,
               zone_model: ZoneModel, alpha: float = 0.5) -> Position:
        """Update enemy position. Returns new position."""
        # Check if should chase (player within 5 tiles, no wall between)
        if self._can_see_player(grid, player_pos):
            self.chase_mode = True
            self.target = player_pos
        else:
            self.chase_mode = False

        if self.chase_mode and self.target:
            self.replan_counter += 1
            if self.replan_counter >= self.replan_interval or not self.path:
                self._replan_chase(grid, tracker, alpha)
                self.replan_counter = 0

            if self.path:
                next_pos = self.path.pop(0)
                self.position = next_pos
                return self.position

        # Patrol mode
        self.position = self.patrol_step()
        return self.position

    def _can_see_player(self, grid: Grid, player_pos: Position) -> bool:
        """Check if player is within 5 Manhattan distance with line of sight."""
        dist = abs(self.position[0] - player_pos[0]) + abs(self.position[1] - player_pos[1])
        if dist > 5:
            return False
        # Simple line of sight: check if straight path has walls
        # For simplicity, just check Manhattan distance for now
        return True

    def _replan_chase(self, grid: Grid, tracker: Tracker, alpha: float) -> None:
        """Replan chase path using A* with cost map from tracker."""
        prob_map = tracker.probability_map()
        cost_map = {pos: -alpha * prob for pos, prob in prob_map.items()}
        self.path, _ = astar(grid, self.position, self.target, cost_map)
        if self.path and self.path[0] == self.position:
            self.path.pop(0)


class Patroller(Enemy):
    """Patroller follows a fixed loop."""
    def __init__(self, start_pos: Position, patrol_route: List[Position]):
        super().__init__(start_pos, patrol_route)


class Flanker(Enemy):
    """Flanker waits at exit of hottest zone."""
    def __init__(self, start_pos: Position, patrol_route: List[Position], grid: Grid):
        super().__init__(start_pos, patrol_route)
        self.grid = grid
        self.wait_position: Optional[Position] = None

    def update(self, grid: Grid, player_pos: Position, tracker: Tracker,
               zone_model: ZoneModel, alpha: float = 0.5) -> Position:
        # Find hottest zone
        prob_map = tracker.probability_map()
        zone_heat = {'west': 0.0, 'middle': 0.0, 'east': 0.0}
        for pos, prob in prob_map.items():
            zone = grid.zone_of(pos)
            zone_heat[zone] += prob

        hottest_zone = max(zone_heat, key=zone_heat.get)

        # Find exit of hottest zone (tile adjacent to another zone or map edge)
        self.wait_position = self._find_zone_exit(grid, hottest_zone)

        # If player in hottest zone, chase
        player_zone = grid.zone_of(player_pos)
        if player_zone == hottest_zone:
            self.chase_mode = True
            self.target = player_pos
        else:
            self.chase_mode = False
            # Move toward wait position
            if self.wait_position and self.position != self.wait_position:
                if not self.path or self.replan_counter >= self.replan_interval:
                    prob_map = tracker.probability_map()
                    cost_map = {pos: -alpha * prob for pos, prob in prob_map.items()}
                    self.path, _ = astar(grid, self.position, self.wait_position, cost_map)
                    self.replan_counter = 0
                    if self.path and self.path[0] == self.position:
                        self.path.pop(0)
                self.replan_counter += 1
                if self.path:
                    self.position = self.path.pop(0)
                    return self.position

        # Chase logic (same as base but with replan)
        if self.chase_mode and self.target:
            self.replan_counter += 1
            if self.replan_counter >= self.replan_interval or not self.path:
                prob_map = tracker.probability_map()
                cost_map = {pos: -alpha * prob for pos, prob in prob_map.items()}
                self.path, _ = astar(grid, self.position, self.target, cost_map)
                self.replan_counter = 0
                if self.path and self.path[0] == self.position:
                    self.path.pop(0)
            if self.path:
                self.position = self.path.pop(0)
                return self.position

        # Default: patrol
        self.position = self.patrol_step()
        return self.position

    def _find_zone_exit(self, grid: Grid, zone: str) -> Position:
        """Find a tile at the boundary of the given zone."""
        third = grid.width // 3
        if zone == 'west':
            target_x = third - 1
        elif zone == 'middle':
            target_x = 2 * third - 1
        else:  # east
            target_x = grid.width - 2

        # Find walkable tile near target_x
        for y in range(grid.height):
            for x in [target_x, target_x + 1, target_x - 1]:
                if 0 <= x < grid.width and grid.is_walkable((x, y)):
                    return (x, y)
        return grid.start