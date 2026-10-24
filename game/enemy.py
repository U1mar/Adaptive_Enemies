"""Enemy classes with state machine: Patroller, Flanker."""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum

from grid import Grid, Position
from pathing import astar
from learning import Tracker, ZoneModel
from ai.costs import build_cost_map, build_flank_targets


class EnemyState(Enum):
    PATROL = "patrol"
    CHASE = "chase"
    FLANK = "flank"


@dataclass
class Enemy:
    """Base enemy class with state machine."""
    position: Position
    patrol_route: List[Position]
    route_index: int = 0
    state: EnemyState = EnemyState.PATROL
    target: Optional[Position] = None
    path: List[Position] = field(default_factory=list)
    replan_counter: int = 0
    replan_interval: int = 3
    sight_radius: int = 5

    def sees(self, grid: Grid, player_pos: Position) -> bool:
        """Check if player is within sight radius and line of sight (no wall on straight line)."""
        # Check distance
        dist = abs(self.position[0] - player_pos[0]) + abs(self.position[1] - player_pos[1])
        if dist > self.sight_radius:
            return False

        # Simple stepping line check for line of sight
        return self._line_of_sight(grid, self.position, player_pos)

    def _line_of_sight(self, grid: Grid, start: Position, end: Position) -> bool:
        """Check if there's a clear line of sight using Bresenham-style stepping."""
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            # Skip the start position
            if (x, y) != start and (x, y) != end:
                if grid.is_wall((x, y)):
                    return False
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True

    def patrol_step(self) -> Position:
        """Move to next position in patrol route."""
        if not self.patrol_route:
            return self.position
        self.route_index = (self.route_index + 1) % len(self.patrol_route)
        return self.patrol_route[self.route_index]

    def update(self, grid: Grid, player_pos: Position, tracker: Tracker,
               zone_model: ZoneModel, alpha: float = 0.5) -> Position:
        """Update enemy position based on state machine. Returns new position."""
        # State transitions
        if self.state == EnemyState.PATROL:
            if self.sees(grid, player_pos):
                self.state = EnemyState.CHASE
                self.target = player_pos
                self.replan_counter = self.replan_interval  # Force replan on state change
            else:
                # Continue patrolling
                self.position = self.patrol_step()
                return self.position

        elif self.state == EnemyState.CHASE:
            if not self.sees(grid, player_pos):
                self.state = EnemyState.PATROL
                self.target = None
                self.path = []
                self.position = self.patrol_step()
                return self.position
            else:
                # Update target to current player position
                self.target = player_pos
                self.replan_counter += 1
                if self.replan_counter >= self.replan_interval or not self.path:
                    self._replan_chase(grid, tracker, alpha)
                    self.replan_counter = 0

                if self.path:
                    next_pos = self.path.pop(0)
                    self.position = next_pos
                    return self.position
                else:
                    # No path found, stay in place
                    return self.position

        elif self.state == EnemyState.FLANK:
            # Flanker-specific logic handled in Flanker subclass
            pass

        return self.position

    def _replan_chase(self, grid: Grid, tracker: Tracker, alpha: float) -> None:
        """Replan chase path using A* with cost map from tracker."""
        cost_map = build_cost_map(tracker, alpha)
        if self.target:
            self.path, _ = astar(grid, self.position, self.target, cost_map)
            if self.path and self.path[0] == self.position:
                self.path.pop(0)

    def _replan_to_target(self, grid: Grid, tracker: Tracker, alpha: float, target: Position) -> bool:
        """Replan path to a specific target. Returns True if path found."""
        cost_map = build_cost_map(tracker, alpha)
        self.path, _ = astar(grid, self.position, target, cost_map)
        if self.path and self.path[0] == self.position:
            self.path.pop(0)
        return len(self.path) > 0


class Patroller(Enemy):
    """Patroller follows a fixed loop, switches to chase when seeing player."""
    def __init__(self, start_pos: Position, patrol_route: List[Position]):
        super().__init__(start_pos, patrol_route)


class Flanker(Enemy):
    """Flanker waits at exit of hottest zone, ambushes when player enters."""
    def __init__(self, start_pos: Position, patrol_route: List[Position], grid: Grid):
        super().__init__(start_pos, patrol_route)
        self.grid = grid
        self.flank_target: Optional[Position] = None
        self.state = EnemyState.FLANK

    def update(self, grid: Grid, player_pos: Position, tracker: Tracker,
               zone_model: ZoneModel, alpha: float = 0.5) -> Position:
        # Find hottest zone
        prob_map = tracker.probability_map()
        zone_heat: Dict[str, float] = {'west': 0.0, 'middle': 0.0, 'east': 0.0}
        for pos, prob in prob_map.items():
            zone = grid.zone_of(pos)
            zone_heat[zone] += prob

        hottest_zone = max(zone_heat, key=zone_heat.get)

        # Find flank target (exit-side of hottest zone)
        self.flank_target = build_flank_targets(grid, prob_map)

        # Check if player is in hottest zone
        player_zone = grid.zone_of(player_pos)

        if self.state == EnemyState.FLANK:
            if self.sees(grid, player_pos):
                self.state = EnemyState.CHASE
                self.target = player_pos
                self.replan_counter = self.replan_interval  # Force replan
            elif player_zone == hottest_zone and self.flank_target:
                # Player in hottest zone but not seen - move toward flank target to ambush
                if self.position != self.flank_target:
                    self.replan_counter += 1
                    if self.replan_counter >= self.replan_interval or not self.path:
                        self._replan_to_target(grid, tracker, alpha, self.flank_target)
                        self.replan_counter = 0
                    if self.path:
                        self.position = self.path.pop(0)
                        return self.position
                # At flank target - wait in ambush
                return self.position
            else:
                # Patrol when player not in hottest zone and not seen
                self.position = self.patrol_step()
                return self.position

        elif self.state == EnemyState.CHASE:
            if not self.sees(grid, player_pos):
                self.state = EnemyState.FLANK
                self.target = None
                self.path = []
                # Will handle flank logic on next update
                return self.position
            else:
                self.target = player_pos
                self.replan_counter += 1
                if self.replan_counter >= self.replan_interval or not self.path:
                    self._replan_chase(grid, tracker, alpha)
                    self.replan_counter = 0

                if self.path:
                    self.position = self.path.pop(0)
                    return self.position
                else:
                    return self.position

        return self.position