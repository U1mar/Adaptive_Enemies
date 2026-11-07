"""Enemy classes with state machine: Patroller, Flanker."""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
import random
import math

from grid import Grid, Position
from pathing import astar
from learning import Tracker, ZoneModel
from ai.costs import build_cost_map, build_flank_targets
from game.constants import DEFAULT_SIGHT_RADIUS, REPLAN_INTERVAL


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
    replan_interval: int = REPLAN_INTERVAL
    sight_radius: int = DEFAULT_SIGHT_RADIUS

    def _can_see_player(self, grid: Grid, player_pos: Position) -> bool:
        """Check if player is within sight radius and line of sight (Bresenham). Walls block vision."""
        # Check distance
        dist = abs(self.position[0] - player_pos[0]) + abs(self.position[1] - player_pos[1])
        if dist > self.sight_radius:
            return False

        # Bresenham line of sight check
        return self._bresenham_los(grid, self.position, player_pos)

    def sees(self, grid: Grid, player_pos: Position) -> bool:
        """Public method to check if enemy can see player. Alias for _can_see_player."""
        return self._can_see_player(grid, player_pos)

    def _bresenham_los(self, grid: Grid, start: Position, end: Position) -> bool:
        """Check line of sight using Bresenham's line algorithm. Walls block vision."""
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

    def _move_one_step_toward(self, grid: Grid, target: Position, cost_map: Dict[Position, float] = None) -> Position:
        """Move one tile toward target using A*. Returns new position."""
        if self.position == target:
            return self.position

        path, _ = astar(grid, self.position, target, cost_map)
        if path and len(path) > 1:
            # path[0] is current position, path[1] is next step
            return path[1]
        return self.position

    def update(self, grid: Grid, player_pos: Position, tracker: Tracker,
               zone_model: ZoneModel, alpha: float = 3.0) -> Position:
        """Update enemy position based on state machine. Returns new position."""
        # State transitions
        if self.state == EnemyState.PATROL:
            if self._can_see_player(grid, player_pos):
                self.state = EnemyState.CHASE
                self.target = player_pos
                self.replan_counter = self.replan_interval  # Force replan on state change
            else:
                # Continue patrolling - move along patrol route
                self.position = self._patrol_step(grid, tracker, alpha)
                return self.position

        elif self.state == EnemyState.CHASE:
            if not self._can_see_player(grid, player_pos):
                self.state = EnemyState.PATROL
                self.target = None
                self.path = []
                self.position = self._patrol_step(grid, tracker, alpha)
                return self.position
            else:
                # Update target to current player position
                self.target = player_pos
                self.replan_counter += 1
                cost_map = build_cost_map(tracker, alpha)
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

    def _patrol_step(self, grid: Grid, tracker: Tracker, alpha: float) -> Position:
        """Move one step along patrol route. To be overridden by subclasses."""
        if not self.patrol_route:
            return self.position
        self.route_index = (self.route_index + 1) % len(self.patrol_route)
        return self.patrol_route[self.route_index]

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
    """Patroller follows a connected loop of waypoints, switches to chase when seeing player.
    When not chasing, chooses next waypoint by sampling from tracker probability map.
    """

    def __init__(self, waypoints: List[Position], grid: Grid):
        # Build connected patrol route from waypoints using A*
        self.waypoints = waypoints
        self.waypoint_index = 0
        self.current_target = waypoints[0] if waypoints else None
        self.grid = grid
        # Build full patrol route connecting waypoints
        full_route = self._build_connected_route(grid, waypoints)
        super().__init__(full_route[0] if full_route else (0, 0), full_route)

    def _build_connected_route(self, grid: Grid, waypoints: List[Position]) -> List[Position]:
        """Build a connected patrol route by running A* between consecutive waypoints."""
        if len(waypoints) < 2:
            return waypoints

        full_route = []
        for i in range(len(waypoints)):
            start = waypoints[i]
            end = waypoints[(i + 1) % len(waypoints)]  # Loop back to first
            path, _ = astar(grid, start, end)
            if path:
                # Don't duplicate the start waypoint (except for first)
                if i == 0:
                    full_route.extend(path)
                else:
                    full_route.extend(path[1:])  # Skip duplicate
        return full_route

    def _choose_next_waypoint(self, tracker: Tracker) -> Position:
        """Choose next waypoint by sampling from tracker probability map, weighted by inverse distance."""
        prob_map = tracker.probability_map()
        if not prob_map or not self.waypoints:
            # Fallback: cycle through waypoints
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
            return self.waypoints[self.waypoint_index]

        # Filter waypoints that have probability data, weight by probability * inverse distance
        weighted_waypoints = []
        for wp in self.waypoints:
            prob = prob_map.get(wp, 0.01)  # Small default to allow exploration
            # Weight by inverse distance (closer waypoints more likely)
            dist = abs(self.position[0] - wp[0]) + abs(self.position[1] - wp[1])
            distance_weight = 1.0 / (1.0 + dist * 0.1)  # Decay with distance
            weight = prob * distance_weight
            weighted_waypoints.append((wp, weight))

        # Sample based on weighted probability
        total = sum(w for _, w in weighted_waypoints)
        if total > 0:
            r = random.random() * total
            accum = 0.0
            for wp, weight in weighted_waypoints:
                accum += weight
                if accum >= r:
                    return wp

        # Fallback
        self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
        return self.waypoints[self.waypoint_index]

    def _patrol_step(self, grid: Grid, tracker: Tracker, alpha: float) -> Position:
        """Move one step toward current waypoint target, choose new when reached."""
        if not self.patrol_route:
            return self.position

        # If we need a target or reached current target, pick next waypoint
        if self.current_target is None or self.position == self.current_target:
            self.current_target = self._choose_next_waypoint(tracker)

        # Move one step toward current target using A*
        cost_map = build_cost_map(tracker, alpha)
        new_pos = self._move_one_step_toward(grid, self.current_target, cost_map)
        return new_pos


class Flanker(Enemy):
    """Flanker waits on a chokepoint tile of the hottest zone that is on the shortest route toward the exit,
    and only chases when it can see the player.
    """

    def __init__(self, start_pos: Position, patrol_route: List[Position], grid: Grid):
        super().__init__(start_pos, patrol_route)
        self.grid = grid
        self.flank_target: Optional[Position] = None
        self.last_hottest_zone: Optional[str] = None
        self.state = EnemyState.FLANK
        self.chokepoints: List[Position] = []  # Chokepoints on routes (set by engine)

    def update(self, grid: Grid, player_pos: Position, tracker: Tracker,
               zone_model: ZoneModel, alpha: float = 3.0) -> Position:
        # Find hottest zone from tracker
        prob_map = tracker.probability_map()
        # Initialize zone_heat with all actual zone names from the grid
        zone_heat: Dict[str, float] = {zone: 0.0 for zone in grid.get_zone_names()}
        for pos, prob in prob_map.items():
            zone = grid.zone_of(pos)
            if zone in zone_heat:
                zone_heat[zone] += prob

        hottest_zone = max(zone_heat, key=zone_heat.get) if prob_map and zone_heat else 'route_0'

        # Find flank target: chokepoint in hottest zone on shortest path toward exit
        # Only recompute when hottest zone changes
        if self.flank_target is None or hottest_zone != self.last_hottest_zone:
            self.flank_target = self._compute_flank_target(grid, prob_map, hottest_zone)
            self.last_hottest_zone = hottest_zone

        if self.state == EnemyState.FLANK:
            if self._can_see_player(grid, player_pos):
                self.state = EnemyState.CHASE
                self.target = player_pos
                self.replan_counter = self.replan_interval  # Force replan
                # Clear flank target when switching to chase
                self.flank_target = None
            elif self.flank_target and self.position != self.flank_target:
                # Move toward flank target
                cost_map = build_cost_map(tracker, alpha)
                new_pos = self._move_one_step_toward(grid, self.flank_target, cost_map)
                self.position = new_pos
                return self.position
            else:
                # At flank target - wait in ambush
                return self.position

        elif self.state == EnemyState.CHASE:
            if not self._can_see_player(grid, player_pos):
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

    def _compute_flank_target(self, grid: Grid, prob_map: Dict[Position, float], hottest_zone: str) -> Optional[Position]:
        """Find a chokepoint in the hottest zone that lies on the shortest path toward the exit."""
        # First, try to find a chokepoint in the hottest zone
        candidates = []
        for cp in self.chokepoints:
            if grid.zone_of(cp) == hottest_zone:
                # Check if this chokepoint is on a reasonable path toward exit
                path_to_exit, _ = astar(grid, cp, grid.exit)
                if path_to_exit and len(path_to_exit) > 1:
                    prob = prob_map.get(cp, 0.0)
                    candidates.append((cp, prob, len(path_to_exit)))

        if candidates:
            # Prefer chokepoints with higher probability and shorter path to exit
            best = max(candidates, key=lambda c: c[1] - 0.01 * c[2])
            return best[0]

        # Fallback: any walkable tile in the hottest zone on path toward exit
        fallback_candidates = []
        for y in range(grid.height):
            for x in range(grid.width):
                pos = (x, y)
                if not grid.is_walkable(pos):
                    continue
                if grid.zone_of(pos) != hottest_zone:
                    continue

                path_to_exit, _ = astar(grid, pos, grid.exit)
                if path_to_exit and len(path_to_exit) > 1:
                    prob = prob_map.get(pos, 0.0)
                    fallback_candidates.append((pos, prob, len(path_to_exit)))

        if fallback_candidates:
            best = max(fallback_candidates, key=lambda c: c[1] - 0.01 * c[2])
            return best[0]

        # Ultimate fallback: any walkable tile in the hottest zone
        for y in range(grid.height):
            for x in range(grid.width):
                pos = (x, y)
                if grid.is_walkable(pos) and grid.zone_of(pos) == hottest_zone:
                    return pos
        return None