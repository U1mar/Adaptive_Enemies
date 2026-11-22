"""Enemy classes with state machine: Patroller, Flanker."""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
import random
import math

from grid import Grid, Position
from pathing import astar, astar_result, distance_map
from learning import Tracker, ZoneModel
from ai.costs import build_cost_map, build_flank_targets, get_step_costs
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
    # Task 2.1: every enemy owns its random generator (set by the engine); never use global random.
    rng: random.Random = field(default_factory=lambda: random.Random(0), repr=False)
    # Task 2.4: GUI explored-node overlay (only filled when record_explored is True)
    record_explored: bool = False
    last_explored: List[Position] = field(default_factory=list, repr=False)
    # Task 2.4: the A* path of the latest patrol/flank step, for drawing only (never used for movement)
    path_preview: List[Position] = field(default_factory=list, repr=False)

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

    def _search(self, grid: Grid, target: Position, cost_map) -> List[Position]:
        """A* from the current position. cost_map may be a dict or a shared StepCostMap."""
        if self.record_explored:
            res = astar_result(grid, self.position, target, cost_map, record=True)
            self.last_explored = res.explored
            return res.path
        path, _ = astar(grid, self.position, target, cost_map)
        return path

    def _move_one_step_toward(self, grid: Grid, target: Position, cost_map=None) -> Position:
        """Move one tile toward target using A*. Returns new position."""
        if self.position == target:
            return self.position

        path = self._search(grid, target, cost_map)
        self.path_preview = path
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
        if self.target:
            self.path = self._search(grid, self.target, get_step_costs(grid, tracker, alpha))
            if self.path and self.path[0] == self.position:
                self.path.pop(0)

    def _replan_to_target(self, grid: Grid, tracker: Tracker, alpha: float, target: Position) -> bool:
        """Replan path to a specific target. Returns True if path found."""
        self.path = self._search(grid, target, get_step_costs(grid, tracker, alpha))
        if self.path and self.path[0] == self.position:
            self.path.pop(0)
        return len(self.path) > 0


class Patroller(Enemy):
    """Patroller follows a connected loop of waypoints, switches to chase when seeing player.
    When not chasing, chooses next waypoint by sampling from tracker probability map.
    """

    def __init__(self, waypoints: List[Position], grid: Grid, rng: Optional[random.Random] = None):
        # Build connected patrol route from waypoints using A*
        self.waypoints = waypoints
        self.waypoint_index = 0
        self.current_target = waypoints[0] if waypoints else None
        self.grid = grid
        # Build full patrol route connecting waypoints
        full_route = self._build_connected_route(grid, waypoints)
        super().__init__(full_route[0] if full_route else (0, 0), full_route)
        if rng is not None:
            self.rng = rng

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
            r = self.rng.random() * total
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
        new_pos = self._move_one_step_toward(grid, self.current_target, get_step_costs(grid, tracker, alpha))
        return new_pos


class Flanker(Enemy):
    """Flanker waits on a chokepoint tile of the hottest zone that is on the shortest route toward the exit,
    and only chases when it can see the player.
    """

    def __init__(self, start_pos: Position, patrol_route: List[Position], grid: Grid,
                 rng: Optional[random.Random] = None):
        super().__init__(start_pos, patrol_route)
        if rng is not None:
            self.rng = rng
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
                new_pos = self._move_one_step_toward(grid, self.flank_target, get_step_costs(grid, tracker, alpha))
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
        """Find a chokepoint in the hottest zone that lies on the shortest path toward the exit.

        Speed (task 2.2): the old code ran A* from every candidate tile to the exit. One BFS from the
        exit gives every tile's distance at once (cached per grid); path length = steps + 1, so the
        scores and the winner are the same as before.
        """
        dist_exit = distance_map(grid, grid.exit)

        def score_all(tiles):
            out = []
            for pos in tiles:
                steps = dist_exit.get(pos)
                if steps is not None and steps + 1 > 1:     # reachable and not the exit itself
                    out.append((pos, prob_map.get(pos, 0.0), steps + 1))
            return out

        # First, try to find a chokepoint in the hottest zone
        candidates = score_all(cp for cp in self.chokepoints if grid.zone_of(cp) == hottest_zone)
        if candidates:
            # Prefer chokepoints with higher probability and shorter path to exit
            return max(candidates, key=lambda c: c[1] - 0.01 * c[2])[0]

        # Fallback: any walkable tile in the hottest zone on path toward exit
        zone_tiles = [(x, y) for y in range(grid.height) for x in range(grid.width)
                      if grid.is_walkable((x, y)) and grid.zone_of((x, y)) == hottest_zone]
        fallback = score_all(zone_tiles)
        if fallback:
            return max(fallback, key=lambda c: c[1] - 0.01 * c[2])[0]

        # Ultimate fallback: any walkable tile in the hottest zone
        return zone_tiles[0] if zone_tiles else None
