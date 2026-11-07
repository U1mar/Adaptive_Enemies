"""Bot policies for automated play."""
from __future__ import annotations
from typing import List, Tuple, Callable, Dict, Optional
from grid import Grid, Position
from pathing import astar
import random
from collections import Counter, deque


Policy = Callable[[Grid, Position, bool, List[Position]], Position]


def find_path_to(grid: Grid, start: Position, goal: Position) -> Position:
    """Get next step toward goal using A*."""
    path, _ = astar(grid, start, goal)
    if len(path) > 1:
        return path[1]
    return start


def _get_corridor_waypoints(grid: Grid) -> Dict[str, Dict[str, List[Position]]]:
    """Generate waypoints for each detected corridor dynamically.
    Returns dict mapping zone_name -> {'to_vault': [...], 'to_exit': [...]}
    """
    corridors = grid.get_corridors()
    if not corridors:
        return {}

    waypoints = {}
    start_row = grid.start[1]
    vault_row = grid.vault[1]
    exit_row = grid.exit[1]

    for corridor_start, corridor_end, zone_name in corridors:
        mid_x = (corridor_start + corridor_end) // 2
        to_vault = [(mid_x, start_row), (mid_x, vault_row)]
        to_exit = [(mid_x, vault_row), (mid_x, exit_row)]

        waypoints[zone_name] = {
            'to_vault': to_vault,
            'to_exit': to_exit,
        }

    return waypoints


class FixedRouteBot:
    """Always takes the rightmost corridor from S to V to E."""

    def __init__(self, seed: int = 0):
        self.waypoint_index = 0
        self.corridor_waypoints = {}
        self.current_corridor = None

    def _ensure_waypoints(self, grid: Grid):
        if not self.corridor_waypoints:
            self.corridor_waypoints = _get_corridor_waypoints(grid)
            if self.corridor_waypoints:
                self.current_corridor = list(self.corridor_waypoints.keys())[-1]

    def __call__(self, grid: Grid, player_pos: Position, got_key: bool,
                 enemy_positions: List[Position]) -> Position:
        self._ensure_waypoints(grid)

        if not self.corridor_waypoints or self.current_corridor not in self.corridor_waypoints:
            target = grid.vault if not got_key else grid.exit
            return find_path_to(grid, player_pos, target)

        waypoints = self.corridor_waypoints[self.current_corridor]

        if not got_key:
            wp_list = waypoints['to_vault']
            final_target = grid.vault
        else:
            wp_list = waypoints['to_exit']
            final_target = grid.exit

        if self.waypoint_index < len(wp_list):
            target = wp_list[self.waypoint_index]
            if player_pos == target:
                self.waypoint_index += 1
        else:
            target = final_target

        return find_path_to(grid, player_pos, target)

    def new_round(self):
        self.waypoint_index = 0


class SwitchingBot:
    """Changes corridor every 3 rounds."""

    def __init__(self, seed: int = 0):
        self.round_count = 0
        self.corridor_index = 0
        self.waypoint_index = 0
        self.corridor_waypoints = {}
        self.corridor_names = []

    def _ensure_waypoints(self, grid: Grid):
        if not self.corridor_waypoints:
            self.corridor_waypoints = _get_corridor_waypoints(grid)
            self.corridor_names = list(self.corridor_waypoints.keys())
            if self.corridor_names:
                self.corridor_index = 0

    def __call__(self, grid: Grid, player_pos: Position, got_key: bool,
                 enemy_positions: List[Position]) -> Position:
        self._ensure_waypoints(grid)

        if not self.corridor_names:
            target = grid.vault if not got_key else grid.exit
            return find_path_to(grid, player_pos, target)

        corridor = self.corridor_names[self.corridor_index]
        waypoints = self.corridor_waypoints[corridor]

        if not got_key:
            wp_list = waypoints['to_vault']
            final_target = grid.vault
        else:
            wp_list = waypoints['to_exit']
            final_target = grid.exit

        if self.waypoint_index < len(wp_list):
            target = wp_list[self.waypoint_index]
            if player_pos == target:
                self.waypoint_index += 1
        else:
            target = final_target

        return find_path_to(grid, player_pos, target)

    def new_round(self):
        self.round_count += 1
        if self.round_count % 3 == 0 and self.corridor_names:
            self.corridor_index = (self.corridor_index + 1) % len(self.corridor_names)
        self.waypoint_index = 0


class RandomBot:
    """Random valid moves."""

    def __init__(self, seed: int = 0):
        import random
        self.random = random

    def __call__(self, grid: Grid, player_pos: Position, got_key: bool,
                 enemy_positions: List[Position]) -> Position:
        neighbors = grid.neighbors(player_pos)
        if not neighbors:
            return player_pos
        return self.random.choice(neighbors)


class AdaptiveBot:
    """Bot that adapts to enemy positions - tries to avoid enemies while heading to goal."""

    def __init__(self, avoidance_radius: int = 5, seed: int = 0):
        self.avoidance_radius = avoidance_radius
        self.waypoint_index = 0
        self.corridor_waypoints = {}
        self.current_corridor = None

    def _ensure_waypoints(self, grid: Grid):
        if not self.corridor_waypoints:
            self.corridor_waypoints = _get_corridor_waypoints(grid)
            if self.corridor_waypoints:
                self.current_corridor = list(self.corridor_waypoints.keys())[-1]

    def _is_safe(self, grid: Grid, pos: Position, enemy_positions: List[Position]) -> bool:
        for enemy_pos in enemy_positions:
            dist = abs(pos[0] - enemy_pos[0]) + abs(pos[1] - enemy_pos[1])
            if dist <= self.avoidance_radius:
                return False
        return True

    def _find_safe_step(self, grid: Grid, current: Position, goal: Position,
                        enemy_positions: List[Position]) -> Position:
        path, _ = astar(grid, current, goal)
        if len(path) <= 1:
            return current

        next_pos = path[1]
        if self._is_safe(grid, next_pos, enemy_positions):
            return next_pos

        neighbors = grid.neighbors(current)
        safe_neighbors = [n for n in neighbors if self._is_safe(grid, n, enemy_positions)]

        if safe_neighbors:
            best = min(safe_neighbors, key=lambda n: grid.path_distance(n, goal) or float('inf'))
            return best

        return next_pos

    def __call__(self, grid: Grid, player_pos: Position, got_key: bool,
                 enemy_positions: List[Position]) -> Position:
        self._ensure_waypoints(grid)

        if not got_key:
            target = grid.vault
        else:
            target = grid.exit

        if not enemy_positions:
            return find_path_to(grid, player_pos, target)

        path, _ = astar(grid, player_pos, target)
        if len(path) > 1:
            next_pos = path[1]
            if self._is_safe(grid, next_pos, enemy_positions):
                return next_pos

        return self._find_safe_step(grid, player_pos, target, enemy_positions)

    def new_round(self):
        self.waypoint_index = 0


# ============================================================
# NEW BOT CLASSES WITH PER-ROUND ROUTE VARIATION
# ============================================================

class _RouteBasedBot:
    """Base class for bots that use the level's validated routes (S->V->E)."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.round_num = 0
        self.waypoint_index = 0
        self.current_route_idx = 0
        self.current_route = None
        self.route_waypoints = None
        self._route_history = deque(maxlen=5)  # Last 5 routes used
        self._explorer_phase = 'explore'  # For ExplorerBot
        self._explorer_waypoints = []  # For ExplorerBot
        self._explorer_wp_index = 0

    def _get_routes(self, grid: Grid) -> List[List[Position]]:
        """Get the validated S->V->E routes from the grid."""
        if hasattr(grid, 'routes') and grid.routes:
            return grid.routes
        # Fallback: corridor-based zones
        return []

    def _get_route_waypoints(self, route: List[Position]) -> List[Position]:
        """Sample waypoints from a route for patrol."""
        if len(route) < 2:
            return route
        num_waypoints = min(5, max(3, len(route) // 3))
        step = max(1, len(route) // num_waypoints)
        waypoints = [route[i] for i in range(0, len(route), step)][:num_waypoints]
        return waypoints

    def _apply_move_noise(self, grid: Grid, player_pos: Position,
                          enemy_positions: List[Position], intended_move: Position) -> Position:
        """Apply per-move noise: 5% pause, 5% sidestep, switch route if enemy within 3."""
        # 5% chance to pause
        if self.rng.random() < 0.05:
            return player_pos

        # Check if enemy within 3 tiles - may switch to alternate route
        for enemy_pos in enemy_positions:
            dist = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
            if dist <= 3 and self.rng.random() < 0.5:
                # Try to find alternate route
                routes = self._get_routes(grid)
                if routes and len(routes) > 1:
                    alt_routes = [i for i in range(len(routes)) if i != self.current_route_idx]
                    if alt_routes:
                        self.current_route_idx = self.rng.choice(alt_routes)
                        self.route_waypoints = self._get_route_waypoints(routes[self.current_route_idx])
                        self.waypoint_index = 0
                        # Recalculate move after route switch
                        return self._get_next_step(grid, player_pos, enemy_positions)

        # 5% chance to sidestep if free
        if self.rng.random() < 0.05:
            neighbors = grid.neighbors(player_pos)
            free_neighbors = [n for n in neighbors if n != intended_move]
            if free_neighbors:
                return self.rng.choice(free_neighbors)

        return intended_move

    def _get_next_step(self, grid: Grid, player_pos: Position,
                       enemy_positions: List[Position]) -> Position:
        """Override in subclasses to get the intended next step."""
        raise NotImplementedError

    def __call__(self, grid: Grid, player_pos: Position, got_key: bool,
                 enemy_positions: List[Position]) -> Position:
        intended = self._get_next_step(grid, player_pos, enemy_positions)
        return self._apply_move_noise(grid, player_pos, enemy_positions, intended)

    def new_round(self):
        """Called at start of each round. Override in subclasses."""
        self.round_num += 1
        self.waypoint_index = 0
        self._explorer_wp_index = 0
        self._explorer_phase = 'explore'


class HabitBot(_RouteBasedBot):
    """Uses one preferred route ~85% of rounds, deviates randomly 15%."""

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.preferred_route_idx = None

    def _get_next_step(self, grid: Grid, player_pos: Position,
                       enemy_positions: List[Position]) -> Position:
        routes = self._get_routes(grid)
        if not routes:
            return find_path_to(grid, player_pos, grid.vault if not hasattr(self, '_got_key') else grid.exit)

        # Choose preferred route on first round
        if self.preferred_route_idx is None:
            self.preferred_route_idx = self.rng.randrange(len(routes))
            self.current_route_idx = self.preferred_route_idx
            self.current_route = routes[self.current_route_idx]
            self.route_waypoints = self._get_route_waypoints(self.current_route)

        # 15% chance to deviate to a different route this round
        if self.round_num > 0 and self.rng.random() < 0.15 and len(routes) > 1:
            alt_routes = [i for i in range(len(routes)) if i != self.preferred_route_idx]
            self.current_route_idx = self.rng.choice(alt_routes)
            self.current_route = routes[self.current_route_idx]
            self.route_waypoints = self._get_route_waypoints(self.current_route)
            self.waypoint_index = 0
        elif self.current_route_idx != self.preferred_route_idx:
            # Return to preferred route
            self.current_route_idx = self.preferred_route_idx
            self.current_route = routes[self.current_route_idx]
            self.route_waypoints = self._get_route_waypoints(self.current_route)
            self.waypoint_index = 0

        if not self.route_waypoints:
            return find_path_to(grid, player_pos, grid.vault)

        # Follow waypoints along the route
        target = self.route_waypoints[self.waypoint_index] if self.waypoint_index < len(self.route_waypoints) else grid.vault
        if player_pos == target and self.waypoint_index < len(self.route_waypoints) - 1:
            self.waypoint_index += 1
            target = self.route_waypoints[self.waypoint_index]

        return find_path_to(grid, player_pos, target)


class RotatingBot(_RouteBasedBot):
    """Cycles through routes every 3 rounds, in a random order."""

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.route_order = []

    def _get_next_step(self, grid: Grid, player_pos: Position,
                       enemy_positions: List[Position]) -> Position:
        routes = self._get_routes(grid)
        if not routes:
            return find_path_to(grid, player_pos, grid.vault)

        # Initialize random route order on first round
        if not self.route_order:
            self.route_order = list(range(len(routes)))
            self.rng.shuffle(self.route_order)

        # Change route every 3 rounds (0-indexed: rounds 0-2 use first route, 3-5 use second, etc.)
        cycle_idx = ((self.round_num - 1) // 3) % len(self.route_order)
        self.current_route_idx = self.route_order[cycle_idx]

        if self.current_route != routes[self.current_route_idx]:
            self.current_route = routes[self.current_route_idx]
            self.route_waypoints = self._get_route_waypoints(self.current_route)
            self.waypoint_index = 0

        if not self.route_waypoints:
            return find_path_to(grid, player_pos, grid.vault)

        target = self.route_waypoints[self.waypoint_index] if self.waypoint_index < len(self.route_waypoints) else grid.vault
        if player_pos == target and self.waypoint_index < len(self.route_waypoints) - 1:
            self.waypoint_index += 1
            target = self.route_waypoints[self.waypoint_index]

        return find_path_to(grid, player_pos, target)


class RandomRouteBot(_RouteBasedBot):
    """Picks a new random route every round."""

    def _get_next_step(self, grid: Grid, player_pos: Position,
                       enemy_positions: List[Position]) -> Position:
        routes = self._get_routes(grid)
        if not routes:
            return find_path_to(grid, player_pos, grid.vault)

        # Pick random route each round
        self.current_route_idx = self.rng.randrange(len(routes))
        self.current_route = routes[self.current_route_idx]
        self.route_waypoints = self._get_route_waypoints(self.current_route)
        self.waypoint_index = 0

        if not self.route_waypoints:
            return find_path_to(grid, player_pos, grid.vault)

        target = self.route_waypoints[self.waypoint_index] if self.waypoint_index < len(self.route_waypoints) else grid.vault
        if player_pos == target and self.waypoint_index < len(self.route_waypoints) - 1:
            self.waypoint_index += 1
            target = self.route_waypoints[self.waypoint_index]

        return find_path_to(grid, player_pos, target)


class CounterBot(_RouteBasedBot):
    """Uses the route used least in the last 5 rounds."""

    def _get_next_step(self, grid: Grid, player_pos: Position,
                       enemy_positions: List[Position]) -> Position:
        routes = self._get_routes(grid)
        if not routes:
            return find_path_to(grid, player_pos, grid.vault)

        # Count route usage in history
        if not self._route_history:
            # First round - pick random
            self.current_route_idx = self.rng.randrange(len(routes))
        else:
            counts = Counter(self._route_history)
            # Find least used route(s)
            min_count = min(counts.get(i, 0) for i in range(len(routes)))
            least_used = [i for i in range(len(routes)) if counts.get(i, 0) == min_count]
            self.current_route_idx = self.rng.choice(least_used)

        if self.current_route != routes[self.current_route_idx]:
            self.current_route = routes[self.current_route_idx]
            self.route_waypoints = self._get_route_waypoints(self.current_route)
            self.waypoint_index = 0

        if not self.route_waypoints:
            return find_path_to(grid, player_pos, grid.vault)

        target = self.route_waypoints[self.waypoint_index] if self.waypoint_index < len(self.route_waypoints) else grid.vault
        if player_pos == target and self.waypoint_index < len(self.route_waypoints) - 1:
            self.waypoint_index += 1
            target = self.route_waypoints[self.waypoint_index]

        return find_path_to(grid, player_pos, target)

    def new_round(self):
        super().new_round()
        if hasattr(self, 'current_route_idx') and self.current_route_idx is not None:
            self._route_history.append(self.current_route_idx)


class ExplorerBot(_RouteBasedBot):
    """Visits 1-2 random waypoints first, then heads to V and E."""

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self._explorer_phase = 'explore'
        self._explorer_waypoints = []
        self._explorer_wp_index = 0
        self._num_explore_wps = 0

    def _get_next_step(self, grid: Grid, player_pos: Position,
                       enemy_positions: List[Position]) -> Position:
        routes = self._get_routes(grid)
        if not routes:
            return find_path_to(grid, player_pos, grid.vault)

        # Phase 1: Explore random waypoints
        if self._explorer_phase == 'explore':
            if not self._explorer_waypoints:
                # Pick a random route for exploration
                self.current_route_idx = self.rng.randrange(len(routes))
                self.current_route = routes[self.current_route_idx]
                # Get all walkable tiles on this route as potential waypoints
                route_tiles = [t for t in self.current_route if grid.is_walkable(t)]
                self._num_explore_wps = self.rng.randint(1, 2)
                if route_tiles:
                    self._explorer_waypoints = self.rng.sample(route_tiles,
                        min(self._num_explore_wps, len(route_tiles)))
                self._explorer_wp_index = 0
                self._explorer_phase = 'to_vault'

            if self._explorer_wp_index < len(self._explorer_waypoints):
                target = self._explorer_waypoints[self._explorer_wp_index]
                if player_pos == target:
                    self._explorer_wp_index += 1
                    if self._explorer_wp_index >= len(self._explorer_waypoints):
                        self._explorer_phase = 'to_vault'
                return find_path_to(grid, player_pos, target)
            else:
                self._explorer_phase = 'to_vault'

        # Phase 2: Go to vault
        if self._explorer_phase == 'to_vault':
            if player_pos == grid.vault:
                self._explorer_phase = 'to_exit'
            return find_path_to(grid, player_pos, grid.vault)

        # Phase 3: Go to exit
        if self._explorer_phase == 'to_exit':
            return find_path_to(grid, player_pos, grid.exit)

        return find_path_to(grid, player_pos, grid.vault)

    def new_round(self):
        super().new_round()
        self._explorer_phase = 'explore'
        self._explorer_waypoints = []
        self._explorer_wp_index = 0


# For compatibility
Policy = Callable[[Grid, Position, bool, List[Position]], Position]