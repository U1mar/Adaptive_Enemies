"""Game engine: runs rounds, manages player/enemies, tracks results."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional
import random

from grid import Grid, Position
from game.levels import Level, generate_level, get_level_config, get_num_levels
from game.enemy import Patroller, Flanker, EnemyState
from learning import Tracker, ZoneModel
from ai.costs import build_cost_map
from game.constants import DECAY_INTERVAL


@dataclass
class RoundResult:
    caught: bool
    got_key: bool
    escaped: bool
    turns: int
    path: List[Position]
    zones_visited: List[str]


Policy = Callable[[Grid, Position, bool, List[Position]], Position]


def build_patrol_routes(level: Level) -> List[List[Position]]:
    """Build patrol routes from the 3 S->V->E routes of the level.
    Each route becomes a patrol loop by connecting waypoints.
    """
    all_routes = []

    for route in level.routes:
        if len(route) < 2:
            continue

        # Sample 3-5 waypoints from this route
        num_waypoints = min(5, max(3, len(route) // 3))
        step = max(1, len(route) // num_waypoints)
        waypoints = [route[i] for i in range(0, len(route), step)][:num_waypoints]

        # Ensure we have the key points
        if level.start not in waypoints:
            waypoints.insert(0, level.start)
        if level.vault not in waypoints:
            waypoints.append(level.vault)
        if level.exit not in waypoints:
            waypoints.append(level.exit)

        # Build connected route using A* between waypoints
        from pathing import astar
        full_route = []
        for i in range(len(waypoints)):
            start = waypoints[i]
            end = waypoints[(i + 1) % len(waypoints)]  # Loop back to first
            path, _ = astar(level.grid, start, end)
            if path:
                if i == 0:
                    full_route.extend(path)
                else:
                    full_route.extend(path[1:])  # Skip duplicate

        if full_route:
            all_routes.append(full_route)

    return all_routes


def build_patrol_routes_legacy(grid: Grid) -> List[List[Position]]:
    """Build patrol routes using legacy corridor-based detection (for backward compatibility).
    Returns list of waypoint lists, one per corridor (west, middle, east).
    """
    corridors = grid.get_corridors()

    special_positions = {grid.start, grid.vault, grid.exit}

    # Junction rows where corridors connect horizontally - exclude from patrol
    junction_rows = {grid.start[1], grid.vault[1], grid.exit[1]}

    all_routes = []

    # Always create 3 routes (west, middle, east) for consistency
    zone_names = ['west', 'middle', 'east']

    for i, zone_name in enumerate(zone_names):
        candidates = []
        # Find corridor for this zone
        zone_corridor = None
        for corridor_start, corridor_end, z_name in corridors:
            if z_name == zone_name:
                zone_corridor = (corridor_start, corridor_end)
                break

        if zone_corridor:
            corridor_start, corridor_end = zone_corridor
            for y in range(1, grid.height - 1):
                if y in junction_rows:
                    continue  # Skip junction rows
                for x in range(corridor_start, corridor_end + 1):
                    pos = (x, y)
                    if grid.is_walkable(pos) and pos not in special_positions:
                        candidates.append(pos)

        # Select well-distributed waypoints for this corridor (corners of vertical segments)
        def select_waypoints(candidates: List[Position], count: int = 4) -> List[Position]:
            if not candidates:
                return []
            xs = [p[0] for p in candidates]
            ys = [p[1] for p in candidates]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Target points at corners of the vertical corridor segments
            # (avoiding junction rows which are excluded from candidates)
            target_points = [
                (min_x, min_y),  # top of upper segment
                (max_x, min_y),
                (max_x, max_y),  # bottom of lower segment
                (min_x, max_y),
            ]

            waypoints = []
            for tx, ty in target_points:
                best = min(candidates, key=lambda p: abs(p[0] - tx) + abs(p[1] - ty))
                if best not in waypoints:
                    waypoints.append(best)
            return waypoints[:count]

        waypoints = select_waypoints(candidates, 4)
        if waypoints:
            # Build connected route from waypoints using A*
            from pathing import astar
            full_route = []
            for j in range(len(waypoints)):
                start = waypoints[j]
                end = waypoints[(j + 1) % len(waypoints)]
                path, _ = astar(grid, start, end)
                if path:
                    if j == 0:
                        full_route.extend(path)
                    else:
                        full_route.extend(path[1:])
            if full_route:
                all_routes.append(full_route)

    return all_routes


def find_spawn_position(grid: Grid, min_distance: int = 8, exclude: List[Position] = None,
                        zone: str = None, rng: random.Random = None) -> Position:
    """Find a walkable position at least min_distance away from start by path distance.
    If zone specified, only consider positions in that zone.
    Uses the provided RNG for reproducible but varied spawns per round.
    If no tile satisfies min_distance, relaxes the distance step by step.
    """
    exclude = exclude or []
    rng = rng or random.Random()

    def collect_candidates(current_min_dist: int) -> List[Tuple[Position, int]]:
        """Collect all valid candidates at the given minimum distance."""
        cand = []
        if zone is not None:
            for y in range(1, grid.height - 1):
                for x in range(1, grid.width - 1):
                    pos = (x, y)
                    if grid.is_walkable(pos) and grid.zone_of(pos) == zone and pos not in exclude and pos != grid.start:
                        dist = grid.path_distance(grid.start, pos)
                        if dist is not None and dist >= current_min_dist:
                            cand.append((pos, dist))
        else:
            for y in range(1, grid.height - 1):
                for x in range(1, grid.width - 1):
                    pos = (x, y)
                    if grid.is_walkable(pos) and pos != grid.start and pos not in exclude:
                        dist = grid.path_distance(grid.start, pos)
                        if dist is not None and dist >= current_min_dist:
                            cand.append((pos, dist))
        return cand

    # Try with the requested min_distance, then relax step by step
    current_min_dist = min_distance
    while current_min_dist >= 0:
        candidates = collect_candidates(current_min_dist)
        if candidates:
            # Choose randomly from valid candidates using per-round RNG
            chosen = rng.choice(candidates)
            return chosen[0]
        # Relax distance and try again
        current_min_dist = max(0, current_min_dist - 2)

    # Fallback: any walkable position in zone not at start (distance 0)
    if zone:
        for y in range(1, grid.height - 1):
            for x in range(1, grid.width - 1):
                pos = (x, y)
                if grid.is_walkable(pos) and grid.zone_of(pos) == zone and pos != grid.start and pos not in exclude:
                    return pos
    # Last resort: any walkable not at start
    for y in range(1, grid.height - 1):
        for x in range(1, grid.width - 1):
            pos = (x, y)
            if grid.is_walkable(pos) and pos != grid.start and pos not in exclude:
                return pos
    return grid.start


def _get_per_round_rng(level_seed: int, round_number: int) -> random.Random:
    """Create deterministic RNG for a specific level seed and round number."""
    # Use hash of (level_seed, round_number) for reproducible but different spawns per round
    seed = hash((level_seed, round_number))
    return random.Random(seed)


def _sample_patrol_waypoints(route: List[Position], rng: random.Random, min_wp: int = 3, max_wp: int = 5) -> List[Position]:
    """Sample 3-5 random waypoints from a route for patrol."""
    if len(route) < 2:
        return route

    num_waypoints = rng.randint(min_wp, max_wp)
    # Ensure we include start, vault, exit if they're on this route
    # For now, just sample evenly spaced points with some randomness
    step = max(1, len(route) // num_waypoints)
    waypoints = []
    for i in range(num_waypoints):
        idx = min(i * step + rng.randint(-step // 2, step // 2), len(route) - 1)
        idx = max(0, idx)
        if route[idx] not in waypoints:
            waypoints.append(route[idx])

    # Ensure at least 3 waypoints
    if len(waypoints) < 3 and len(route) >= 3:
        waypoints = [route[0], route[len(route)//2], route[-1]]

    return waypoints


class Game:
    """Main game engine."""

    def __init__(self, *, level: int = 1, grid: Grid = None, alpha: float = None, seed: int = None):
        # Load level configuration
        if grid is not None:
            self.grid = grid
            self.level = 1
            config = get_level_config(self.level)
            # For backward compatibility with provided grid, generate a level from it
            self.level_obj = Level(
                grid=grid,
                start=grid.start,
                vault=grid.vault,
                exit=grid.exit,
                routes=[],  # Will be built from corridors if empty
                seed=seed or 0,
                archetype="legacy",
                level_number=1,
                chokepoints=[]
            )
        else:
            self.level = level
            self.level_obj = generate_level(level, seed if seed is not None else 0)
            self.grid = self.level_obj.grid
            config = get_level_config(level)

        # Use level config alpha if not explicitly provided
        self.alpha = alpha if alpha is not None else config.get("alpha", 1.5)
        self.tracker = Tracker(decay=config.get("decay_rate", 0.95))
        # Initialize ZoneModel with the grid's zone names (route-based)
        self.zone_model = ZoneModel(zones=self.grid.get_zone_names())

        # Decay interval from level config (default to constant)
        self.decay_interval = config.get("decay_interval", DECAY_INTERVAL)

        # Build patrol routes - use new route-based system if routes exist, else legacy corridor-based
        if self.level_obj.routes:
            self.patrol_routes = build_patrol_routes(self.level_obj)
        else:
            self.patrol_routes = build_patrol_routes_legacy(self.grid)

        self.enemies: List[Patroller | Flanker] = []
        self.got_key = False
        self.turn_count = 0
        self.last_zone = None
        self.last_player_pos = None
        self.level = level
        self.config = config
        self.round_number = 0  # Track round number for per-round RNG
        self._init_enemies()

    def _init_enemies(self) -> None:
        """Initialize patrollers and flankers based on level config.
        Spawns are randomized per round using seed (level_seed, round_number).
        """
        used_positions = []
        num_patrollers = self.config.get("num_patrollers", 2)
        num_flankers = self.config.get("num_flankers", 1)
        sight_radius = self.config.get("sight_radius", 8)

        # Get per-round RNG for this level and round
        rng = _get_per_round_rng(self.level_obj.seed, self.round_number)

        # Zone names from the grid (route-based or corridor-based)
        zone_names = self.grid.get_zone_names()

        # Create patrollers for each route/corridor
        for i in range(num_patrollers):
            if i < len(self.patrol_routes) and self.patrol_routes[i]:
                zone = zone_names[i] if i < len(zone_names) else zone_names[-1]
                spawn_pos = find_spawn_position(self.grid, min_distance=8, exclude=used_positions, zone=zone, rng=rng)

                # For procedural levels, sample new waypoints from the route
                if self.level_obj.routes:
                    route_idx = i % len(self.level_obj.routes)
                    waypoints = _sample_patrol_waypoints(self.level_obj.routes[route_idx], rng)
                else:
                    # For legacy, use the existing patrol route waypoints
                    waypoints = self.patrol_routes[i][::max(1, len(self.patrol_routes[i]) // 4)][:4]
                    if len(waypoints) < 2:
                        waypoints = [self.patrol_routes[i][0], self.patrol_routes[i][-1]]

                patroller = Patroller(waypoints, self.grid)
                patroller.position = spawn_pos
                patroller.sight_radius = sight_radius
                self.enemies.append(patroller)
                used_positions.append(spawn_pos)

        # Create flankers - they wait at chokepoints on the hottest route
        chokepoints = self.level_obj.chokepoints
        for i in range(num_flankers):
            # Only create flankers if we have patrol routes
            if not self.patrol_routes:
                break

            route_idx = min(num_patrollers + i, len(self.patrol_routes) - 1)
            if route_idx < len(self.patrol_routes) and self.patrol_routes[route_idx]:
                zone = zone_names[route_idx] if route_idx < len(zone_names) else zone_names[-1]
                spawn_pos = find_spawn_position(self.grid, min_distance=8, exclude=used_positions, zone=zone, rng=rng)
                flanker = Flanker(spawn_pos, self.patrol_routes[route_idx], self.grid)
                flanker.sight_radius = sight_radius
                # Set chokepoints for this flanker
                flanker.chokepoints = chokepoints
                self.enemies.append(flanker)
                used_positions.append(spawn_pos)
            elif self.patrol_routes:
                zone = zone_names[-1]
                spawn_pos = find_spawn_position(self.grid, min_distance=8, exclude=used_positions, zone=zone, rng=rng)
                flanker = Flanker(spawn_pos, self.patrol_routes[-1], self.grid)
                flanker.sight_radius = sight_radius
                flanker.chokepoints = chokepoints
                self.enemies.append(flanker)
                used_positions.append(spawn_pos)

    def step(self, player_move: Position) -> Tuple[bool, bool, bool]:
        """
        Advance exactly one turn.
        Returns (caught, got_key, escaped).
        Handles capture when:
        - Enemy moves onto player
        - Player moves onto enemy
        - They swap tiles (cross paths)
        """
        player_pos = player_move
        enemy_positions_before = [e.position for e in self.enemies]

        # Check: player moves onto enemy
        if player_pos in enemy_positions_before:
            return True, self.got_key, False

        # Record player position with Tracker
        self.tracker.record(player_pos)

        # Update ZoneModel with zone transitions
        current_zone = self.grid.zone_of(player_pos)
        if self.last_zone is not None and current_zone != self.last_zone:
            self.zone_model.record_transition(self.last_zone, current_zone)
        self.last_zone = current_zone

        # Decay tracker every decay_interval turns
        self.turn_count += 1
        if self.turn_count % self.decay_interval == 0:
            self.tracker.decay_counts()

        # Check for key pickup at V (vault) - persists once acquired
        if player_pos == self.grid.vault:
            self.got_key = True

        # Check for escape - at exit E with key
        escaped = player_pos == self.grid.exit and self.got_key
        if escaped:
            return False, self.got_key, True

        # Update enemies
        enemy_positions_after = []
        for enemy in self.enemies:
            enemy.update(self.grid, player_pos, self.tracker, self.zone_model, self.alpha)
            enemy_positions_after.append(enemy.position)

        # Check: enemy moves onto player
        if player_pos in enemy_positions_after:
            return True, self.got_key, False

        # Check for cross-path swap: player moved from A to B, enemy moved from B to A
        for i, before in enumerate(enemy_positions_before):
            after = enemy_positions_after[i]
            if before != after:  # Enemy moved
                if after == self.last_player_pos and before == player_pos:
                    # Swap detected
                    return True, self.got_key, False

        # Store current player position for next turn's swap detection
        self.last_player_pos = player_pos

        return False, self.got_key, escaped

    def play_round(self, policy: Policy, max_turns: int = None) -> RoundResult:
        """Run one round with the given policy. Returns RoundResult."""
        if max_turns is None:
            max_turns = self.config.get("max_turns", 500)
        player_pos = self.grid.start
        self.got_key = False
        self.turn_count = 0
        self.last_zone = self.grid.zone_of(player_pos)
        self.last_player_pos = player_pos
        path = [player_pos]
        zones_visited = [self.last_zone]

        for turn in range(max_turns):
            # Get next move from policy
            enemy_positions = [e.position for e in self.enemies]
            next_pos = policy(self.grid, player_pos, self.got_key, enemy_positions)

            # Validate move
            if not self.grid.is_walkable(next_pos):
                next_pos = player_pos  # Stay in place if invalid

            # Step the game
            caught, got_key, escaped = self.step(next_pos)

            player_pos = next_pos
            path.append(player_pos)
            zones_visited.append(self.grid.zone_of(player_pos))

            if caught:
                return RoundResult(True, got_key, False, turn + 1, path, zones_visited)
            if escaped:
                return RoundResult(False, got_key, True, turn + 1, path, zones_visited)

        # Max turns reached without escape or capture
        return RoundResult(False, self.got_key, False, max_turns, path, zones_visited)

    def reset_enemies(self) -> None:
        """Reset enemy positions for new round, but KEEP tracker and zone_model for learning persistence.
        Increments round number so new spawns are generated.
        """
        self.enemies.clear()
        # Do NOT reset tracker and zone_model - learning persists across rounds
        self.got_key = False
        self.turn_count = 0
        self.last_zone = None
        self.last_player_pos = None
        self.round_number += 1  # Increment round for new spawns
        self._init_enemies()