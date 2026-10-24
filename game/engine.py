"""Game engine: runs rounds, manages player/enemies, tracks results."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional

from grid import Grid, Position, DEFAULT_MAP
from game.enemy import Patroller, Flanker, EnemyState
from learning import Tracker, ZoneModel
from ai.costs import build_cost_map


@dataclass
class RoundResult:
    caught: bool
    got_key: bool
    escaped: bool
    turns: int
    path: List[Position]
    zones_visited: List[str]


Policy = Callable[[Grid, Position, bool, List[Position]], Position]


def build_patrol_routes(grid: Grid) -> Tuple[List[Position], List[Position], List[Position]]:
    """Build three patrol routes as waypoints for west, middle, east corridors.
    Returns waypoints that form connected loops via A* pathfinding."""
    # Define strategic waypoints for each zone - these will be connected by A*
    routes = {'west': [], 'middle': [], 'east': []}
    third = grid.width // 3
    special_positions = {grid.start, grid.vault, grid.exit}

    # West zone waypoints (left third)
    west_candidates = []
    for y in range(1, grid.height - 1):
        for x in range(1, third - 1):
            pos = (x, y)
            if grid.is_walkable(pos) and pos not in special_positions:
                west_candidates.append(pos)

    # Middle zone waypoints
    middle_candidates = []
    for y in range(1, grid.height - 1):
        for x in range(third, 2 * third - 1):
            pos = (x, y)
            if grid.is_walkable(pos) and pos not in special_positions:
                middle_candidates.append(pos)

    # East zone waypoints
    east_candidates = []
    for y in range(1, grid.height - 1):
        for x in range(2 * third, grid.width - 1):
            pos = (x, y)
            if grid.is_walkable(pos) and pos not in special_positions:
                east_candidates.append(pos)

    # Select well-distributed waypoints for each zone (corners and center-ish)
    def select_waypoints(candidates: List[Position], zone: str, count: int = 4) -> List[Position]:
        if not candidates:
            return []
        # Pick waypoints that are far apart - use corners and center of zone
        xs = [p[0] for p in candidates]
        ys = [p[1] for p in candidates]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        target_points = [
            (min_x, min_y),  # top-left
            (max_x, min_y),  # top-right
            (max_x, max_y),  # bottom-right
            (min_x, max_y),  # bottom-left
        ]

        waypoints = []
        for tx, ty in target_points:
            # Find closest walkable candidate
            best = min(candidates, key=lambda p: abs(p[0] - tx) + abs(p[1] - ty))
            if best not in waypoints:
                waypoints.append(best)
        return waypoints[:count]

    routes['west'] = select_waypoints(west_candidates, 'west', 4)
    routes['middle'] = select_waypoints(middle_candidates, 'middle', 4)
    routes['east'] = select_waypoints(east_candidates, 'east', 4)

    return routes['west'], routes['middle'], routes['east']


def find_spawn_position(grid: Grid, min_distance: int = 8, exclude: List[Position] = None) -> Position:
    """Find a walkable position at least min_distance away from start by path distance."""
    exclude = exclude or []
    candidates = []
    for y in range(1, grid.height - 1):
        for x in range(1, grid.width - 1):
            pos = (x, y)
            if grid.is_walkable(pos) and pos not in exclude and pos != grid.start:
                dist = grid.path_distance(grid.start, pos)
                if dist is not None and dist >= min_distance:
                    candidates.append((pos, dist))

    if not candidates:
        # Fallback: any walkable position not at start
        for y in range(1, grid.height - 1):
            for x in range(1, grid.width - 1):
                pos = (x, y)
                if grid.is_walkable(pos) and pos != grid.start and pos not in exclude:
                    return pos
        return grid.start

    # Prefer positions with good distance but not too far
    candidates.sort(key=lambda c: c[1])
    return candidates[0][0]


class Game:
    """Main game engine."""

    def __init__(self, grid: Grid = None, alpha: float = 3.0):
        self.grid = grid or Grid.from_ascii(DEFAULT_MAP)
        self.alpha = alpha
        self.tracker = Tracker()
        self.zone_model = ZoneModel()
        self.west_route, self.middle_route, self.east_route = build_patrol_routes(self.grid)
        self.enemies: List[Patroller | Flanker] = []
        self.got_key = False
        self.turn_count = 0
        self.last_zone = None
        self.last_player_pos = None
        self._init_enemies()

    def _init_enemies(self) -> None:
        """Initialize 2 patrollers and 1 flanker, spawning away from player start."""
        used_positions = []

        # Patroller 1: west route
        if self.west_route:
            spawn_pos = find_spawn_position(self.grid, min_distance=8, exclude=used_positions)
            self.enemies.append(Patroller(self.west_route, self.grid))
            self.enemies[-1].position = spawn_pos
            used_positions.append(spawn_pos)

        # Patroller 2: middle route
        if self.middle_route:
            spawn_pos = find_spawn_position(self.grid, min_distance=8, exclude=used_positions)
            self.enemies.append(Patroller(self.middle_route, self.grid))
            self.enemies[-1].position = spawn_pos
            used_positions.append(spawn_pos)

        # Flanker: east route
        if self.east_route:
            spawn_pos = find_spawn_position(self.grid, min_distance=8, exclude=used_positions)
            self.enemies.append(Flanker(spawn_pos, self.east_route, self.grid))
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

        # Decay tracker every 5 turns
        self.turn_count += 1
        if self.turn_count % 5 == 0:
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

    def play_round(self, policy: Policy, max_turns: int = 500) -> RoundResult:
        """Run one round with the given policy. Returns RoundResult."""
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
        """Reset enemy positions for new round."""
        self.enemies.clear()
        self.tracker = Tracker()
        self.zone_model = ZoneModel()
        self.got_key = False
        self.turn_count = 0
        self.last_zone = None
        self.last_player_pos = None
        self._init_enemies()