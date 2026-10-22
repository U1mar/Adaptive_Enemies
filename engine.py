"""Game engine: runs rounds, manages player/enemies, tracks results."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional
import copy

from grid import Grid, Position, DEFAULT_MAP
from enemy import Patroller, Flanker
from learning import Tracker, ZoneModel
from pathing import astar


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
    """Build three patrol routes for west, middle, east corridors."""
    routes = {'west': [], 'middle': [], 'east': []}
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.is_walkable((x, y)):
                zone = grid.zone_of((x, y))
                routes[zone].append((x, y))
    # Sort each route to form a loop (by y then x)
    for zone in routes:
        routes[zone].sort(key=lambda p: (p[1], p[0]))
    return routes['west'], routes['middle'], routes['east']


class Game:
    """Main game engine."""

    def __init__(self, grid: Grid = None, alpha: float = 0.5):
        self.grid = grid or Grid.from_ascii(DEFAULT_MAP)
        self.alpha = alpha
        self.tracker = Tracker()
        self.zone_model = ZoneModel()
        self.west_route, self.middle_route, self.east_route = build_patrol_routes(self.grid)
        self.enemies: List[Patroller | Flanker] = []
        self._init_enemies()

    def _init_enemies(self) -> None:
        """Initialize 2 patrollers and 1 flanker."""
        # Patroller 1: west route
        if self.west_route:
            self.enemies.append(Patroller(self.west_route[0], self.west_route))
        # Patroller 2: middle route
        if self.middle_route:
            self.enemies.append(Patroller(self.middle_route[0], self.middle_route))
        # Flanker: east route
        if self.east_route:
            self.enemies.append(Flanker(self.east_route[0], self.east_route, self.grid))

    def run_round(self, policy: Policy, max_turns: int = 500) -> RoundResult:
        """Run one round with the given policy. Returns RoundResult."""
        player_pos = self.grid.start
        got_key = False
        path = [player_pos]
        zones_visited = [self.grid.zone_of(player_pos)]
        last_zone = zones_visited[0]

        for turn in range(max_turns):
            # Record player position
            self.tracker.record(player_pos)
            current_zone = self.grid.zone_of(player_pos)
            if current_zone != last_zone:
                self.zone_model.record_transition(last_zone, current_zone)
                last_zone = current_zone
            zones_visited.append(current_zone)

            # Rebuild cost map every 5 turns
            if turn % 5 == 0 and turn > 0:
                self.tracker.decay_counts()

            # Get next move from policy
            enemy_positions = [e.position for e in self.enemies]
            next_pos = policy(self.grid, player_pos, got_key, enemy_positions)

            # Validate move
            if not self.grid.is_walkable(next_pos):
                next_pos = player_pos  # Stay in place if invalid

            player_pos = next_pos
            path.append(player_pos)

            # Check for key pickup
            if player_pos == self.grid.vault:
                got_key = True

            # Check for capture
            if player_pos in enemy_positions:
                return RoundResult(True, got_key, False, turn + 1, path, zones_visited)

            # Check for escape
            if player_pos == self.grid.exit and got_key:
                return RoundResult(False, got_key, True, turn + 1, path, zones_visited)

            # Update enemies
            for enemy in self.enemies:
                enemy.update(self.grid, player_pos, self.tracker, self.zone_model, self.alpha)

        return RoundResult(False, got_key, False, max_turns, path, zones_visited)

    def reset_enemies(self) -> None:
        """Reset enemy positions for new round."""
        self.enemies.clear()
        self._init_enemies()