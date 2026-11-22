"""Grid representation with walls, neighbors, and zone detection."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
from collections import deque

# Heatmap visualization gradient: space (empty) -> light -> medium -> dark -> full
HEATMAP_CHARS = ' ░▒▓█'

Position = Tuple[int, int]


@dataclass
class Grid:
    """2D grid with walls and walkable tiles."""
    width: int
    height: int
    walls: Set[Position]
    start: Position
    vault: Position
    exit: Position
    # New: routes for route-based zone detection (3 routes S->V->E)
    routes: List[List[Position]] = field(default_factory=list)
    # Cache for route-based zone mapping
    _route_zone_cache: Dict[Position, int] = field(default_factory=dict, init=False)

    def in_bounds(self, p: Position) -> bool:
        x, y = p
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, p: Position) -> bool:
        return p in self.walls

    def is_walkable(self, p: Position) -> bool:
        return self.in_bounds(p) and not self.is_wall(p)

    def neighbors(self, p: Position) -> List[Position]:
        """4-directional neighbors (up, down, left, right)."""
        x, y = p
        dirs = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [n for n in dirs if self.is_walkable(n)]

    def _detect_corridors(self) -> List[Tuple[int, int, str]]:
        """Detect vertical corridors by finding columns that are mostly walls.
        Returns list of (start_x, end_x, zone_name) for each corridor.
        Caches result for performance.
        """
        if hasattr(self, '_corridors'):
            return self._corridors

        # Find columns that are walls for most rows (vertical walls)
        wall_columns = []
        for x in range(self.width):
            wall_count = sum(1 for y in range(self.height) if (x, y) in self.walls)
            # If column is wall for > 50% of rows, it's a vertical wall
            if wall_count > self.height * 0.5:
                wall_columns.append(x)

        # Sort wall columns
        wall_columns.sort()

        # Filter out boundary walls (x=0 and x=width-1) as they are map edges, not corridor separators
        interior_walls = [x for x in wall_columns if 0 < x < self.width - 1]

        # Corridors are the walkable regions between vertical walls
        corridors = []
        prev_wall = 0  # Left boundary (wall at x=0)

        zone_names = ['west', 'middle', 'east', 'far_east', 'far_far_east', 'far_far_far_east']

        for i, wall_x in enumerate(interior_walls):
            # Corridor is between prev_wall and this wall
            corridor_start = prev_wall + 1
            corridor_end = wall_x - 1
            if corridor_start <= corridor_end:
                # Check if there are walkable tiles in this corridor
                has_walkable = any(
                    self.is_walkable((x, y))
                    for x in range(corridor_start, corridor_end + 1)
                    for y in range(self.height)
                )
                if has_walkable:
                    zone_name = zone_names[i] if i < len(zone_names) else f'zone_{i}'
                    corridors.append((corridor_start, corridor_end, zone_name))
            prev_wall = wall_x

        # Check corridor after last interior wall to right boundary
        corridor_start = prev_wall + 1
        corridor_end = self.width - 2  # Right boundary wall at width-1
        if corridor_start <= corridor_end:
            has_walkable = any(
                self.is_walkable((x, y))
                for x in range(corridor_start, corridor_end + 1)
                for y in range(self.height)
            )
            if has_walkable:
                zone_name = zone_names[len(corridors)] if len(corridors) < len(zone_names) else f'zone_{len(corridors)}'
                corridors.append((corridor_start, corridor_end, zone_name))

        self._corridors = corridors
        return corridors

    def zone_of(self, p: Position) -> str:
        """Return zone name based on nearest route (route_0, route_1, route_2).
        Falls back to corridor-based detection if no routes are defined.
        """
        # Use route-based zones if routes are available.
        # Computed with ONE multi-source BFS per route (instead of a BFS for every tile/route-tile
        # pair), then cached. Same result as before: nearest route by path distance, ties -> lowest index.
        if self.routes:
            zmap = self._route_zone_map()
            if p in zmap:
                return f'route_{zmap[p]}'

        # Fallback to corridor-based detection for legacy maps
        x, _ = p
        corridors = self._detect_corridors()
        for start_x, end_x, name in corridors:
            if start_x <= x <= end_x:
                return name
        # Fallback: assign to nearest corridor
        if corridors:
            if x < corridors[0][0]:
                return corridors[0][2]
            return corridors[-1][2]
        # Ultimate fallback
        third = self.width // 3
        if x < third:
            return 'west'
        elif x < 2 * third:
            return 'middle'
        return 'east'

    def _route_zone_map(self) -> Dict[Position, int]:
        """Map walkable tile -> index of the nearest route (by path distance). Built lazily."""
        sig = tuple(len(r) for r in self.routes)
        cached = getattr(self, '_zone_map_cache', None)
        if cached is not None and cached[0] == sig:
            return cached[1]
        best: Dict[Position, Tuple[int, int]] = {}
        for i, route in enumerate(self.routes):
            sources = [t for t in route if self.is_walkable(t)]
            if not sources:
                continue
            dist = {t: 0 for t in sources}
            queue = deque(sources)
            while queue:
                cur = queue.popleft()
                d = dist[cur] + 1
                for nb in self.neighbors(cur):
                    if nb not in dist:
                        dist[nb] = d
                        queue.append(nb)
            for pos, d in dist.items():
                if pos not in best or d < best[pos][0]:   # strict < : earliest route wins ties
                    best[pos] = (d, i)
        zmap = {pos: v[1] for pos, v in best.items()}
        self._zone_map_cache = (sig, zmap)
        return zmap

    def distance_map(self, source: Position) -> Dict[Position, int]:
        """Path length from `source` to every reachable tile (one BFS). Use instead of calling
        path_distance() once per tile."""
        dist: Dict[Position, int] = {}
        if not self.is_walkable(source):
            return dist
        dist[source] = 0
        queue = deque([source])
        while queue:
            cur = queue.popleft()
            d = dist[cur] + 1
            for nb in self.neighbors(cur):
                if nb not in dist:
                    dist[nb] = d
                    queue.append(nb)
        return dist

    def get_corridors(self) -> List[Tuple[int, int, str]]:
        """Return list of (start_x, end_x, zone_name) for each corridor."""
        return self._detect_corridors()

    def get_zone_names(self) -> List[str]:
        """Return list of zone names in left-to-right order.
        Uses route-based zones (route_0, route_1, route_2) if routes available,
        otherwise uses corridor-based zones.
        """
        if self.routes:
            return [f'route_{i}' for i in range(len(self.routes))]
        return [name for _, _, name in self._detect_corridors()]

    def path_distance(self, start: Position, goal: Position) -> Optional[int]:
        """Return shortest path length between two positions using BFS, or None if unreachable."""
        if start == goal:
            return 0
        if not self.is_walkable(start) or not self.is_walkable(goal):
            return None

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current, dist = queue.popleft()
            if current == goal:
                return dist
            for neighbor in self.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return None

    @classmethod
    def from_ascii(cls, ascii_map: str) -> Grid:
        """Create Grid from ASCII map.
        # = wall, . = floor, S = start, V = vault, E = exit
        """
        lines = ascii_map.strip().split('\n')
        height = len(lines)
        width = len(lines[0])
        walls = set()
        start = vault = exit_pos = (0, 0)

        for y, line in enumerate(lines):
            # Only process up to width characters to handle inconsistent line lengths
            for x, ch in enumerate(line[:width]):
                pos = (x, y)
                if ch == '#':
                    walls.add(pos)
                elif ch == 'S':
                    start = pos
                elif ch == 'V':
                    vault = pos
                elif ch == 'E':
                    exit_pos = pos

        return cls(width, height, walls, start, vault, exit_pos)

    def to_ascii(self, player: Position = None, enemies: List[Position] = None,
                 show_heatmap: bool = False, heatmap: dict = None) -> str:
        """Render grid as ASCII string."""
        enemy_set = set(enemies or [])
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                if pos in self.walls:
                    row.append('#')
                elif player and pos == player:
                    row.append('@')
                elif pos in enemy_set:
                    row.append('X')
                elif pos == self.start:
                    row.append('S')
                elif pos == self.vault:
                    row.append('V')
                elif pos == self.exit:
                    row.append('E')
                elif show_heatmap and heatmap and pos in heatmap:
                    val = heatmap[pos]
                    # Map value [0, 1] to HEATMAP_CHARS indices [0, 4]
                    idx = int(val * (len(HEATMAP_CHARS) - 1) + 0.5)
                    idx = max(0, min(idx, len(HEATMAP_CHARS) - 1))
                    row.append(HEATMAP_CHARS[idx])
                else:
                    row.append('.')
            lines.append(''.join(row))
        return '\n'.join(lines)


# 30x15 map with three separate corridors (west, middle, east)
# Corridors only connect at S (row 1), V (row 7), E (row 13)
# Vertical walls at x=11 and x=22 with openings at rows 1, 7, 13
# Layout: west corridor x=1-10, middle x=12-21, east x=23-28
# All lines must be exactly 30 characters
DEFAULT_MAP = (
    "##############################\n"  # row 0: 30 chars
    "#S.........#..........#......#\n"  # row 1: S at (1,1), walls at 11,22
    "#..........#..........#......#\n"  # row 2
    "#..........#..........#......#\n"  # row 3
    "#..........#..........#......#\n"  # row 4
    "#..........#..........#......#\n"  # row 5
    "#..........#..........#......#\n"  # row 6
    "#..........#......V....#......#\n"  # row 7: V at (18,7)
    "#..........#..........#......#\n"  # row 8
    "#..........#..........#......#\n"  # row 9
    "#..........#..........#......#\n"  # row 10
    "#..........#..........#......#\n"  # row 11
    "#..........#..........#......#\n"  # row 12
    "#............................E#\n"  # row 13: E at (29,13)
    "##############################"   # row 14
)