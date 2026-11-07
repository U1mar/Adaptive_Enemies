"""Procedural level generation with multiple archetypes."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
from collections import deque
import random
import hashlib

from grid import Grid, Position

Position = Tuple[int, int]


@dataclass
class Level:
    """Generated level with all metadata."""
    grid: Grid
    start: Position
    vault: Position
    exit: Position
    routes: List[List[Position]]  # 3 distinct S->V->E routes
    seed: int
    archetype: str
    level_number: int
    chokepoints: List[Position]  # tiles on shortest route with degree <= 2


# Archetype selection by level number
ARCHETYPES = [
    "rooms_and_corridors",  # level 1
    "cave",                  # level 2
    "loop_maze",             # level 3
    "three_lane",            # level 4
]


def get_archetype_for_level(level_number: int) -> str:
    """Get archetype for a level number (1-indexed, cycles through archetypes)."""
    return ARCHETYPES[(level_number - 1) % len(ARCHETYPES)]


def get_map_size_for_level(level_number: int) -> Tuple[int, int]:
    """Get map dimensions for a level. Grows from 30x15 to 50x25."""
    # Level 1: 30x15, Level 4: 50x25, interpolate
    if level_number <= 1:
        return 30, 15
    elif level_number >= 4:
        return 50, 25
    else:
        # Linear interpolation
        t = (level_number - 1) / 3
        width = int(30 + t * 20)
        height = int(15 + t * 10)
        return width, height


def _make_rng(seed: int) -> random.Random:
    """Create isolated RNG from seed."""
    return random.Random(seed)


def _hash_seed(*parts: int) -> int:
    """Create deterministic seed from multiple parts."""
    h = hashlib.md5()
    for p in parts:
        h.update(str(p).encode())
    return int(h.hexdigest()[:8], 16)


# ============================================================
# ARCHETYPE A: Rooms and Corridors
# ============================================================

def _generate_rooms_and_corridors(rng: random.Random, width: int, height: int) -> Tuple[Set[Position], Position, Position, Position]:
    """Generate rooms connected by corridors. Returns (walls, start, vault, exit)."""
    walls = set()

    # Fill with walls initially
    for y in range(height):
        for x in range(width):
            walls.add((x, y))

    # Carve rooms
    num_rooms = rng.randint(4, 7)
    rooms = []  # (x, y, w, h)

    for _ in range(num_rooms):
        for attempt in range(50):
            w = rng.randint(4, 8)
            h = rng.randint(3, 6)
            x = rng.randint(1, width - w - 1)
            y = rng.randint(1, height - h - 1)

            # Check overlap with existing rooms
            overlap = False
            for rx, ry, rw, rh in rooms:
                if not (x + w + 1 < rx or rx + rw + 1 < x or y + h + 1 < ry or ry + rh + 1 < y):
                    overlap = True
                    break

            if not overlap:
                rooms.append((x, y, w, h))
                # Carve room
                for ry in range(y, y + h):
                    for rx in range(x, x + w):
                        walls.discard((rx, ry))
                break

    # Connect rooms with corridors (minimum spanning tree style)
    if len(rooms) >= 2:
        # Connect each room to the nearest other room
        for i in range(len(rooms) - 1):
            x1, y1, w1, h1 = rooms[i]
            x2, y2, w2, h2 = rooms[i + 1]
            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2

            # L-shaped corridor
            if rng.random() < 0.5:
                # Horizontal then vertical
                for x in range(min(cx1, cx2), max(cx1, cx2) + 1):
                    walls.discard((x, cy1))
                for y in range(min(cy1, cy2), max(cy1, cy2) + 1):
                    walls.discard((cx2, y))
            else:
                # Vertical then horizontal
                for y in range(min(cy1, cy2), max(cy1, cy2) + 1):
                    walls.discard((cx1, y))
                for x in range(min(cx1, cx2), max(cx1, cx2) + 1):
                    walls.discard((x, cy2))

    # Add 4-6 extra loop corridors (increased for more route diversity)
    extra_corridors = rng.randint(4, 6)
    for _ in range(extra_corridors):
        if len(rooms) < 2:
            break
        r1 = rng.choice(rooms)
        r2 = rng.choice(rooms)
        if r1 == r2:
            continue
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2

        if rng.random() < 0.5:
            for x in range(min(cx1, cx2), max(cx1, cx2) + 1):
                walls.discard((x, cy1))
            for y in range(min(cy1, cy2), max(cy1, cy2) + 1):
                walls.discard((cx2, y))
        else:
            for y in range(min(cy1, cy2), max(cy1, cy2) + 1):
                walls.discard((cx1, y))
            for x in range(min(cx1, cx2), max(cx1, cx2) + 1):
                walls.discard((x, cy2))

    # Add some random connectors between nearby rooms
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if rng.random() < 0.3:  # 30% chance for extra connection
                x1, y1, w1, h1 = rooms[i]
                x2, y2, w2, h2 = rooms[j]
                cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
                cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
                # Direct connection
                if rng.random() < 0.5:
                    for x in range(min(cx1, cx2), max(cx1, cx2) + 1):
                        walls.discard((x, cy1))
                    for y in range(min(cy1, cy2), max(cy1, cy2) + 1):
                        walls.discard((cx2, y))
                else:
                    for y in range(min(cy1, cy2), max(cy1, cy2) + 1):
                        walls.discard((cx1, y))
                    for x in range(min(cx1, cx2), max(cx1, cx2) + 1):
                        walls.discard((x, cy2))

    # Ensure boundary walls
    for x in range(width):
        walls.add((x, 0))
        walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y))
        walls.add((width - 1, y))

    # Pick S, V, E from walkable tiles in different rooms
    walkable = [(x, y) for y in range(height) for x in range(width) if (x, y) not in walls]
    if len(walkable) < 3:
        raise ValueError("Not enough walkable space")

    # Pick start from first room area, vault from middle, exit from last
    start = rng.choice(walkable[:len(walkable)//3])
    vault = rng.choice(walkable[len(walkable)//3:2*len(walkable)//3])
    exit_pos = rng.choice(walkable[2*len(walkable)//3:])

    return walls, start, vault, exit_pos


# ============================================================
# ARCHETYPE B: Cave (Cellular Automata)
# ============================================================

def _generate_cave(rng: random.Random, width: int, height: int) -> Tuple[Set[Position], Position, Position, Position]:
    """Generate cave using cellular automata. Returns (walls, start, vault, exit)."""
    walls = set()

    # Initial random fill (45% walls)
    for y in range(height):
        for x in range(width):
            if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                walls.add((x, y))
            elif rng.random() < 0.45:
                walls.add((x, y))

    # Cellular automata iterations
    for _ in range(5):
        new_walls = set()
        for y in range(height):
            for x in range(width):
                if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                    new_walls.add((x, y))
                    continue

                wall_neighbors = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if (nx, ny) in walls:
                            wall_neighbors += 1

                if wall_neighbors >= 5:
                    new_walls.add((x, y))
        walls = new_walls

    # Find connected components, keep only largest
    walkable = [(x, y) for y in range(height) for x in range(width) if (x, y) not in walls]
    if not walkable:
        raise ValueError("No walkable space")

    # BFS to find connected components
    visited = set()
    components = []
    for pos in walkable:
        if pos in visited:
            continue
        component = []
        queue = deque([pos])
        visited.add(pos)
        while queue:
            curr = queue.popleft()
            component.append(curr)
            for n in _get_neighbors(curr, walls, width, height):
                if n not in visited:
                    visited.add(n)
                    queue.append(n)
        components.append(component)

    # Keep only largest component
    largest = max(components, key=len)
    new_walls = set(walls)
    for pos in walkable:
        if pos not in largest:
            new_walls.add(pos)
    walls = new_walls

    # Pick S, V, E from far apart positions in largest component
    component = largest
    if len(component) < 3:
        raise ValueError("Largest component too small")

    # Sort by distance from corner to get spread
    start = component[0]
    vault = max(component, key=lambda p: _manhattan(p, start))
    exit_pos = max(component, key=lambda p: _manhattan(p, vault) + _manhattan(p, start))

    return walls, start, vault, exit_pos


def _get_neighbors(pos: Position, walls: Set[Position], width: int, height: int) -> List[Position]:
    x, y = pos
    dirs = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    return [n for n in dirs if 0 <= n[0] < width and 0 <= n[1] < height and n not in walls]


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ============================================================
# ARCHETYPE C: Loop Maze (Perfect Maze + 15% wall removal)
# ============================================================

def _generate_loop_maze(rng: random.Random, width: int, height: int) -> Tuple[Set[Position], Position, Position, Position]:
    """Generate perfect maze using randomized Kruskal, then remove 15% interior walls."""
    # Only odd dimensions work well for maze
    maze_w = width if width % 2 == 1 else width - 1
    maze_h = height if height % 2 == 1 else height - 1

    # Start with all walls
    walls = set()
    for y in range(height):
        for x in range(width):
            walls.add((x, y))

    # Maze cells are at odd coordinates
    cells = [(x, y) for y in range(1, maze_h, 2) for x in range(1, maze_w, 2)]

    # Kruskal's algorithm
    edges = []
    for cx, cy in cells:
        for dx, dy in [(2, 0), (0, 2)]:
            nx, ny = cx + dx, cy + dy
            if nx < maze_w and ny < maze_h:
                edges.append(((cx, cy), (nx, ny), (cx + dx//2, cy + dy//2)))

    rng.shuffle(edges)

    parent = {cell: cell for cell in cells}

    def find(cell):
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            return True
        return False

    for (c1, c2, wall) in edges:
        if union(c1, c2):
            walls.discard(c1)
            walls.discard(c2)
            walls.discard(wall)

    # Carve all cells
    for cx, cy in cells:
        walls.discard((cx, cy))

    # Remove ~25% of interior walls to create loops (increased for more route diversity)
    interior_walls = [(x, y) for x in range(1, width-1) for y in range(1, height-1)
                      if (x, y) in walls and (x % 2 == 0 or y % 2 == 0)]
    num_to_remove = max(1, int(len(interior_walls) * 0.25))
    for wall in rng.sample(interior_walls, min(num_to_remove, len(interior_walls))):
        walls.discard(wall)

    # Ensure boundary walls
    for x in range(width):
        walls.add((x, 0))
        walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y))
        walls.add((width - 1, y))

    # Pick S, V, E from walkable
    walkable = [(x, y) for y in range(height) for x in range(width) if (x, y) not in walls]
    if len(walkable) < 3:
        raise ValueError("Not enough walkable space")

    start = rng.choice(walkable)
    vault = max(walkable, key=lambda p: _manhattan(p, start))
    exit_pos = max(walkable, key=lambda p: _manhattan(p, vault))

    return walls, start, vault, exit_pos


# ============================================================
# ARCHETYPE D: Three-Lane
# ============================================================

def _generate_three_lane(rng: random.Random, width: int, height: int) -> Tuple[Set[Position], Position, Position, Position]:
    """Generate three vertical lanes with cross-connections at random positions."""
    walls = set()

    # Three vertical corridors
    corridor_width = max(3, width // 10)
    lane_starts = [1, width // 3, 2 * width // 3]

    # Fill with walls
    for y in range(height):
        for x in range(width):
            walls.add((x, y))

    # Carve three main vertical lanes
    for lane_x in lane_starts:
        for y in range(1, height - 1):
            for dx in range(corridor_width):
                if lane_x + dx < width - 1:
                    walls.discard((lane_x + dx, y))

    # Add horizontal cross-connections at 5-8 random rows (increased for more route diversity)
    num_crossings = rng.randint(5, 8)
    crossing_rows = rng.sample(range(2, height - 2), num_crossings)

    for cy in crossing_rows:
        # Connect all three lanes horizontally
        for x in range(lane_starts[0], lane_starts[2] + corridor_width):
            if 0 < x < width - 1:
                walls.discard((x, cy))

    # Add some diagonal connections for even more route options
    for _ in range(rng.randint(2, 4)):
        cy = rng.randint(2, height - 3)
        cx = rng.randint(lane_starts[0] + corridor_width, lane_starts[2] - 1)
        if 0 < cx < width - 1 and 0 < cy < height - 1:
            walls.discard((cx, cy))
            if cx + 1 < width - 1:
                walls.discard((cx + 1, cy))
            if cy + 1 < height - 1:
                walls.discard((cx, cy + 1))

    # Ensure boundary walls
    for x in range(width):
        walls.add((x, 0))
        walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y))
        walls.add((width - 1, y))

    # Pick S from left lane, V from middle lane area, E from right lane
    left_walkable = [(x, y) for y in range(1, height-1) for x in range(lane_starts[0], lane_starts[0] + corridor_width) if (x, y) not in walls]
    mid_walkable = [(x, y) for y in range(1, height-1) for x in range(lane_starts[1], lane_starts[1] + corridor_width) if (x, y) not in walls]
    right_walkable = [(x, y) for y in range(1, height-1) for x in range(lane_starts[2], lane_starts[2] + corridor_width) if (x, y) not in walls]

    start = rng.choice(left_walkable) if left_walkable else rng.choice([p for p in [(1,1)] if p not in walls])
    vault = rng.choice(mid_walkable) if mid_walkable else rng.choice([p for p in [(width//2, height//2)] if p not in walls])
    exit_pos = rng.choice(right_walkable) if right_walkable else rng.choice([p for p in [(width-2, height-2)] if p not in walls])

    return walls, start, vault, exit_pos


# ============================================================
# VALIDATION
# ============================================================

def _validate_level(grid: Grid, start: Position, vault: Position, exit_pos: Position, routes: List[List[Position]]) -> bool:
    """Validate level meets all criteria."""
    # 1. S, V, E mutually reachable
    if grid.path_distance(start, vault) is None:
        return False
    if grid.path_distance(vault, exit_pos) is None:
        return False
    if grid.path_distance(start, exit_pos) is None:
        return False

    # 2. At least 3 routes with <40% overlap
    if len(routes) < 3:
        return False

    # Check pairwise overlap
    route_tiles = [set(r) for r in routes]
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            overlap = len(route_tiles[i] & route_tiles[j])
            total = len(route_tiles[i] | route_tiles[j])
            if total > 0 and overlap / total >= 0.4:
                return False

    # 3. At least 2 chokepoints on shortest route
    shortest = min(routes, key=len)
    chokepoints = 0
    for i, tile in enumerate(shortest):
        if i == 0 or i == len(shortest) - 1:
            continue
        neighbors = [n for n in grid.neighbors(tile) if n in shortest]
        if len(neighbors) <= 2:
            chokepoints += 1
    if chokepoints < 2:
        return False

    # 4. V and E far apart (at least 1/3 of map diagonal)
    min_dist = (grid.width + grid.height) // 3
    if grid.path_distance(vault, exit_pos) is not None and grid.path_distance(vault, exit_pos) < min_dist:
        return False

    return True


def _find_routes(grid: Grid, start: Position, vault: Position, exit_pos: Position) -> List[List[Position]]:
    """Find up to 3 distinct S->V->E routes using cost-based diversification.
    Uses A* with increasing cost penalties for previously used tiles to find alternative routes.
    """
    from pathing import astar
    routes = []
    # Track how many times each tile has been used in previous routes
    tile_usage: Dict[Position, int] = {}

    for attempt in range(10):  # Try up to 10 times to find 3 routes
        if len(routes) >= 3:
            break

        # Build cost map: tiles used more often get higher cost
        cost_map = {}
        for tile, count in tile_usage.items():
            # Penalty increases with usage: 1.0, 3.0, 5.0, etc.
            cost_map[tile] = 1.0 + count * 2.0

        # Find path from start to vault
        path1, _ = astar(grid, start, vault, cost_map if cost_map else None)
        if not path1:
            break

        # Find path from vault to exit
        path2, _ = astar(grid, vault, exit_pos, cost_map if cost_map else None)
        if not path2:
            break

        # Combine (avoid duplicate vault)
        full_path = path1 + path2[1:]

        # Check if this route is sufficiently different from existing routes
        if not _is_route_distinct(full_path, routes, threshold=0.4):
            # If too similar, increase usage of overlapping tiles and retry
            for tile in full_path[1:-1]:
                if tile not in (start, vault, exit_pos):
                    tile_usage[tile] = tile_usage.get(tile, 0) + 1
            continue

        routes.append(full_path)

        # Increase usage for tiles in this route
        for tile in full_path[1:-1]:
            if tile not in (start, vault, exit_pos):
                tile_usage[tile] = tile_usage.get(tile, 0) + 1

    return routes


def _is_route_distinct(new_route: List[Position], existing_routes: List[List[Position]], threshold: float = 0.4) -> bool:
    """Check if a route is sufficiently distinct from existing routes (overlap < threshold)."""
    if not existing_routes:
        return True

    new_set = set(new_route)
    for existing in existing_routes:
        existing_set = set(existing)
        overlap = len(new_set & existing_set)
        total = len(new_set | existing_set)
        if total > 0 and overlap / total >= threshold:
            return False
    return True


def _bfs_path(grid: Grid, start: Position, goal: Position, blocked: Set[Position]) -> Optional[List[Position]]:
    """BFS pathfinding avoiding blocked tiles (legacy fallback)."""
    if start == goal:
        return [start]
    if not grid.is_walkable(start) or not grid.is_walkable(goal):
        return None

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path
        for neighbor in grid.neighbors(current):
            if neighbor not in visited and neighbor not in blocked:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


def _find_chokepoints(grid: Grid, route: List[Position]) -> List[Position]:
    """Find chokepoints on a route (tiles with <=2 neighbors on the route)."""
    route_set = set(route)
    chokepoints = []
    for i, tile in enumerate(route):
        if i == 0 or i == len(route) - 1:
            continue
        neighbors = [n for n in grid.neighbors(tile) if n in route_set]
        if len(neighbors) <= 2:
            chokepoints.append(tile)
    return chokepoints


# ============================================================
# MAIN GENERATION FUNCTION
# ============================================================

def generate_level(level_number: int, seed: int) -> Level:
    """
    Generate a level with the given level number and seed.
    Retries with seed+1 up to 50 times until validation passes.
    """
    archetype = get_archetype_for_level(level_number)
    width, height = get_map_size_for_level(level_number)

    for attempt in range(50):
        try_seed = seed + attempt
        rng = _make_rng(try_seed)

        # Generate map based on archetype
        if archetype == "rooms_and_corridors":
            walls, start, vault, exit_pos = _generate_rooms_and_corridors(rng, width, height)
        elif archetype == "cave":
            walls, start, vault, exit_pos = _generate_cave(rng, width, height)
        elif archetype == "loop_maze":
            walls, start, vault, exit_pos = _generate_loop_maze(rng, width, height)
        elif archetype == "three_lane":
            walls, start, vault, exit_pos = _generate_three_lane(rng, width, height)
        else:
            raise ValueError(f"Unknown archetype: {archetype}")

        # Create grid
        grid = Grid(width, height, walls, start, vault, exit_pos)

        # Find routes
        routes = _find_routes(grid, start, vault, exit_pos)

        # Validate
        if _validate_level(grid, start, vault, exit_pos, routes):
            # Set routes on grid for route-based zone detection
            grid.routes = routes

            # Find chokepoints on shortest route
            shortest = min(routes, key=len)
            chokepoints = _find_chokepoints(grid, shortest)

            return Level(
                grid=grid,
                start=start,
                vault=vault,
                exit=exit_pos,
                routes=routes,
                seed=try_seed,
                archetype=archetype,
                level_number=level_number,
                chokepoints=chokepoints
            )

    raise RuntimeError(f"Failed to generate valid level {level_number} after 50 attempts")


# ============================================================
# COMPATIBILITY FUNCTIONS (for existing code)
# ============================================================

def get_level_config(level: int) -> dict:
    """Get difficulty config for a level (legacy compatibility)."""
    configs = {
        1: {"num_patrollers": 2, "num_flankers": 1, "alpha": 1.5, "sight_radius": 8, "decay_rate": 0.95, "decay_interval": 5, "max_turns": 500},
        2: {"num_patrollers": 3, "num_flankers": 1, "alpha": 2.0, "sight_radius": 8, "decay_rate": 0.92, "decay_interval": 5, "max_turns": 600},
        3: {"num_patrollers": 3, "num_flankers": 2, "alpha": 2.5, "sight_radius": 9, "decay_rate": 0.90, "decay_interval": 4, "max_turns": 800},
        4: {"num_patrollers": 2, "num_flankers": 2, "alpha": 2.25, "sight_radius": 7, "decay_rate": 0.93, "decay_interval": 5, "max_turns": 600},
    }
    if level < 1:
        level = 1
    if level > len(configs):
        level = len(configs)
    return configs[level]


def get_num_levels() -> int:
    return 4


# These are kept for backward compatibility but now delegate to generate_level
_level_cache: Dict[Tuple[int, int], Level] = {}

def get_level_grid(level: int, seed: int = 0) -> Grid:
    """Get Grid object for a given level and seed."""
    key = (level, seed)
    if key not in _level_cache:
        _level_cache[key] = generate_level(level, seed)
    return _level_cache[key].grid


def get_level_map(level: int) -> str:
    """Get ASCII map for a given level (legacy - uses seed 0)."""
    return get_level_grid(level, 0).to_ascii()


if __name__ == "__main__":
    # Quick test
    for level in range(1, 5):
        for seed in range(3):
            try:
                lvl = generate_level(level, seed)
                print(f"Level {level} seed {seed}: {lvl.archetype}, {lvl.grid.width}x{lvl.grid.height}, routes={len(lvl.routes)}, chokepoints={len(lvl.chokepoints)}")
            except Exception as e:
                print(f"Level {level} seed {seed} failed: {e}")