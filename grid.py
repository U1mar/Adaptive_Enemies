"""Grid representation with walls, neighbors, and zone detection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
from collections import deque

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

    def zone_of(self, p: Position) -> str:
        """Return 'west', 'middle', or 'east' based on x-coordinate third."""
        x, _ = p
        third = self.width // 3
        if x < third:
            return 'west'
        elif x < 2 * third:
            return 'middle'
        return 'east'

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
            for x, ch in enumerate(line):
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
                    if val > 0.7:
                        row.append('█')
                    elif val > 0.4:
                        row.append('▓')
                    elif val > 0.1:
                        row.append('░')
                    else:
                        row.append('·')
                else:
                    row.append('.')
            lines.append(''.join(row))
        return '\n'.join(lines)


DEFAULT_MAP = """\
##############################
#S..............#............#
#.#############.#.###########.#
#.#..........V#.#.#.........#.#
#.#.#########.#.#.#########.#.#
#.#.#.......#.#.#.#.......#.#.#
#.#.#.#####.#.#.#.#####.#.#.#.#
#.#.#.#...#.#.#.#.#...#.#.#.#.#
#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#
#.#.#.#...#...#...#...#.#.#.#.#
#.#.#####.#.#######.#.#####.#.#
#.#.......#.......#.......#.#.#
#.#############.###########.#.#
#...........................E.#
##############################\
"""