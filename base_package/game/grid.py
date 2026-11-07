"""Grid representation for adaptive_enemies."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Set, List

Pos = tuple[int, int]


@dataclass
class Grid:
    width: int
    height: int
    walls: Set[Pos]
    start: Pos
    vault: Pos
    exit: Pos

    def in_bounds(self, p: Pos) -> bool:
        x, y = p
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, p: Pos) -> bool:
        return self.in_bounds(p) and p not in self.walls

    def neighbors(self, p: Pos) -> List[Pos]:
        x, y = p
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in candidates if self.passable(c)]

    @classmethod
    def from_ascii(cls, text: str) -> Grid:
        lines = text.strip('\n').split('\n')
        height = len(lines)
        width = len(lines[0]) if height > 0 else 0
        walls: Set[Pos] = set()
        start = vault = exit = (0, 0)

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
                    exit = pos

        return cls(width, height, walls, start, vault, exit)

    def zone_of(self, p: Pos) -> str:
        x, _ = p
        third = self.width / 3
        if x < third:
            return "west"
        elif x < 2 * third:
            return "middle"
        else:
            return "east"