"""Tests for game/grid.py"""
from __future__ import annotations
from collections import deque
import pytest
from adaptive_enemies.game.grid import Grid, Pos
from adaptive_enemies.game.levels import DEFAULT_LEVEL


def test_bounds():
    """Test in_bounds method."""
    grid = Grid.from_ascii(DEFAULT_LEVEL)
    assert grid.in_bounds((0, 0)) is True
    assert grid.in_bounds((grid.width - 1, grid.height - 1)) is True
    assert grid.in_bounds((-1, 0)) is False
    assert grid.in_bounds((grid.width, 0)) is False
    assert grid.in_bounds((0, -1)) is False
    assert grid.in_bounds((0, grid.height)) is False


def test_walls():
    """Test passable method with walls."""
    grid = Grid.from_ascii(DEFAULT_LEVEL)
    # Corners are walls
    assert grid.passable((0, 0)) is False
    assert grid.passable((grid.width - 1, 0)) is False
    # Start, vault, exit should be passable (not walls)
    assert grid.passable(grid.start) is True
    assert grid.passable(grid.vault) is True
    assert grid.passable(grid.exit) is True


def test_neighbors():
    """Test neighbors returns 4-directional passable neighbors."""
    grid = Grid.from_ascii(DEFAULT_LEVEL)
    # Start position should have neighbors
    neighbors = grid.neighbors(grid.start)
    assert len(neighbors) > 0
    # All neighbors should be passable
    for n in neighbors:
        assert grid.passable(n) is True
    # Neighbors should be 4-directional (no diagonals)
    for n in neighbors:
        dx = abs(n[0] - grid.start[0])
        dy = abs(n[1] - grid.start[1])
        assert (dx == 1 and dy == 0) or (dx == 0 and dy == 1)


def test_from_ascii_parsing():
    """Test from_ascii correctly parses the map."""
    grid = Grid.from_ascii(DEFAULT_LEVEL)
    assert grid.width == 40
    assert grid.height == 20
    # Check that walls are correctly identified
    assert len(grid.walls) > 0
    # Check S, V, E positions are found
    assert grid.start is not None
    assert grid.vault is not None
    assert grid.exit is not None
    # Verify they are different positions
    assert grid.start != grid.vault
    assert grid.vault != grid.exit
    assert grid.start != grid.exit


def test_zone_of():
    """Test zone_of returns correct zone for all three zones."""
    grid = Grid.from_ascii(DEFAULT_LEVEL)
    third = grid.width / 3

    # West zone (x < width/3)
    west_pos = (int(third / 2), grid.height // 2)
    assert grid.zone_of(west_pos) == "west"

    # Middle zone (width/3 <= x < 2*width/3)
    middle_pos = (int(third + third / 2), grid.height // 2)
    assert grid.zone_of(middle_pos) == "middle"

    # East zone (x >= 2*width/3)
    east_pos = (int(2 * third + third / 2), grid.height // 2)
    assert grid.zone_of(east_pos) == "east"


def bfs_path(grid: Grid, start: Pos, goal: Pos) -> list[Pos] | None:
    """Simple BFS to find path from start to goal."""
    if not grid.passable(start) or not grid.passable(goal):
        return None
    queue = deque([start])
    visited = {start}
    parent = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return list(reversed(path))
        for neighbor in grid.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    return None


def test_default_level_reachability():
    """Test that S, V, E are reachable from each other in the default level."""
    grid = Grid.from_ascii(DEFAULT_LEVEL)

    # S -> V
    path_sv = bfs_path(grid, grid.start, grid.vault)
    assert path_sv is not None, "Start should be able to reach Vault"

    # V -> E
    path_ve = bfs_path(grid, grid.vault, grid.exit)
    assert path_ve is not None, "Vault should be able to reach Exit"

    # S -> E (direct)
    path_se = bfs_path(grid, grid.start, grid.exit)
    assert path_se is not None, "Start should be able to reach Exit"

    # E -> S (reverse)
    path_es = bfs_path(grid, grid.exit, grid.start)
    assert path_es is not None, "Exit should be able to reach Start"

    # V -> S (reverse)
    path_vs = bfs_path(grid, grid.vault, grid.start)
    assert path_vs is not None, "Vault should be able to reach Start"

    # E -> V (reverse)
    path_ev = bfs_path(grid, grid.exit, grid.vault)
    assert path_ev is not None, "Exit should be able to reach Vault"