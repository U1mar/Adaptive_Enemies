"""Tests for pathing algorithms."""
from __future__ import annotations
import random
import networkx as nx
import pytest
from adaptive_enemies.game.grid import Grid, Pos
from adaptive_enemies.pathing import bfs, dijkstra, astar, PathResult


def make_random_grid(width: int, height: int, wall_prob: float = 0.2, seed: int = None) -> Grid:
    """Generate a random grid with walls."""
    if seed is not None:
        random.seed(seed)

    walls: set[Pos] = set()
    for y in range(height):
        for x in range(width):
            if random.random() < wall_prob:
                walls.add((x, y))

    # Ensure start and goal are passable
    start = (0, 0)
    goal = (width - 1, height - 1)
    walls.discard(start)
    walls.discard(goal)

    # Use vault as goal, exit as same for simplicity
    return Grid(width, height, walls, start, goal, goal)


def grid_to_networkx(grid: Grid) -> nx.Graph:
    """Convert grid to networkx graph for comparison."""
    G = nx.Graph()
    for y in range(grid.height):
        for x in range(grid.width):
            pos = (x, y)
            if grid.passable(pos):
                G.add_node(pos)
                for neighbor in grid.neighbors(pos):
                    G.add_edge(pos, neighbor, weight=1)
    return G


def test_path_lengths_match_networkx():
    """BFS, Dijkstra, A* should match networkx shortest path length on unweighted grids."""
    random.seed(42)

    for i in range(200):
        width = random.randint(10, 30)
        height = random.randint(10, 30)
        grid = make_random_grid(width, height, wall_prob=0.25, seed=i)

        G = grid_to_networkx(grid)

        try:
            nx_length = nx.shortest_path_length(G, grid.start, grid.vault)
        except nx.NetworkXNoPath:
            # Unreachable - all should return None
            bfs_result = bfs(grid, grid.start, grid.vault)
            dijkstra_result = dijkstra(grid, grid.start, grid.vault)
            astar_result = astar(grid, grid.start, grid.vault)
            assert bfs_result.path is None
            assert dijkstra_result.path is None
            assert astar_result.path is None
            continue

        # All three should find a path of the same length
        bfs_result = bfs(grid, grid.start, grid.vault)
        dijkstra_result = dijkstra(grid, grid.start, grid.vault)
        astar_result = astar(grid, grid.start, grid.vault)

        assert bfs_result.path is not None, f"BFS failed on grid {i}"
        assert dijkstra_result.path is not None, f"Dijkstra failed on grid {i}"
        assert astar_result.path is not None, f"A* failed on grid {i}"

        assert len(bfs_result.path) - 1 == nx_length, f"BFS length mismatch on grid {i}"
        assert len(dijkstra_result.path) - 1 == nx_length, f"Dijkstra length mismatch on grid {i}"
        assert len(astar_result.path) - 1 == nx_length, f"A* length mismatch on grid {i}"


def test_cost_map_avoidance():
    """Cost map should make Dijkstra and A* avoid costly tile when equal-length alternative exists."""
    # 6x3 grid with two equal-length paths (length 7 each)
    # S=(0,1), G=(5,1)
    # Path 1 (top): goes via row 0
    # Path 2 (bottom): goes via row 2
    # Middle row (y=1) blocked except start/goal
    walls = {(1, 1), (2, 1), (3, 1), (4, 1)}
    grid = Grid(6, 3, walls, start=(0, 1), vault=(5, 1), exit=(5, 1))

    # Without cost map - both paths equal length 7
    d1 = dijkstra(grid, grid.start, grid.vault)
    a1 = astar(grid, grid.start, grid.vault)
    assert d1.path is not None
    assert len(d1.path) - 1 == 7

    # Add cost to a tile unique to path 1 (top path), e.g., (1, 0)
    cost_map = {(1, 0): 10.0}
    d2 = dijkstra(grid, grid.start, grid.vault, cost_map)
    a2 = astar(grid, grid.start, grid.vault, cost_map)
    assert d2.path is not None
    assert a2.path is not None
    # Should take the bottom path (avoiding (1, 0))
    assert (1, 0) not in d2.path, "Dijkstra should avoid costly tile"
    assert (1, 0) not in a2.path, "A* should avoid costly tile"
    assert len(d2.path) - 1 == 7  # Still optimal length


def test_unreachable_returns_none():
    """Unreachable goals should return path=None."""
    # Horizontal wall completely blocking
    walls = {(x, 5) for x in range(10)}  # Wall at y=5
    grid = Grid(10, 10, walls, start=(0, 0), vault=(9, 9), exit=(9, 9))

    bfs_result = bfs(grid, grid.start, grid.vault)
    dijkstra_result = dijkstra(grid, grid.start, grid.vault)
    astar_result = astar(grid, grid.start, grid.vault)

    assert bfs_result.path is None
    assert dijkstra_result.path is None
    assert astar_result.path is None
    # Explored should be > 0 (they searched)
    assert bfs_result.explored > 0
    assert dijkstra_result.explored > 0
    assert astar_result.explored > 0


def test_start_equals_goal():
    """Start equals goal should return path of length 1."""
    grid = Grid(5, 5, set(), start=(2, 2), vault=(2, 2), exit=(2, 2))

    bfs_result = bfs(grid, grid.start, grid.vault)
    dijkstra_result = dijkstra(grid, grid.start, grid.vault)
    astar_result = astar(grid, grid.start, grid.vault)

    assert bfs_result.path == [(2, 2)]
    assert dijkstra_result.path == [(2, 2)]
    assert astar_result.path == [(2, 2)]
    assert bfs_result.cost == 0
    assert dijkstra_result.cost == 0
    assert astar_result.cost == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])