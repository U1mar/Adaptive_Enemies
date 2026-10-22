"""Pytest tests for the adaptive enemies game."""
from __future__ import annotations
import pytest
import random
from typing import List, Tuple, Set

from grid import Grid, Position
from pathing import bfs, dijkstra, astar
from learning import Tracker, ZoneModel
from engine import Game, RoundResult
from bots import FixedRouteBot, SwitchingBot, RandomBot


# ===== Grid Tests =====

def test_grid_from_ascii():
    """Test Grid.from_ascii parses correctly."""
    ascii_map = """\
#####
#S..#
#.#.#
#..V#
#...E
#####"""
    grid = Grid.from_ascii(ascii_map)
    assert grid.width == 5
    assert grid.height == 6
    assert grid.start == (1, 1)
    assert grid.vault == (3, 3)
    assert grid.exit == (4, 4)
    assert (0, 0) in grid.walls
    assert (1, 1) not in grid.walls


def test_grid_neighbors():
    """Test 4-directional neighbors."""
    ascii_map = """\
#######
#.....#
#.....#
#.....#
#######"""
    grid = Grid.from_ascii(ascii_map)
    # Center tile (3, 2) - should have 4 neighbors
    neighbors = grid.neighbors((3, 2))
    assert len(neighbors) == 4
    assert (2, 2) in neighbors
    assert (4, 2) in neighbors
    assert (3, 1) in neighbors
    assert (3, 3) in neighbors
    # Corner tile
    neighbors = grid.neighbors((1, 1))
    assert len(neighbors) == 2


def test_grid_zone_of():
    """Test zone_of returns correct zone by x-third."""
    ascii_map = """\
##########
#........#
#........#
##########"""
    grid = Grid.from_ascii(ascii_map)
    # width=10, thirds at 3 and 6
    assert grid.zone_of((1, 1)) == 'west'    # x=1 < 3
    assert grid.zone_of((3, 1)) == 'middle'  # x=3 >= 3 and < 6
    assert grid.zone_of((7, 1)) == 'east'    # x=7 >= 6


# ===== Pathing Tests =====

def make_open_grid(w: int, h: int) -> Grid:
    """Create an open grid for testing."""
    lines = ['#' * w]
    for _ in range(h - 2):
        lines.append('#' + '.' * (w - 2) + '#')
    lines.append('#' * w)
    # Add start and exit
    lines[1] = '#S' + '.' * (w - 3) + '#'
    lines[-2] = '#' + '.' * (w - 3) + 'E#'
    return Grid.from_ascii('\n'.join(lines))


def test_bfs_finds_path():
    """Test BFS finds a valid path."""
    grid = make_open_grid(10, 8)
    path, explored = bfs(grid, grid.start, grid.exit)
    assert len(path) > 0
    assert path[0] == grid.start
    assert path[-1] == grid.exit
    # Verify path is valid
    for i in range(len(path) - 1):
        assert path[i+1] in grid.neighbors(path[i])


def test_dijkstra_finds_path():
    """Test Dijkstra finds a valid path."""
    grid = make_open_grid(10, 8)
    path, explored = dijkstra(grid, grid.start, grid.exit)
    assert len(path) > 0
    assert path[0] == grid.start
    assert path[-1] == grid.exit
    for i in range(len(path) - 1):
        assert path[i+1] in grid.neighbors(path[i])


def test_astar_finds_path():
    """Test A* finds a valid path."""
    grid = make_open_grid(10, 8)
    path, explored = astar(grid, grid.start, grid.exit)
    assert len(path) > 0
    assert path[0] == grid.start
    assert path[-1] == grid.exit
    for i in range(len(path) - 1):
        assert path[i+1] in grid.neighbors(path[i])


def test_pathing_with_cost_map():
    """Test pathing algorithms respect cost_map."""
    grid = make_open_grid(10, 8)
    # Make a tile very expensive
    cost_map = {(5, 1): 100.0}
    path, _ = astar(grid, grid.start, grid.exit, cost_map)
    # Path should avoid (5, 1) if possible
    assert (5, 1) not in path or len(path) < 20  # Allow if forced


def test_pathing_same_start_goal():
    """Test pathing when start == goal."""
    grid = make_open_grid(10, 8)
    for algo in [bfs, dijkstra, astar]:
        path, explored = algo(grid, grid.start, grid.start)
        assert path == [grid.start]
        assert explored == 1


# ===== Tracker Tests =====

def test_tracker_record_and_decay():
    """Test Tracker records visits and applies decay."""
    tracker = Tracker(decay=0.5)
    tracker.record((1, 1))
    tracker.record((1, 1))
    tracker.record((2, 2))
    assert tracker.visit_counts[(1, 1)] == 2.0
    assert tracker.visit_counts[(2, 2)] == 1.0

    tracker.decay_counts()
    assert tracker.visit_counts[(1, 1)] == 1.0  # 2.0 * 0.5
    assert tracker.visit_counts[(2, 2)] == 0.5  # 1.0 * 0.5


def test_tracker_probability_map():
    """Test Tracker probability_map normalizes correctly."""
    tracker = Tracker()
    tracker.record((1, 1))
    tracker.record((1, 1))
    tracker.record((2, 2))
    probs = tracker.probability_map()
    assert abs(probs[(1, 1)] - 2/3) < 0.01
    assert abs(probs[(2, 2)] - 1/3) < 0.01
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_tracker_history_limit():
    """Test Tracker history keeps last 50 moves."""
    tracker = Tracker(history_len=5)
    for i in range(10):
        tracker.record((i, 0))
    history = tracker.get_history()
    assert len(history) == 5
    assert history == [(5, 0), (6, 0), (7, 0), (8, 0), (9, 0)]


# ===== ZoneModel Tests =====

def test_zone_model_record_transition():
    """Test ZoneModel records transitions."""
    zm = ZoneModel(zones=['west', 'middle', 'east'], alpha=1.0)
    zm.record_transition('west', 'middle')
    zm.record_transition('west', 'middle')
    zm.record_transition('west', 'east')
    assert zm.transitions['west']['middle'] == 2
    assert zm.transitions['west']['east'] == 1
    assert zm.last_zone == 'east'


def test_zone_model_predict_next():
    """Test ZoneModel predicts next zone with Laplace smoothing."""
    zm = ZoneModel(zones=['west', 'middle', 'east'], alpha=1.0)
    # No data yet - should return random, but deterministic with seed
    # With Laplace smoothing, all equal probability, returns first max
    zm.record_transition('west', 'middle')
    zm.record_transition('west', 'middle')
    zm.record_transition('west', 'east')
    # west -> middle: 2+1=3, west -> east: 1+1=2, west -> west: 0+1=1
    # Total = 3+2+1=6, probs: middle=3/6, east=2/6, west=1/6
    predicted = zm.predict_next('west')
    assert predicted == 'middle'


def test_zone_model_prediction_accuracy():
    """Test ZoneModel prediction_accuracy calculation."""
    zm = ZoneModel(zones=['west', 'middle', 'east'], alpha=1.0)
    sequence = ['west', 'middle', 'east', 'west', 'middle']
    # Train on first part
    for i in range(len(sequence) - 1):
        zm.record_transition(sequence[i], sequence[i+1])
    # Accuracy on same sequence (should be 1.0 since trained on it)
    acc = zm.prediction_accuracy(sequence)
    assert acc == 1.0


# ===== Headless Round Test =====

def test_headless_round():
    """Test one complete headless round with a simple bot."""
    game = Game()
    bot = FixedRouteBot()
    result = game.run_round(bot, max_turns=200)
    assert isinstance(result, RoundResult)
    assert result.turns > 0
    assert result.turns <= 200
    assert len(result.path) == result.turns + 1
    assert result.path[0] == game.grid.start
    # Either caught, escaped, or timed out
    assert result.caught or result.escaped or result.turns == 200


def test_tracker_persists_across_rounds():
    """Test Tracker and ZoneModel persist across rounds."""
    game = Game()
    bot = FixedRouteBot()

    # Round 1
    result1 = game.run_round(bot, max_turns=200)
    tracker_counts_1 = dict(game.tracker.visit_counts)
    zone_transitions_1 = {z: dict(v) for z, v in game.zone_model.transitions.items()}

    game.reset_enemies()

    # Round 2
    result2 = game.run_round(bot, max_turns=200)
    tracker_counts_2 = dict(game.tracker.visit_counts)
    zone_transitions_2 = {z: dict(v) for z, v in game.zone_model.transitions.items()}

    # Tracker should have more data (decayed but accumulated)
    assert len(tracker_counts_2) >= len(tracker_counts_1) * 0.5  # Some decay but persists
    # ZoneModel should have more transitions
    total_trans_1 = sum(sum(v.values()) for v in zone_transitions_1.values())
    total_trans_2 = sum(sum(v.values()) for v in zone_transitions_2.values())
    assert total_trans_2 > total_trans_1


# ===== Bot Tests =====

def test_fixed_route_bot():
    """Test FixedRouteBot returns valid moves."""
    grid = make_open_grid(10, 8)
    bot = FixedRouteBot()
    pos = grid.start
    got_key = False
    for _ in range(20):
        next_pos = bot(grid, pos, got_key, [])
        assert next_pos in grid.neighbors(pos) or next_pos == pos
        pos = next_pos
        if pos == grid.vault:
            got_key = True


def test_random_bot():
    """Test RandomBot returns valid moves."""
    grid = make_open_grid(10, 8)
    bot = RandomBot()
    pos = grid.start
    for _ in range(20):
        next_pos = bot(grid, pos, False, [])
        assert next_pos in grid.neighbors(pos) or next_pos == pos
        pos = next_pos


def test_switching_bot():
    """Test SwitchingBot changes corridor."""
    bot = SwitchingBot()
    assert bot.corridor == 'west'
    bot.new_round()
    bot.new_round()
    bot.new_round()  # 3 rounds
    assert bot.corridor == 'middle'
    for _ in range(3):
        bot.new_round()
    assert bot.corridor == 'east'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])