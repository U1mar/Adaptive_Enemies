"""Tests for game engine."""
from __future__ import annotations
import pytest
from typing import List, Tuple

from grid import Grid, Position
from game.engine import Game, RoundResult, Policy
from learning import Tracker, ZoneModel
from game.enemy import Patroller, Flanker


def make_trivial_grid() -> Grid:
    """Create a trivial open grid with no walls between start, vault, and exit."""
    ascii_map = """\
##########
#S...V..E#
##########"""
    return Grid.from_ascii(ascii_map)


def test_player_standing_still_gets_caught():
    """A player who doesn't move should be caught by patrolling enemies."""
    grid = Grid.from_ascii("""\
########
#S.....#
#.####.#
#...V..#
#.####.#
#.....E#
########""")

    game = Game(grid=grid)

    # Policy that makes player stand still
    def stand_still_policy(grid: Grid, pos: Position, got_key: bool, enemies: List[Position]) -> Position:
        return pos

    result = game.play_round(stand_still_policy, max_turns=100)

    assert result.caught == True
    assert result.escaped == False
    assert result.turns > 0


def test_player_on_trivial_map_with_no_enemies_escapes():
    """On a simple map with no enemies, player should reach exit with key."""
    grid = make_trivial_grid()

    game = Game(grid=grid)
    # Remove all enemies
    game.enemies.clear()

    # Simple policy: move right towards vault then exit
    def simple_policy(grid: Grid, pos: Position, got_key: bool, enemies: List[Position]) -> Position:
        x, y = pos
        if not got_key:
            # Move right towards vault (which is at x=5)
            if x < 5:
                return (x + 1, y)
            else:
                return pos
        else:
            # Move right towards exit (which is at x=8)
            if x < 8:
                return (x + 1, y)
            else:
                return pos

    result = game.play_round(simple_policy, max_turns=50)

    assert result.caught == False
    assert result.got_key == True
    assert result.escaped == True
    assert result.turns > 0
    # Path should include start, vault, and exit
    assert result.path[0] == grid.start
    assert grid.vault in result.path
    assert result.path[-1] == grid.exit


def test_tracker_holds_data_after_round():
    """Tracker should retain visit counts after a round completes."""
    grid = make_trivial_grid()
    game = Game(grid=grid)
    game.enemies.clear()  # No enemies to simplify

    def simple_policy(grid: Grid, pos: Position, got_key: bool, enemies: List[Position]) -> Position:
        x, y = pos
        if not got_key and x < 5:
            return (x + 1, y)
        elif got_key and x < 8:
            return (x + 1, y)
        return pos

    # Run first round
    result1 = game.play_round(simple_policy, max_turns=50)

    # Check tracker has data
    visit_counts_after_round1 = dict(game.tracker.visit_counts)
    assert len(visit_counts_after_round1) > 0
    assert sum(visit_counts_after_round1.values()) > 0

    # Reset enemies for next round (tracker and zone_model persist)
    game.reset_enemies()

    # Run second round
    result2 = game.play_round(simple_policy, max_turns=50)

    # Tracker should still have data (possibly decayed but accumulated)
    visit_counts_after_round2 = dict(game.tracker.visit_counts)
    assert len(visit_counts_after_round2) > 0

    # Total visits should be at least as many as round 1 (some decay but new visits added)
    total_visits_1 = sum(visit_counts_after_round1.values())
    total_visits_2 = sum(visit_counts_after_round2.values())
    assert total_visits_2 >= total_visits_1 * 0.5  # Allow for decay


def test_zone_model_persists_across_rounds():
    """ZoneModel should retain transitions across rounds."""
    grid = make_trivial_grid()
    game = Game(grid=grid)
    game.enemies.clear()

    def simple_policy(grid: Grid, pos: Position, got_key: bool, enemies: List[Position]) -> Position:
        x, y = pos
        if not got_key and x < 5:
            return (x + 1, y)
        elif got_key and x < 8:
            return (x + 1, y)
        return pos

    # Run first round
    result1 = game.play_round(simple_policy, max_turns=50)

    # Check zone_model has transitions
    transitions_1 = {z: dict(v) for z, v in game.zone_model.transitions.items()}
    total_trans_1 = sum(sum(v.values()) for v in transitions_1.values())
    assert total_trans_1 > 0

    # Reset enemies for next round (this re-adds default enemies, so clear again)
    game.reset_enemies()
    game.enemies.clear()

    # Run second round
    result2 = game.play_round(simple_policy, max_turns=50)

    # ZoneModel should have more transitions (accumulated from both rounds)
    transitions_2 = {z: dict(v) for z, v in game.zone_model.transitions.items()}
    total_trans_2 = sum(sum(v.values()) for v in transitions_2.values())
    assert total_trans_2 > total_trans_1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])