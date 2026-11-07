"""Tests for enemy AI: vision, chase, patrol, flank behavior."""
from __future__ import annotations
import pytest
from typing import List, Tuple

from grid import Grid, Position
from pathing import astar
from learning import Tracker, ZoneModel
from game.enemy import Patroller, Flanker, EnemyState


def make_open_room_grid() -> Grid:
    """Create a simple open room grid for testing."""
    ascii_map = """\
##########
#S.......#
#........#
#........#
#.......E#
##########"""
    return Grid.from_ascii(ascii_map)


def make_grid_with_wall() -> Grid:
    """Create a grid with a wall blocking line of sight."""
    ascii_map = """\
##########
#S.......#
#.######.#
#........#
#.......E#
##########"""
    return Grid.from_ascii(ascii_map)


def make_patrol_route(grid: Grid) -> List[Position]:
    """Create a simple patrol route - returns waypoints (corners) for Patroller to connect."""
    # Return corner waypoints that Patroller will connect with A*
    # For an open room, use the four corners
    walkable = []
    for y in range(1, grid.height - 1):
        for x in range(1, grid.width - 1):
            if grid.is_walkable((x, y)):
                walkable.append((x, y))

    if not walkable:
        return []

    xs = [p[0] for p in walkable]
    ys = [p[1] for p in walkable]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Return four corners as waypoints
    return [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]


class TestEnemyVision:
    """Tests for enemy line of sight."""

    def test_vision_in_range_open_room(self):
        """Test that enemy sees player within radius in open room."""
        grid = make_open_room_grid()
        patrol_route = make_patrol_route(grid)
        enemy = Patroller(patrol_route, grid)
        enemy.position = (2, 2)
        player_pos = (4, 2)  # Distance 2, no walls

        assert enemy.sees(grid, player_pos) == True

    def test_vision_out_of_range(self):
        """Test that enemy does not see player beyond sight radius."""
        grid = make_open_room_grid()
        patrol_route = make_patrol_route(grid)
        enemy = Patroller(patrol_route, grid)
        enemy.position = (1, 1)
        player_pos = (8, 4)  # Distance > 5

        assert enemy.sees(grid, player_pos) == False

    def test_vision_blocked_by_wall(self):
        """Test that wall blocks line of sight."""
        grid = make_grid_with_wall()
        patrol_route = make_patrol_route(grid)
        enemy = Patroller(patrol_route, grid)
        enemy.position = (2, 1)  # Above the wall
        player_pos = (2, 3)  # Below the wall, distance 2 but wall in between

        assert enemy.sees(grid, player_pos) == False

    def test_vision_not_blocked_by_wall_beside(self):
        """Test that wall to the side doesn't block if not on line."""
        grid = make_grid_with_wall()
        patrol_route = make_patrol_route(grid)
        enemy = Patroller(patrol_route, grid)
        enemy.position = (1, 2)  # Left of wall
        player_pos = (3, 2)  # Right of wall, same row - wall at (2,2) is between

        # This should be blocked since wall is directly between them
        assert enemy.sees(grid, player_pos) == False

    def test_vision_adjacent_no_wall(self):
        """Test vision works for adjacent tiles."""
        grid = make_open_room_grid()
        patrol_route = make_patrol_route(grid)
        enemy = Patroller(patrol_route, grid)
        enemy.position = (3, 2)
        player_pos = (4, 2)

        assert enemy.sees(grid, player_pos) == True


class TestEnemyPatrol:
    """Tests for patrol behavior."""

    def test_patrol_loops(self):
        """Test that patroller loops through patrol route."""
        grid = make_open_room_grid()
        patrol_route = make_patrol_route(grid)
        enemy = Patroller(patrol_route, grid)

        # Move through entire route
        positions = []
        full_route = enemy.patrol_route  # The connected route from waypoints
        for _ in range(len(full_route) + 2):
            enemy.update(grid, (100, 100), Tracker(), ZoneModel())  # Player far away
            positions.append(enemy.position)

        # Should have visited all route positions and looped back
        # full_route includes start position at both beginning and end, so loop completes in len(full_route) - 1 steps
        assert positions[0] == full_route[1]  # First step
        assert positions[len(full_route) - 2] == full_route[0]  # Completed loop (back at start)
        assert positions[len(full_route) - 1] == full_route[1]  # Looped again (second step of next loop)

    def test_patrol_stays_on_route(self):
        """Test that patrol only uses positions from the route."""
        grid = make_open_room_grid()
        patrol_route = make_patrol_route(grid)
        enemy = Patroller(patrol_route, grid)
        full_route = enemy.patrol_route  # The connected route from waypoints

        for _ in range(20):
            enemy.update(grid, (100, 100), Tracker(), ZoneModel())
            assert enemy.position in full_route


class TestEnemyChase:
    """Tests for chase behavior."""

    def test_chase_reaches_adjacent_player_open_room(self):
        """Test that chaser reaches adjacent to player in open room."""
        grid = make_open_room_grid()
        patrol_route = make_patrol_route(grid)
        enemy = Patroller(patrol_route, grid)
        enemy.position = (2, 2)

        # Create tracker with some data so cost map isn't empty
        tracker = Tracker()
        tracker.record((4, 2))
        tracker.record((4, 2))
        tracker.record((4, 2))

        zone_model = ZoneModel()
        player_pos = (4, 2)

        # Update enemy - should see player and chase
        for _ in range(10):
            enemy.update(grid, player_pos, tracker, zone_model)
            if enemy.position == player_pos:
                break

        # Enemy should reach player or be adjacent
        dist = abs(enemy.position[0] - player_pos[0]) + abs(enemy.position[1] - player_pos[1])
        assert dist <= 1, f"Enemy at {enemy.position}, player at {player_pos}, dist={dist}"

    def test_chase_switches_back_to_patrol_when_lost_sight(self):
        """Test that enemy returns to patrol when losing sight of player."""
        grid = make_grid_with_wall()
        patrol_route = make_patrol_route(grid)
        enemy = Patroller(patrol_route, grid)
        enemy.position = (2, 1)

        tracker = Tracker()
        zone_model = ZoneModel()

        # Player behind wall - can't see
        player_pos = (2, 3)
        enemy.update(grid, player_pos, tracker, zone_model)
        assert enemy.state == EnemyState.PATROL

        # Player moves to visible position
        player_pos = (4, 1)
        enemy.update(grid, player_pos, tracker, zone_model)
        assert enemy.state == EnemyState.CHASE

        # Player moves far away - out of sight radius
        player_pos = (100, 100)
        enemy.update(grid, player_pos, tracker, zone_model)
        assert enemy.state == EnemyState.PATROL


class TestEnemyFlanker:
    """Tests for flanker behavior."""

    def test_flanker_moves_to_flank_target(self):
        """Test that flanker moves toward flank target when player in hottest zone."""
        grid = make_open_room_grid()
        patrol_route = make_patrol_route(grid)
        flanker = Flanker(patrol_route[0], patrol_route, grid)
        flanker.position = (2, 2)

        tracker = Tracker()
        # Make east zone hottest (near exit)
        for _ in range(10):
            tracker.record((8, 3))
        zone_model = ZoneModel()

        player_pos = (8, 3)  # In east zone (hottest)

        # Update a few times - should move toward flank target
        initial_pos = flanker.position
        for _ in range(5):
            flanker.update(grid, player_pos, tracker, zone_model)

        # Should have moved from initial position
        assert flanker.position != initial_pos or flanker.position == flanker.flank_target

    def test_flanker_waits_at_flank_target(self):
        """Test that flanker waits at flank target when player in hottest zone."""
        grid = make_open_room_grid()
        patrol_route = make_patrol_route(grid)
        flanker = Flanker(patrol_route[0], patrol_route, grid)

        tracker = Tracker()
        for _ in range(10):
            tracker.record((8, 3))
        zone_model = ZoneModel()

        player_pos = (8, 3)  # In east zone

        # Move to flank target
        for _ in range(20):
            flanker.update(grid, player_pos, tracker, zone_model)
            if flanker.position == flanker.flank_target:
                break

        # Once at flank target, should stay there (wait in ambush)
        pos_at_target = flanker.position
        for _ in range(5):
            flanker.update(grid, player_pos, tracker, zone_model)
            assert flanker.position == pos_at_target, "Flanker should wait at flank target"

    def test_flanker_chases_when_sees_player(self):
        """Test that flanker switches to chase when seeing player."""
        grid = make_open_room_grid()
        patrol_route = make_patrol_route(grid)
        flanker = Flanker(patrol_route[0], patrol_route, grid)
        flanker.position = (4, 2)

        tracker = Tracker()
        for _ in range(10):
            tracker.record((8, 3))
        zone_model = ZoneModel()

        # Player in hottest zone and visible
        player_pos = (6, 2)
        flanker.update(grid, player_pos, tracker, zone_model)

        assert flanker.state == EnemyState.CHASE


class TestEnemyStateMachine:
    """Tests for state transitions."""

    def test_patrol_to_chase_transition(self):
        """Test patrol -> chase transition on sight."""
        grid = make_open_room_grid()
        patrol_route = make_patrol_route(grid)
        enemy = Patroller(patrol_route, grid)
        enemy.position = (3, 2)

        tracker = Tracker()
        zone_model = ZoneModel()

        assert enemy.state == EnemyState.PATROL

        # Player comes into view
        player_pos = (5, 2)
        enemy.update(grid, player_pos, tracker, zone_model)

        assert enemy.state == EnemyState.CHASE

    def test_chase_to_patrol_transition(self):
        """Test chase -> patrol transition when losing sight."""
        grid = make_open_room_grid()
        patrol_route = make_patrol_route(grid)
        enemy = Patroller(patrol_route, grid)
        enemy.position = (3, 2)

        tracker = Tracker()
        zone_model = ZoneModel()

        # Chase player
        player_pos = (5, 2)
        enemy.update(grid, player_pos, tracker, zone_model)
        assert enemy.state == EnemyState.CHASE

        # Player moves far away
        player_pos = (100, 100)
        enemy.update(grid, player_pos, tracker, zone_model)
        assert enemy.state == EnemyState.PATROL

    def test_replan_every_3_turns(self):
        """Test that A* is re-planned every 3 turns during chase."""
        grid = make_open_room_grid()
        patrol_route = make_patrol_route(grid)
        enemy = Patroller(patrol_route, grid)
        enemy.position = (4, 2)

        tracker = Tracker()
        tracker.record((6, 2))
        zone_model = ZoneModel()

        player_pos = (6, 2)  # Distance 2, within sight radius
        # Force chase state with a pre-existing path (so first update doesn't trigger immediate replan)
        enemy.state = EnemyState.CHASE
        enemy.target = player_pos
        enemy.path = [(5, 2), (6, 2)]  # Pre-populate path
        enemy.replan_counter = 0

        # First update - counter becomes 1
        enemy.update(grid, player_pos, tracker, zone_model)
        assert enemy.replan_counter == 1

        # Second update - counter becomes 2
        enemy.update(grid, player_pos, tracker, zone_model)
        assert enemy.replan_counter == 2

        # Third update - counter becomes 3, triggers replan, resets to 0
        enemy.update(grid, player_pos, tracker, zone_model)
        assert enemy.replan_counter == 0

    def test_replan_on_state_change(self):
        """Test that A* is re-planned immediately on state change."""
        grid = make_open_room_grid()
        patrol_route = make_patrol_route(grid)
        enemy = Patroller(patrol_route, grid)
        enemy.position = (3, 2)

        tracker = Tracker()
        zone_model = ZoneModel()

        # Start in patrol
        player_pos = (100, 100)
        enemy.update(grid, player_pos, tracker, zone_model)
        assert enemy.state == EnemyState.PATROL

        # Player appears - state changes to chase, should replan immediately
        player_pos = (5, 2)
        enemy.update(grid, player_pos, tracker, zone_model)
        assert enemy.state == EnemyState.CHASE
        # replan_counter should be set to replan_interval to force replan
        assert enemy.replan_counter == enemy.replan_interval


class TestCostMapIntegration:
    """Tests for cost map integration with enemy pathfinding."""

    def test_build_cost_map_clamps_min_cost(self):
        """Test that build_cost_map clamps minimum step cost."""
        from ai.costs import build_cost_map

        tracker = Tracker()
        tracker.record((5, 5))
        tracker.record((5, 5))
        tracker.record((5, 5))
        tracker.record((6, 6))

        cost_map = build_cost_map(tracker, alpha=1.0, min_step_cost=0.1)

        # All costs should be >= -0.9 (so 1.0 + cost >= 0.1)
        for cost in cost_map.values():
            assert cost >= -0.9, f"Cost {cost} below minimum"

    def test_build_flank_targets_returns_valid_position(self):
        """Test that build_flank_targets returns a walkable position."""
        from ai.costs import build_flank_targets

        grid = make_open_room_grid()
        tracker = Tracker()
        for _ in range(10):
            tracker.record((8, 3))  # East zone

        prob_map = tracker.probability_map()
        target = build_flank_targets(grid, prob_map)

        assert target is not None
        assert grid.is_walkable(target)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])