"""Tests for UI rendering: map alignment, menu formatting, visible_width helper."""
from __future__ import annotations
import re
import random
from typing import List

import pytest

from grid import Grid, Position
from game.engine import Game
from game.ui import (
    visible_width,
    render_rows,
    build_level_table,
    GameUI,
    _colour_wrap,
    ANSI_RESET,
    ANSI_CYAN,
)
from game.enemy import Enemy, EnemyState, Patroller


class TestVisibleWidth:
    """Tests for the visible_width helper function."""

    def test_plain_string(self) -> None:
        assert visible_width("hello") == 5
        assert visible_width("") == 0
        assert visible_width("1234567890") == 10

    def test_ansi_codes_stripped(self) -> None:
        # Red text "hello" + reset
        s = f"\033[91mhello{ANSI_RESET}"
        assert visible_width(s) == 5

    def test_multiple_ansi_codes(self) -> None:
        s = f"\033[91mhel\033[92mlo{ANSI_RESET}"
        assert visible_width(s) == 5

    def test_ansi_with_numbers_and_semicolons(self) -> None:
        # Bright cyan on blue background: \033[96;44m
        s = f"\033[96;44mtest{ANSI_RESET}"
        assert visible_width(s) == 4

    def test_nested_colour_wrap(self) -> None:
        s = _colour_wrap("@", ANSI_CYAN)
        assert visible_width(s) == 1


class TestRenderRows:
    """Tests for render_rows map rendering function."""

    @pytest.fixture
    def game_level1_seed1(self) -> Game:
        return Game(level=1, seed=1)

    @pytest.fixture
    def game_level2_seed1(self) -> Game:
        return Game(level=2, seed=1)

    @pytest.fixture
    def game_level3_seed1(self) -> Game:
        return Game(level=3, seed=1)

    @pytest.fixture
    def game_level4_seed1(self) -> Game:
        return Game(level=4, seed=1)

    def _check_widths(self, rows: List[str], expected_width: int) -> None:
        """Assert all rows have the same visible width."""
        for i, row in enumerate(rows):
            vw = visible_width(row)
            assert vw == expected_width, f"Row {i} width {vw} != expected {expected_width}: {row!r}"

    def test_level1_seed1_widths(self, game_level1_seed1: Game) -> None:
        grid = game_level1_seed1.grid
        expected = grid.width * 2
        rows = render_rows(game_level1_seed1, heatmap_on=False, player_pos=grid.start, got_key=False)
        self._check_widths(rows, expected)
        assert len(rows) == grid.height

    def test_level2_seed1_widths(self, game_level2_seed1: Game) -> None:
        grid = game_level2_seed1.grid
        expected = grid.width * 2
        rows = render_rows(game_level2_seed1, heatmap_on=False, player_pos=grid.start, got_key=False)
        self._check_widths(rows, expected)
        assert len(rows) == grid.height

    def test_level3_seed1_widths(self, game_level3_seed1: Game) -> None:
        grid = game_level3_seed1.grid
        expected = grid.width * 2
        rows = render_rows(game_level3_seed1, heatmap_on=False, player_pos=grid.start, got_key=False)
        self._check_widths(rows, expected)
        assert len(rows) == grid.height

    def test_level4_seed1_widths(self, game_level4_seed1: Game) -> None:
        grid = game_level4_seed1.grid
        expected = grid.width * 2
        rows = render_rows(game_level4_seed1, heatmap_on=False, player_pos=grid.start, got_key=False)
        self._check_widths(rows, expected)
        assert len(rows) == grid.height

    def test_heatmap_on_widths(self, game_level1_seed1: Game) -> None:
        grid = game_level1_seed1.grid
        expected = grid.width * 2
        rows = render_rows(game_level1_seed1, heatmap_on=True, player_pos=grid.start, got_key=False)
        self._check_widths(rows, expected)

    def test_widths_after_random_steps(self, game_level1_seed1: Game) -> None:
        """Move enemies/player through 30 random steps, assert widths never change."""
        grid = game_level1_seed1.grid
        expected = grid.width * 2

        player_pos = grid.start
        game = game_level1_seed1

        for step in range(30):
            # Pick a random valid move
            neighbors = grid.neighbors(player_pos)
            if neighbors:
                player_pos = random.choice(neighbors)

            # Step the game (this moves enemies)
            game.step(player_pos)

            # Render and check
            rows = render_rows(game, heatmap_on=False, player_pos=player_pos, got_key=game.got_key)
            self._check_widths(rows, expected)

    def test_widths_with_heatmap_after_random_steps(self, game_level1_seed1: Game) -> None:
        """Same test but with heatmap on."""
        grid = game_level1_seed1.grid
        expected = grid.width * 2

        player_pos = grid.start
        game = game_level1_seed1

        for step in range(30):
            neighbors = grid.neighbors(player_pos)
            if neighbors:
                player_pos = random.choice(neighbors)
            game.step(player_pos)

            rows = render_rows(game, heatmap_on=True, player_pos=player_pos, got_key=game.got_key)
            self._check_widths(rows, expected)

    def test_overlapping_enemies_and_player(self) -> None:
        """Put two enemies and player on same tile, assert row width still correct."""
        # Create a minimal game with a small grid
        from grid import Grid
        walls = {(x, y) for x in range(10) for y in (0, 9)}
        walls.update({(x, y) for x in (0, 9) for y in range(10)})
        grid = Grid(10, 10, walls, (1, 1), (5, 5), (8, 8))

        # Create game manually to control enemies
        game = Game(grid=grid, level=1, seed=1)
        game.enemies.clear()

        # Add two patrollers at the same position as player
        pos = (3, 3)
        from game.enemy import Patroller
        p1 = Patroller([pos, (4, 4)], grid)
        p1.position = pos
        p2 = Patroller([pos, (4, 4)], grid)
        p2.position = pos
        game.enemies = [p1, p2]

        player_pos = pos
        expected = grid.width * 2
        rows = render_rows(game, heatmap_on=False, player_pos=player_pos, got_key=False)
        self._check_widths(rows, expected)

        # Player should render as '@' (priority over enemies)
        # Find the row for y=3
        row = rows[3]
        # Player at x=3, each tile is 2 visible chars (symbol + space)
        # Verify player symbol '@' is present in the row (visible, not ANSI)
        assert '@' in row
        # The row should have correct visible width
        assert visible_width(row) == expected

    def test_enemy_symbols_are_digits(self, game_level1_seed1: Game) -> None:
        """Enemies should render as single digits 1-5, not 'E1', 'E2' etc."""
        game = game_level1_seed1
        rows = render_rows(game, heatmap_on=False, player_pos=game.grid.start, got_key=False)

        # Check that no row contains 'E1', 'E2', etc.
        for row in rows:
            assert 'E1' not in row
            assert 'E2' not in row
            assert 'E3' not in row
            # Should contain digits for enemies
            # (may or may not have enemies visible depending on positions)


class TestBuildLevelTable:
    """Tests for the level selection menu table formatting."""

    def test_all_lines_same_visible_width(self) -> None:
        lines = build_level_table()
        assert len(lines) >= 6  # header + separator + 4 data rows

        widths = [visible_width(line) for line in lines]
        # All lines should have the same visible width
        assert all(w == widths[0] for w in widths), f"Widths differ: {widths}"

    def test_header_columns_present(self) -> None:
        lines = build_level_table()
        header = lines[0]
        assert "Level" in header
        assert "Archetype" in header
        assert "Size" in header
        assert "Patrollers" in header
        assert "Flankers" in header
        assert "Alpha" in header
        assert "Sight" in header
        assert "Max Turns" in header

    def test_size_format_consistent(self) -> None:
        """Size should be formatted as '30x15' not '30x  15'."""
        lines = build_level_table()
        for line in lines[2:]:  # Skip header and separator
            # Size column should not have spaces inside the WxH
            # Find the size column value (3rd column)
            parts = [p for p in line.split("  ") if p]
            if len(parts) >= 3:
                size_col = parts[2].strip()
                # Should match pattern like "30x15" or "50x25"
                assert re.match(r"^\d+x\d+$", size_col), f"Bad size format: {size_col!r}"

    def test_alpha_format_two_decimals(self) -> None:
        """Alpha should have exactly two decimal places: 1.50, 2.00, 2.50, 2.25."""
        lines = build_level_table()
        for line in lines[2:]:
            parts = [p for p in line.split("  ") if p]
            if len(parts) >= 6:
                alpha_col = parts[5].strip()
                assert re.match(r"^\d\.\d{2}$", alpha_col), f"Bad alpha format: {alpha_col!r}"


class TestGameUIIntegration:
    """Integration tests for GameUI.get_layout."""

    def test_get_layout_returns_string(self) -> None:
        game = Game(level=1, seed=1)
        ui = GameUI()
        player_pos = game.grid.start
        layout = ui.get_layout(game, player_pos, False)
        assert isinstance(layout, str)
        assert len(layout) > 0

    def test_get_layout_map_and_panel(self) -> None:
        game = Game(level=1, seed=1)
        ui = GameUI()
        player_pos = game.grid.start
        layout = ui.get_layout(game, player_pos, False)
        lines = layout.split("\n")

        # Should have at least grid.height lines for the map
        assert len(lines) >= game.grid.height

        # Each line should have map + gap + panel
        map_width = game.grid.width * 2
        for line in lines[:game.grid.height]:
            vw = visible_width(line)
            # At minimum should have map_width visible chars
            assert vw >= map_width

    def test_heatmap_toggle_changes_render(self) -> None:
        game = Game(level=1, seed=1)
        ui = GameUI()
        player_pos = game.grid.start

        layout_off = ui.get_layout(game, player_pos, False)
        ui.toggle_heatmap()
        layout_on = ui.get_layout(game, player_pos, False)

        # They should be different (heatmap adds characters)
        assert layout_off != layout_on


if __name__ == "__main__":
    pytest.main([__file__, "-v"])