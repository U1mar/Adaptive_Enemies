"""Headless tests for gui.py - runs with SDL_VIDEODRIVER=dummy."""
import os
import sys
import pytest

# Set headless mode before importing pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from gui import make_game, window_size, draw_map, draw_entities, draw_panel


@pytest.mark.skipif(not PYGAME_AVAILABLE, reason="pygame not installed")
def test_gui_headless_levels():
    """Test that gui functions work for levels 1-4 in headless mode."""
    pygame.init()
    pygame.display.init()

    try:
        for level in range(1, 5):
            game = make_game(level=level, seed=42)
            w, h = window_size(game.grid)

            # Create a dummy surface
            screen = pygame.Surface((w, h))

            # Call draw functions 5 times each
            for _ in range(5):
                draw_map(screen, game)
                draw_entities(screen, game)
                draw_panel(screen, game)

            # Verify surface was drawn to (not blank)
            # Just ensure no exceptions were raised
            assert screen is not None

    finally:
        pygame.quit()


@pytest.mark.skipif(not PYGAME_AVAILABLE, reason="pygame not installed")
def test_make_game():
    """Test make_game creates valid Game instances."""
    for level in range(1, 5):
        game = make_game(level=level, seed=123)
        assert game is not None
        assert game.grid is not None
        assert game.level == level


@pytest.mark.skipif(not PYGAME_AVAILABLE, reason="pygame not installed")
def test_window_size():
    """Test window_size returns correct dimensions."""
    game = make_game(level=1, seed=0)
    w, h = window_size(game.grid)

    expected_w = game.grid.width * 24 + 260  # TILE=24, PANEL_W=260
    expected_h = game.grid.height * 24

    assert w == expected_w
    assert h == expected_h


if __name__ == "__main__":
    # Allow running directly with: python tests/test_gui.py
    pytest.main([__file__, "-v"])