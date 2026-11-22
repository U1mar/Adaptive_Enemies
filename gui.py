"""Pygame GUI for the Adaptive Enemies game."""
from __future__ import annotations

import os
import sys
from typing import Tuple, Optional

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore

from grid import Grid, Position
from game.engine import Game


# ============================================================
# CONSTANTS
# ============================================================

TILE = 24
PANEL_W = 260

# Colours (R, G, B)
COL_WALL = (40, 40, 50)       # dark grey
COL_FLOOR = (30, 35, 50)      # dark blue-grey
COL_START = (100, 200, 100)   # green
COL_VAULT = (180, 100, 220)   # purple
COL_EXIT = (80, 220, 80)      # green (bright)
COL_PLAYER = (50, 200, 220)   # cyan
COL_ENEMY_1 = (220, 60, 60)   # red
COL_ENEMY_2 = (230, 140, 40)  # orange
COL_ENEMY_3 = (220, 200, 50)  # yellow
COL_ENEMIES = [COL_ENEMY_1, COL_ENEMY_2, COL_ENEMY_3]

# Panel colours
COL_PANEL_BG = (20, 22, 30)
COL_PANEL_TEXT = (200, 200, 210)
COL_PANEL_ACCENT = (120, 180, 255)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_game(level: int = 1, seed: Optional[int] = None) -> Game:
    """Create a Game instance for the given level and seed."""
    return Game(level=level, seed=seed)


def window_size(grid: Grid) -> Tuple[int, int]:
    """Calculate window size: map width + side panel."""
    map_w = grid.width * TILE
    map_h = grid.height * TILE
    return map_w + PANEL_W, map_h


def draw_map(screen: "pygame.Surface", game: Game) -> None:
    """Draw the map tiles: walls, floor, start, vault, exit."""
    grid = game.grid
    for y in range(grid.height):
        for x in range(grid.width):
            pos = (x, y)
            rect = pygame.Rect(x * TILE, y * TILE, TILE, TILE)

            if pos in grid.walls:
                pygame.draw.rect(screen, COL_WALL, rect)
            else:
                pygame.draw.rect(screen, COL_FLOOR, rect)
                # Special tiles
                if pos == grid.start:
                    pygame.draw.rect(screen, COL_START, rect)
                elif pos == grid.vault:
                    pygame.draw.rect(screen, COL_VAULT, rect)
                elif pos == grid.exit:
                    pygame.draw.rect(screen, COL_EXIT, rect)

            # Grid lines (subtle)
            pygame.draw.rect(screen, (25, 28, 38), rect, 1)


def draw_entities(screen: "pygame.Surface", game: Game) -> None:
    """Draw entities (player, enemies). Placeholder for now."""
    # Player
    player_pos = game.grid.start  # Placeholder - actual position tracked elsewhere
    # TODO: Get actual player position from game state when available
    # For now, draw at start as placeholder
    player_rect = pygame.Rect(
        player_pos[0] * TILE + 4,
        player_pos[1] * TILE + 4,
        TILE - 8,
        TILE - 8
    )
    pygame.draw.ellipse(screen, COL_PLAYER, player_rect)

    # Enemies
    for i, enemy in enumerate(game.enemies):
        color = COL_ENEMIES[i % len(COL_ENEMIES)]
        enemy_rect = pygame.Rect(
            enemy.position[0] * TILE + 4,
            enemy.position[1] * TILE + 4,
            TILE - 8,
            TILE - 8
        )
        pygame.draw.ellipse(screen, color, enemy_rect)


def draw_panel(screen: "pygame.Surface", game: Game) -> None:
    """Draw the side panel with game info."""
    grid = game.grid
    panel_x = grid.width * TILE
    panel_rect = pygame.Rect(panel_x, 0, PANEL_W, grid.height * TILE)
    pygame.draw.rect(screen, COL_PANEL_BG, panel_rect)
    pygame.draw.line(screen, (50, 55, 70), (panel_x, 0), (panel_x, grid.height * TILE), 2)

    font = pygame.font.Font(None, 20)
    small_font = pygame.font.Font(None, 18)
    y = 20

    # Title
    title = font.render(f"Level {game.level}", True, COL_PANEL_ACCENT)
    screen.blit(title, (panel_x + 15, y))
    y += 30

    # Archetype
    if hasattr(game, 'level_obj') and game.level_obj:
        arch = small_font.render(f"Archetype: {game.level_obj.archetype}", True, COL_PANEL_TEXT)
        screen.blit(arch, (panel_x + 15, y))
        y += 22

    # Seed
    if hasattr(game, 'level_obj') and game.level_obj:
        seed_text = small_font.render(f"Seed: {game.level_obj.seed}", True, COL_PANEL_TEXT)
        screen.blit(seed_text, (panel_x + 15, y))
        y += 22

    y += 10

    # Legend
    legend_title = font.render("Legend", True, COL_PANEL_ACCENT)
    screen.blit(legend_title, (panel_x + 15, y))
    y += 28

    legend_items = [
        (COL_WALL, "Wall"),
        (COL_FLOOR, "Floor"),
        (COL_START, "Start (S)"),
        (COL_VAULT, "Vault (V)"),
        (COL_EXIT, "Exit (E)"),
        (COL_PLAYER, "Player"),
    ]

    for color, label in legend_items:
        pygame.draw.rect(screen, color, (panel_x + 15, y + 2, 16, 16))
        text = small_font.render(label, True, COL_PANEL_TEXT)
        screen.blit(text, (panel_x + 38, y))
        y += 22

    y += 10

    # Enemy legend
    enemy_title = font.render("Enemies", True, COL_PANEL_ACCENT)
    screen.blit(enemy_title, (panel_x + 15, y))
    y += 28

    for i, enemy in enumerate(game.enemies):
        color = COL_ENEMIES[i % len(COL_ENEMIES)]
        state = enemy.state.value.capitalize() if hasattr(enemy.state, 'value') else str(enemy.state)
        pygame.draw.rect(screen, color, (panel_x + 15, y + 2, 16, 16))
        label = small_font.render(f"{type(enemy).__name__} ({state})", True, COL_PANEL_TEXT)
        screen.blit(label, (panel_x + 38, y))
        y += 22

    y += 10

    # Controls
    controls_title = font.render("Controls", True, COL_PANEL_ACCENT)
    screen.blit(controls_title, (panel_x + 15, y))
    y += 28

    controls = [
        "ESC / Q - Quit",
        "R - Restart level",
        "SPACE - Step (TODO)",
    ]
    for ctrl in controls:
        text = small_font.render(ctrl, True, (150, 150, 160))
        screen.blit(text, (panel_x + 15, y))
        y += 20


def run(level: int = 1, seed: Optional[int] = None) -> None:
    """Open the Pygame window and run the main loop."""
    if pygame is None:
        raise RuntimeError("pygame not installed. Install with: pip install pygame")

    # Headless mode support
    is_headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"

    pygame.init()
    if is_headless:
        pygame.display.init()

    game = make_game(level, seed)
    w, h = window_size(game.grid)

    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f"Adaptive Enemies - Level {level}")

    clock = pygame.time.Clock()
    running = True

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    # Restart level
                    game = make_game(level, seed)

        # Draw
        screen.fill(COL_FLOOR)
        draw_map(screen, game)
        draw_entities(screen, game)
        draw_panel(screen, game)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    # Allow running with: python gui.py [level] [seed]
    lvl = 1
    sd = None
    if len(sys.argv) > 1:
        lvl = int(sys.argv[1])
    if len(sys.argv) > 2:
        sd = int(sys.argv[2])
    run(lvl, sd)