"""Pygame GUI for the Adaptive Enemies game - playable version."""
from __future__ import annotations

import os
import sys
import random
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

# UI colours
COL_PANEL_BG = (20, 22, 30)
COL_PANEL_TEXT = (200, 200, 210)
COL_PANEL_ACCENT = (120, 180, 255)
COL_CAUGHT = (220, 60, 60)     # red
COL_ESCAPED = (80, 220, 80)    # green
COL_STATUS_BG = (15, 15, 20)
COL_STATUS_TEXT = (200, 200, 210)

# Key mapping
MOVE_KEYS = {
    pygame.K_w: (0, -1), pygame.K_UP: (0, -1),
    pygame.K_s: (0, 1), pygame.K_DOWN: (0, 1),
    pygame.K_a: (-1, 0), pygame.K_LEFT: (-1, 0),
    pygame.K_d: (1, 0), pygame.K_RIGHT: (1, 0),
    pygame.K_SPACE: (0, 0),  # wait
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_game(level: int = 1, seed: Optional[int] = None) -> Game:
    """Create a Game instance for the given level and seed.
    If seed is None, uses a random seed (matching CLI behavior).
    """
    import random
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    return Game(level=level, seed=seed)


def window_size(grid: Grid) -> Tuple[int, int]:
    """Calculate window size: map width + side panel."""
    map_w = grid.width * TILE
    map_h = grid.height * TILE
    return map_w + PANEL_W, map_h


def draw_map(screen: "pygame.Surface", game: Game) -> None:
    """Draw the map tiles: walls, floor, start, vault (if no key), exit."""
    grid = game.grid
    got_key = game.got_key
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
                elif pos == grid.vault and not got_key:
                    pygame.draw.rect(screen, COL_VAULT, rect)
                elif pos == grid.exit:
                    pygame.draw.rect(screen, COL_EXIT, rect)

            # Grid lines (subtle)
            pygame.draw.rect(screen, (25, 28, 38), rect, 1)


def draw_entities(screen: "pygame.Surface", game: Game, player_pos: Position) -> None:
    """Draw entities: player (cyan circle), enemies (colored circles with state letter)."""
    font = pygame.font.Font(None, 18)

    # Player
    player_rect = pygame.Rect(
        player_pos[0] * TILE + 4,
        player_pos[1] * TILE + 4,
        TILE - 8,
        TILE - 8
    )
    pygame.draw.ellipse(screen, COL_PLAYER, player_rect)

    # Enemies with state letters
    for i, enemy in enumerate(game.enemies):
        color = COL_ENEMIES[i % len(COL_ENEMIES)]
        enemy_rect = pygame.Rect(
            enemy.position[0] * TILE + 4,
            enemy.position[1] * TILE + 4,
            TILE - 8,
            TILE - 8
        )
        pygame.draw.ellipse(screen, color, enemy_rect)

        # Draw state letter in center: P=Patrol, C=Chase, F=Flank
        state_char = enemy.state.value[0].upper()  # 'P', 'C', or 'F'
        text = font.render(state_char, True, (255, 255, 255))
        text_rect = text.get_rect(center=enemy_rect.center)
        screen.blit(text, text_rect)


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
        "WASD / Arrows - Move",
        "SPACE - Wait",
        "R - Restart round",
        "ESC / Q - Quit",
    ]
    for ctrl in controls:
        text = small_font.render(ctrl, True, (150, 150, 160))
        screen.blit(text, (panel_x + 15, y))
        y += 20


def draw_status_bar(screen: "pygame.Surface", game: Game, player_pos: Position) -> None:
    """Draw status bar under the map with round, turn, key status."""
    grid = game.grid
    map_h = grid.height * TILE
    bar_rect = pygame.Rect(0, map_h, grid.width * TILE, 30)
    pygame.draw.rect(screen, COL_STATUS_BG, bar_rect)
    pygame.draw.line(screen, (50, 55, 70), (0, map_h), (grid.width * TILE, map_h), 2)

    font = pygame.font.Font(None, 20)
    key_status = "YES" if game.got_key else "NO"
    key_color = COL_ESCAPED if game.got_key else COL_CAUGHT
    status_text = f"Round: {game.round_number + 1}   Turn: {game.turn_count}   Key: {key_status}"
    text = font.render(status_text, True, COL_STATUS_TEXT)
    screen.blit(text, (10, map_h + 5))

    # Draw key indicator as colored box
    key_rect = pygame.Rect(grid.width * TILE - 80, map_h + 5, 70, 20)
    pygame.draw.rect(screen, key_color, key_rect)
    key_label = font.render("VAULT KEY", True, (255, 255, 255))
    key_label_rect = key_label.get_rect(center=key_rect.center)
    screen.blit(key_label, key_label_rect)


def draw_banner(screen: "pygame.Surface", game: Game, caught: bool, escaped: bool) -> None:
    """Draw CAUGHT/ESCAPED banner over the map."""
    grid = game.grid
    map_w = grid.width * TILE
    map_h = grid.height * TILE

    # Semi-transparent overlay
    overlay = pygame.Surface((map_w, map_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.blit(overlay, (0, 0))

    # Banner text
    font = pygame.font.Font(None, 64)
    small_font = pygame.font.Font(None, 28)

    if caught:
        color = COL_CAUGHT
        main_text = "CAUGHT"
        detail = "Enemies caught you!" + (" (had key)" if game.got_key else " (no key)")
    else:
        color = COL_ESCAPED
        main_text = "ESCAPED"
        detail = "You reached the exit with the key!"

    main_surf = font.render(main_text, True, color)
    main_rect = main_surf.get_rect(center=(map_w // 2, map_h // 2 - 40))
    screen.blit(main_surf, main_rect)

    detail_surf = small_font.render(detail, True, COL_PANEL_TEXT)
    detail_rect = detail_surf.get_rect(center=(map_w // 2, map_h // 2 + 10))
    screen.blit(detail_surf, detail_rect)

    sub_surf = small_font.render("Press any key for next round", True, COL_PANEL_TEXT)
    sub_rect = sub_surf.get_rect(center=(map_w // 2, map_h // 2 + 50))
    screen.blit(sub_surf, sub_rect)


def handle_input(event: "pygame.event.Event", game: Game, player_pos: Position) -> Tuple[Position, bool, bool, bool, bool]:
    """
    Handle a key event. Returns (new_player_pos, round_ended, caught, escaped, should_quit).
    """
    if event.type != pygame.KEYDOWN:
        return player_pos, False, False, False, False

    # Quit
    if event.key in (pygame.K_ESCAPE, pygame.K_q):
        return player_pos, False, False, False, True

    # Restart round
    if event.key == pygame.K_r:
        game.reset_enemies()
        return game.grid.start, False, False, False, False

    # Movement keys
    if event.key in MOVE_KEYS:
        dx, dy = MOVE_KEYS[event.key]
        target = (player_pos[0] + dx, player_pos[1] + dy)

        # Only move if walkable
        if game.grid.is_walkable(target):
            caught, got_key, escaped = game.step(target)
            round_ended = caught or escaped

            if round_ended:
                # Round ended - return actual caught/escaped status
                return target, True, caught, escaped, False

            return target, False, False, False, False

    return player_pos, False, False, False, False


def run(level: int = 1, seed: Optional[int] = None) -> None:
    """Open the Pygame window and run the main game loop."""
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

    # Player starts at grid.start
    player_pos = game.grid.start

    # Round end state
    round_ended = False
    caught = False
    escaped = False

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if round_ended:
                    # Any key starts next round
                    game.reset_enemies()
                    player_pos = game.grid.start
                    round_ended = False
                    caught = False
                    escaped = False
                else:
                    player_pos, round_ended, caught, escaped, should_quit = handle_input(event, game, player_pos)
                    if should_quit:
                        running = False

        # Draw
        screen.fill(COL_FLOOR)
        draw_map(screen, game)
        draw_entities(screen, game, player_pos)
        draw_status_bar(screen, game, player_pos)
        draw_panel(screen, game)

        if round_ended:
            draw_banner(screen, game, caught, escaped)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def run_headless_simulation(level: int = 1, seed: Optional[int] = None, num_steps: int = 20) -> None:
    """Run a headless simulation with random key events for testing."""
    if pygame is None:
        raise RuntimeError("pygame not installed")

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.display.init()

    game = make_game(level, seed)
    w, h = window_size(game.grid)
    screen = pygame.Surface((w, h))

    player_pos = game.grid.start
    round_ended = False

    # Simulate random key events
    move_keys = list(MOVE_KEYS.keys())

    for _ in range(num_steps):
        # Pick random move key
        key = random.choice(move_keys)

        # Create a fake event
        event = pygame.event.Event(pygame.KEYDOWN, key=key)

        if round_ended:
            # Any key starts next round
            game.reset_enemies()
            player_pos = game.grid.start
            round_ended = False
        else:
            dx, dy = MOVE_KEYS[key]
            target = (player_pos[0] + dx, player_pos[1] + dy)

            if game.grid.is_walkable(target):
                caught, got_key, escaped = game.step(target)
                player_pos = target
                round_ended = caught or escaped

        # Draw (just to verify no crashes)
        screen.fill(COL_FLOOR)
        draw_map(screen, game)
        draw_entities(screen, game, player_pos)
        draw_status_bar(screen, game, player_pos)
        draw_panel(screen, game)

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