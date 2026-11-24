"""Pygame GUI for the Adaptive Enemies game - playable version."""
from __future__ import annotations

import os
import sys
import random
from typing import Tuple, Optional, List

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore

from grid import Grid, Position
from game.engine import Game
from game.levels import get_archetype_for_level, get_map_size_for_level, get_num_levels


# ============================================================
# CONSTANTS
# ============================================================

TILE = 24
PANEL_W = 260

# Colours (R, G, B)
COL_WALL = (40, 40, 50)       # dark grey
COL_FLOOR = (30, 35, 50)      # dark blue-grey
COL_START = (100, 150, 255)   # blue (distinct from exit/vault)
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

# UI toggle keys
TOGGLE_HEATMAP = pygame.K_h
TOGGLE_PATH_COMPARISON = pygame.K_p


def compute_optimal_path(grid: Grid) -> List[Position]:
    """Compute the optimal path from Start -> Vault -> Exit using A*."""
    from pathing import astar

    # Path from start to vault
    path1, _ = astar(grid, grid.start, grid.vault)
    if not path1:
        return []

    # Path from vault to exit
    path2, _ = astar(grid, grid.vault, grid.exit)
    if not path2:
        return path1

    # Combine (avoid duplicate vault)
    return path1 + path2[1:]


def draw_heatmap_overlay(screen: "pygame.Surface", game: Game, player_pos: Position) -> None:
    """Draw the tracker probability heatmap overlay on the map."""
    grid = game.grid
    tracker = game.tracker

    prob_map = tracker.probability_map()
    if not prob_map:
        return

    # Heatmap gradient colors (blue -> cyan -> yellow -> red)
    # Higher base alpha for better visibility
    MIN_ALPHA = 100   # Minimum alpha even for low probabilities
    MAX_ALPHA = 220   # Maximum alpha for high probabilities

    def get_heat_color(prob: float) -> Tuple[int, int, int, int]:
        # prob is 0-1, map to color with alpha
        if prob <= 0.0:
            return (0, 0, 0, 0)
        # Interpolate alpha between MIN_ALPHA and MAX_ALPHA based on probability
        alpha = int(MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * prob)

        if prob < 0.25:
            # Blue to cyan
            t = prob / 0.25
            return (0, int(255 * t), 255, alpha)
        elif prob < 0.5:
            # Cyan to yellow
            t = (prob - 0.25) / 0.25
            return (int(255 * t), 255, int(255 * (1 - t)), alpha)
        elif prob < 0.75:
            # Yellow to orange
            t = (prob - 0.5) / 0.25
            return (255, int(255 * (1 - t)), 0, alpha)
        else:
            # Orange to red
            t = (prob - 0.75) / 0.25
            return (255, int(100 * (1 - t)), 0, alpha)

    for pos, prob in prob_map.items():
        if prob <= 0.0:
            continue
        x, y = pos
        rect = pygame.Rect(x * TILE, y * TILE, TILE, TILE)
        color = get_heat_color(prob)
        # Create a surface with per-pixel alpha for the heatmap
        heat_surf = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
        heat_surf.fill(color)
        screen.blit(heat_surf, rect)


def draw_path_comparison(screen: "pygame.Surface", game: Game, player_path: List[Position],
                          optimal_path: List[Position]) -> None:
    """Draw both player path and optimal path for comparison."""
    # Draw optimal path in green (dashed/semi-transparent)
    if optimal_path:
        opt_color = (0, 255, 0, 180)  # Green with alpha
        for i, pos in enumerate(optimal_path):
            x, y = pos
            rect = pygame.Rect(x * TILE + 4, y * TILE + 4, TILE - 8, TILE - 8)
            # Create surface for semi-transparent path
            path_surf = pygame.Surface((TILE - 8, TILE - 8), pygame.SRCALPHA)
            path_surf.fill(opt_color)
            screen.blit(path_surf, rect)

            # Draw step number on optimal path
            if i % 3 == 0:  # Every 3rd step to avoid clutter
                font = pygame.font.Font(None, 14)
                step_text = font.render(str(i), True, (0, 200, 0))
                text_rect = step_text.get_rect(center=(x * TILE + TILE // 2, y * TILE + TILE // 2))
                screen.blit(step_text, text_rect)

    # Draw player's actual path in cyan
    if player_path:
        player_color = (0, 200, 255, 200)  # Cyan with alpha
        for i, pos in enumerate(player_path):
            x, y = pos
            rect = pygame.Rect(x * TILE + 2, y * TILE + 2, TILE - 4, TILE - 4)
            path_surf = pygame.Surface((TILE - 4, TILE - 4), pygame.SRCALPHA)
            path_surf.fill(player_color)
            screen.blit(path_surf, rect)

            # Draw step number on player path
            if i % 3 == 0:
                font = pygame.font.Font(None, 12)
                step_text = font.render(str(i), True, (0, 150, 255))
                text_rect = step_text.get_rect(center=(x * TILE + TILE // 2, y * TILE + TILE // 2))
                screen.blit(step_text, text_rect)

    # Draw legend for path comparison
    font = pygame.font.Font(None, 16)
    legend_y = 10
    # Optimal path legend
    pygame.draw.rect(screen, (0, 255, 0, 180), (10, legend_y, 20, 20))
    opt_label = font.render("Optimal Path", True, (0, 255, 0))
    screen.blit(opt_label, (35, legend_y))
    legend_y += 22
    # Player path legend
    pygame.draw.rect(screen, (0, 200, 255, 200), (10, legend_y, 20, 20))
    player_label = font.render("Your Path", True, (0, 200, 255))
    screen.blit(player_label, (35, legend_y))

# ============================================================
# MENU SYSTEM
# ============================================================

MENU_TITLE = "Adaptive Enemies"
MENU_FONT_SIZE = 48
MENU_ITEM_FONT_SIZE = 28
MENU_SMALL_FONT_SIZE = 20


def build_menu_level_entries() -> List[Tuple[int, str, str]]:
    """Build list of (level_number, archetype_name, size_string) for all levels."""
    num_levels = get_num_levels()
    entries = []
    for level in range(1, num_levels + 1):
        archetype = get_archetype_for_level(level)
        width, height = get_map_size_for_level(level)
        size_str = f"{width}x{height}"
        entries.append((level, archetype, size_str))
    return entries


def draw_menu(screen: "pygame.Surface", selected_idx: int, level_entries: List[Tuple[int, str, str]]) -> None:
    """Draw the start menu screen."""
    screen.fill(COL_FLOOR)

    # Fonts
    title_font = pygame.font.Font(None, MENU_FONT_SIZE)
    item_font = pygame.font.Font(None, MENU_ITEM_FONT_SIZE)
    small_font = pygame.font.Font(None, MENU_SMALL_FONT_SIZE)

    w, h = screen.get_size()
    center_x = w // 2

    # Title
    title_surf = title_font.render(MENU_TITLE, True, COL_PANEL_ACCENT)
    title_rect = title_surf.get_rect(center=(center_x, 80))
    screen.blit(title_surf, title_rect)

    # Subtitle
    subtitle = small_font.render("Select a level to begin", True, COL_PANEL_TEXT)
    subtitle_rect = subtitle.get_rect(center=(center_x, 130))
    screen.blit(subtitle, subtitle_rect)

    # Level entries - use fixed column layout to prevent text overlapping
    start_y = 180
    spacing = 50

    # Column positions
    col_level_x = center_x - 220
    col_archetype_x = center_x - 60
    col_size_x = center_x + 180

    for i, (level_num, archetype, size_str) in enumerate(level_entries):
        y = start_y + i * spacing
        is_selected = (i == selected_idx)

        # Highlight background for selected
        if is_selected:
            highlight_rect = pygame.Rect(center_x - 220, y - 5, 440, 40)
            pygame.draw.rect(screen, (60, 80, 120), highlight_rect, border_radius=5)
            pygame.draw.rect(screen, COL_PANEL_ACCENT, highlight_rect, 2, border_radius=5)

        # Level number (left column)
        level_text = f"Level {level_num}"
        level_surf = item_font.render(level_text, True, COL_PANEL_ACCENT if is_selected else COL_PANEL_TEXT)
        level_rect = level_surf.get_rect(midleft=(col_level_x, y + 15))
        screen.blit(level_surf, level_rect)

        # Archetype (middle column) - truncate if too long
        archetype_display = archetype.replace("_", " ").title()
        arch_surf = item_font.render(archetype_display, True, COL_PANEL_TEXT)
        # Limit archetype text width to prevent overlap with size column
        max_arch_width = col_size_x - col_archetype_x - 20
        if arch_surf.get_width() > max_arch_width:
            # Truncate with ellipsis
            while arch_surf.get_width() > max_arch_width and len(archetype_display) > 3:
                archetype_display = archetype_display[:-1]
                arch_surf = item_font.render(archetype_display + "...", True, COL_PANEL_TEXT)
        arch_rect = arch_surf.get_rect(midleft=(col_archetype_x, y + 15))
        screen.blit(arch_surf, arch_rect)

        # Size (right column, right-aligned)
        size_surf = small_font.render(f"({size_str})", True, (150, 150, 160))
        size_rect = size_surf.get_rect(midright=(col_size_x, y + 15))
        screen.blit(size_surf, size_rect)

    # Controls hint
    hint_text = "UP/DOWN: Select  |  ENTER: Start (auto seed)  |  H: Heatmap  |  P: Path Compare  |  ESC/Q: Quit"
    hint_surf = small_font.render(hint_text, True, (120, 120, 130))
    hint_rect = hint_surf.get_rect(center=(center_x, h - 50))
    screen.blit(hint_surf, hint_rect)


def handle_menu_event(event: "pygame.event.Event", selected_idx: int,
                      level_entries: List[Tuple[int, str, str]]) -> Tuple[int, Optional[int], Optional[int]]:
    """
    Handle a menu event.
    Returns: (new_selected_idx, chosen_level, chosen_seed)
    chosen_level/chosen_seed are None unless the game should start.
    Seed is always None (random) - automatically selected.
    """
    if event.type != pygame.KEYDOWN:
        return selected_idx, None, None

    if event.key in (pygame.K_UP, pygame.K_w):
        selected_idx = (selected_idx - 1) % len(level_entries)
    elif event.key in (pygame.K_DOWN, pygame.K_s):
        selected_idx = (selected_idx + 1) % len(level_entries)
    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
        # Start game with selected level, random seed
        level_num, _, _ = level_entries[selected_idx]
        return selected_idx, level_num, None  # None seed = random
    elif event.key in (pygame.K_ESCAPE, pygame.K_q):
        # Quit
        return selected_idx, -1, -1

    return selected_idx, None, None


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
    """Open the Pygame window and run the main game loop with menu system."""
    if pygame is None:
        raise RuntimeError("pygame not installed. Install with: pip install pygame")

    # Headless mode support
    is_headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"

    pygame.init()
    if is_headless:
        pygame.display.init()

    # Build level entries for menu
    level_entries = build_menu_level_entries()

    # Menu state
    selected_idx = level - 1 if 1 <= level <= len(level_entries) else 0
    in_menu = True

    # Game state (initialized when level selected)
    game = None
    player_pos = (0, 0)
    round_ended = False
    caught = False
    escaped = False
    show_round_end_menu = False

    # UI state
    show_heatmap = False
    show_path_comparison = False
    player_path: List[Position] = []  # Track player's path for comparison
    optimal_path: List[Position] = []  # Pre-computed optimal path

    clock = pygame.time.Clock()
    running = True

    # Initial screen for menu
    screen = pygame.display.set_mode((800, 500))
    pygame.display.set_caption("Adaptive Enemies")

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if event.type != pygame.KEYDOWN:
                continue

            # In main menu
            if in_menu:
                selected_idx, chosen_level, chosen_seed = handle_menu_event(
                    event, selected_idx, level_entries
                )
                if chosen_level == -1:  # Quit
                    running = False
                    continue
                if chosen_level is not None:
                    # Start game with random seed (chosen_seed is always None)
                    in_menu = False
                    game = make_game(chosen_level, None)  # Always random seed
                    w, h = window_size(game.grid)
                    screen = pygame.display.set_mode((w, h))
                    pygame.display.set_caption(f"Adaptive Enemies - Level {chosen_level}")
                    player_pos = game.grid.start
                    round_ended = False
                    caught = False
                    escaped = False
                    show_round_end_menu = False
                    # Reset UI state and path tracking for new game
                    show_heatmap = False
                    show_path_comparison = False
                    player_path = [player_pos]  # Start with initial position
                    optimal_path = compute_optimal_path(game.grid)
                continue

            # In round end menu
            if show_round_end_menu:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    # Next round - keep learning, reset enemies
                    game.reset_enemies()
                    player_pos = game.grid.start
                    round_ended = False
                    caught = False
                    escaped = False
                    show_round_end_menu = False
                    # Reset path tracking for new round
                    player_path = [player_pos]
                elif event.key == pygame.K_m:
                    # Back to main menu
                    in_menu = True
                    show_round_end_menu = False
                    screen = pygame.display.set_mode((800, 500))
                    pygame.display.set_caption("Adaptive Enemies")
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                continue

            # Normal gameplay
            if round_ended:
                # Should not reach here if show_round_end_menu is handled, but fallback
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    game.reset_enemies()
                    player_pos = game.grid.start
                    round_ended = False
                    caught = False
                    escaped = False
                    player_path = [player_pos]
                elif event.key == pygame.K_m:
                    in_menu = True
                    screen = pygame.display.set_mode((800, 500))
                    pygame.display.set_caption("Adaptive Enemies")
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                continue

            # Handle gameplay input
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.key == pygame.K_r:
                game.reset_enemies()
                player_pos = game.grid.start
                player_path = [player_pos]  # Reset path on restart
            elif event.key == TOGGLE_HEATMAP:
                show_heatmap = not show_heatmap
            elif event.key == TOGGLE_PATH_COMPARISON:
                show_path_comparison = not show_path_comparison
            elif event.key in MOVE_KEYS:
                dx, dy = MOVE_KEYS[event.key]
                target = (player_pos[0] + dx, player_pos[1] + dy)

                if game.grid.is_walkable(target):
                    caught, got_key, escaped = game.step(target)
                    player_pos = target
                    player_path.append(player_pos)  # Track player's path
                    round_ended = caught or escaped
                    if round_ended:
                        show_round_end_menu = True

        # Draw
        if in_menu:
            # Menu screen - use fixed size
            if screen.get_size() != (800, 500):
                screen = pygame.display.set_mode((800, 500))
            draw_menu(screen, selected_idx, level_entries)
        else:
            # Game screen
            screen.fill(COL_FLOOR)
            draw_map(screen, game)
            draw_entities(screen, game, player_pos)

            # Draw heatmap overlay if enabled
            if show_heatmap:
                draw_heatmap_overlay(screen, game, player_pos)

            # Draw path comparison if enabled
            if show_path_comparison:
                draw_path_comparison(screen, game, player_path, optimal_path)

            draw_status_bar(screen, game, player_pos)
            draw_panel(screen, game)

            if round_ended:
                draw_banner(screen, game, caught, escaped)
                # Draw round end menu options
                draw_round_end_menu(screen, game, caught, escaped)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def draw_round_end_menu(screen: "pygame.Surface", game: Game, caught: bool, escaped: bool) -> None:
    """Draw the round end menu overlay with options."""
    grid = game.grid
    map_w = grid.width * TILE
    map_h = grid.height * TILE

    # Semi-transparent overlay
    overlay = pygame.Surface((map_w, map_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    screen.blit(overlay, (0, 0))

    font = pygame.font.Font(None, 48)
    small_font = pygame.font.Font(None, 28)

    if caught:
        color = COL_CAUGHT
        main_text = "CAUGHT"
    else:
        color = COL_ESCAPED
        main_text = "ESCAPED"

    main_surf = font.render(main_text, True, color)
    main_rect = main_surf.get_rect(center=(map_w // 2, map_h // 2 - 80))
    screen.blit(main_surf, main_rect)

    # Options
    options = [
        ("ENTER", "Next Round (learning kept)"),
        ("M", "Main Menu"),
        ("Q / ESC", "Quit"),
    ]

    y = map_h // 2
    for key, desc in options:
        key_surf = small_font.render(key, True, COL_PANEL_ACCENT)
        desc_surf = small_font.render(desc, True, COL_PANEL_TEXT)
        key_rect = key_surf.get_rect(midright=(map_w // 2 - 20, y))
        desc_rect = desc_surf.get_rect(midleft=(map_w // 2 + 20, y))
        screen.blit(key_surf, key_rect)
        screen.blit(desc_surf, desc_rect)
        y += 40


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