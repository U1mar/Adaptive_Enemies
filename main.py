"""Main entry point for the adaptive enemies game with human player (interactive)."""
from __future__ import annotations
import sys
from typing import List, Tuple, Optional
import random
import os

import readchar

from grid import Grid, Position
from game.engine import Game
from game.ui import GameUI, build_level_table
from game.levels import get_num_levels


def get_move_from_key(key: str, player_pos: Position, grid: Grid) -> Position:
    """Convert key press to new position."""
    x, y = player_pos
    moves = {
        'w': (x, y - 1),
        'a': (x - 1, y),
        's': (x, y + 1),
        'd': (x + 1, y),
    }
    new_pos = moves.get(key.lower(), player_pos)
    if grid.is_walkable(new_pos):
        return new_pos
    return player_pos


def select_level() -> Tuple[int, int]:
    """Display level selection menu and return selected (level_number, seed)."""
    num_levels = get_num_levels()

    # Build table using the new aligned function
    table_lines = build_level_table()

    # Print header
    print("=" * 80)
    print("Select Level")
    print("=" * 80)
    for line in table_lines:
        print(line)
    print()

    while True:
        try:
            choice = input(f"Select level (1-{num_levels}) [1]: ").strip()
            if choice == "":
                choice = "1"
            level = int(choice)
            if 1 <= level <= num_levels:
                break
            print(f"Please enter a number between 1 and {num_levels}")
        except ValueError:
            print("Please enter a valid number")

    # Seed selection
    print()
    while True:
        try:
            seed_input = input("Enter seed (leave empty for random): ").strip()
            if seed_input == "":
                seed = random.randint(0, 2**31 - 1)
                print(f"Using random seed: {seed}")
                return level, seed
            seed = int(seed_input)
            if seed < 0:
                print("Seed must be non-negative")
                continue
            return level, seed
        except ValueError:
            print("Please enter a valid integer seed")


def run_interactive() -> None:
    """Run the game with human input from CLI using plain text rendering.
    The same Game, tracker, and zone_model persist across rounds until the player quits.
    """
    # Enable ANSI on Windows
    os.system("")

    # Level and seed selection
    level, seed = select_level()
    game = Game(level=level, seed=seed)
    ui = GameUI()

    # Initial UI state
    player_pos = game.grid.start
    ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))

    # Clear screen once at start, hide cursor
    ui.clear_screen_once()
    ui.hide_cursor()

    try:
        # Show initial info
        print("=" * 60)
        print("Adaptive Enemies")
        print("=" * 60)
        print("Controls: W/A/S/D = move  |  H = toggle heatmap  |  Q = quit")
        print(f"Goal: Reach the vault (V), then exit (E) without getting caught")
        print(f"Level: {level}  |  Seed: {seed}  |  Archetype: {game.level_obj.archetype}")
        print("=" * 60)
        print()

        # Initial frame
        layout = ui.get_layout(game, player_pos, False)
        ui.draw_frame(layout)

        while True:
            # Use readchar for single keypress (no Enter needed)
            key = readchar.readkey().lower()

            if key == 'q':
                print("\nQuitting...")
                break
            elif key == 'h':
                ui.toggle_heatmap()
                # Re-render with updated heatmap
                ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
                layout = ui.get_layout(game, player_pos, game.got_key)
                ui.draw_frame(layout)
                continue
            elif key not in ('w', 'a', 's', 'd'):
                # Invalid key - show error briefly then redraw
                layout = ui.get_layout(game, player_pos, game.got_key)
                ui.draw_frame(layout)
                continue

            # Compute intended move
            new_pos = get_move_from_key(key, player_pos, game.grid)
            if new_pos == player_pos:
                # Can't move there - redraw same frame
                layout = ui.get_layout(game, player_pos, game.got_key)
                ui.draw_frame(layout)
                continue

            # Step the game (handles tracker, zone_model, enemies, capture, key, escape)
            caught, got_key, escaped = game.step(new_pos)
            player_pos = new_pos

            if caught:
                ui.set_last_result(f"CAUGHT at turn {game.turn_count}!")
                ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
                layout = ui.get_layout(game, player_pos, got_key)
                ui.draw_frame(layout)
                print()  # New line after frame for game over message
                print("=" * 60)
                print("GAME OVER - CAUGHT!")
                print("=" * 60)
                # Start new round: reset enemies but keep tracker/zone_model
                game.reset_enemies()
                player_pos = game.grid.start
                ui.next_round()
                ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
                layout = ui.get_layout(game, player_pos, False)
                print()
                ui.draw_frame(layout)
                continue

            if escaped:
                ui.set_last_result(f"ESCAPED in {game.turn_count} turns!")
                ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
                layout = ui.get_layout(game, player_pos, got_key)
                ui.draw_frame(layout)
                print()
                print("=" * 60)
                print("VICTORY - ESCAPED!")
                print("=" * 60)
                # Start new round
                game.reset_enemies()
                player_pos = game.grid.start
                ui.next_round()
                ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
                layout = ui.get_layout(game, player_pos, False)
                print()
                ui.draw_frame(layout)
                continue

            # Normal continue: re-render
            ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
            layout = ui.get_layout(game, player_pos, game.got_key)
            ui.draw_frame(layout)

    finally:
        # Always restore cursor on exit
        ui.show_cursor()


if __name__ == "__main__":
    run_interactive()