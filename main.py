"""Main entry point: terminal game or batch mode."""
from __future__ import annotations
import sys
import argparse
import os
from typing import List, Tuple

from grid import Grid, Position, DEFAULT_MAP
from engine import Game
from run_batch import main as run_batch_main


def human_policy(grid: Grid, player_pos: Position, got_key: bool,
                 enemy_positions: List[Position]) -> Position:
    """Human player policy - reads from stdin."""
    raise NotImplementedError("Use play_human() instead")


def get_key_cross_platform() -> str:
    """Get a single keypress cross-platform (Windows/Unix)."""
    if os.name == 'nt':
        import msvcrt
        ch = msvcrt.getch()
        if isinstance(ch, bytes):
            ch = ch.decode('utf-8', errors='ignore')
        return ch
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def play_human():
    """Play the game in terminal with w/a/s/d controls."""
    game = Game()
    player_pos = game.grid.start
    got_key = False
    show_heatmap = False
    turn = 0
    last_zone = game.grid.zone_of(player_pos)

    # Key mappings
    key_map = {
        'w': (0, -1),  # up
        's': (0, 1),   # down
        'a': (-1, 0),  # left
        'd': (1, 0),   # right
    }

    print("=== Adaptive Enemies ===")
    print("Controls: w/a/s/d to move, h to toggle heatmap, q to quit")
    print("Goal: Get key (V) then reach exit (E) without being caught (X)")
    print()

    while True:
        # Show predicted zones
        pred_zones = {}
        for enemy in game.enemies:
            if hasattr(enemy, 'wait_position') and enemy.wait_position:
                zone = game.grid.zone_of(enemy.wait_position)
                pred_zones[zone] = pred_zones.get(zone, 0) + 1
        if pred_zones:
            print(f"Predicted hot zones: {', '.join(f'{z}({c})' for z, c in pred_zones.items())}")

        # Render grid
        heatmap_data = game.tracker.probability_map() if show_heatmap else None
        print(game.grid.to_ascii(player_pos, [e.position for e in game.enemies],
                                  show_heatmap, heatmap_data))
        print(f"Turn: {turn} | Key: {'Yes' if got_key else 'No'} | Heatmap: {'On' if show_heatmap else 'Off'}")

        # Get input
        key = get_key_cross_platform().lower()
        if key == 'q':
            print("Quitting...")
            break
        elif key == 'h':
            show_heatmap = not show_heatmap
            continue
        elif key not in key_map:
            continue

        # Calculate next position
        dx, dy = key_map[key]
        next_pos = (player_pos[0] + dx, player_pos[1] + dy)

        # Validate
        if not game.grid.is_walkable(next_pos):
            continue

        # Record and update tracker
        game.tracker.record(player_pos)
        current_zone = game.grid.zone_of(player_pos)
        if current_zone != last_zone:
            game.zone_model.record_transition(last_zone, current_zone)
            last_zone = current_zone
        turn += 1

        # Decay every 5 turns
        if turn % 5 == 0:
            game.tracker.decay_counts()

        # Move player
        player_pos = next_pos

        # Check key
        if player_pos == game.grid.vault:
            got_key = True
            print("Got the key!")

        # Check capture
        if player_pos in [e.position for e in game.enemies]:
            print(game.grid.to_ascii(player_pos, [e.position for e in game.enemies]))
            print("CAUGHT! Game Over.")
            break

        # Check escape
        if player_pos == game.grid.exit and got_key:
            print(game.grid.to_ascii(player_pos, [e.position for e in game.enemies]))
            print("ESCAPED! You win!")
            break

        # Update enemies
        for enemy in game.enemies:
            enemy.update(game.grid, player_pos, game.tracker, game.zone_model, game.alpha)

    # Show final stats
    print(f"\nGame Over. Turns: {turn}")
    print(f"Zones visited: {game.zone_model.transitions}")


def main():
    parser = argparse.ArgumentParser(description='Adaptive Enemies Grid Game')
    parser.add_argument('--bots', action='store_true', help='Run batch mode with bots')
    args = parser.parse_args()

    if args.bots:
        run_batch_main()
    else:
        play_human()


if __name__ == '__main__':
    main()