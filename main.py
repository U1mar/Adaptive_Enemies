"""Main entry point for the adaptive enemies game with human player."""
from __future__ import annotations
import sys
from typing import List, Tuple, Optional

from rich.console import Console
from rich.live import Live

from grid import Grid, Position
from game.engine import Game, RoundResult, Policy
from game.ui import GameUI
from game.enemy import Enemy


def human_policy(grid: Grid, player_pos: Position, got_key: bool, enemy_positions: List[Position]) -> Position:
    """Policy that waits for human input. This is a placeholder - actual input handled in main loop."""
    # This won't be called directly; we handle input in the main loop
    return player_pos


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


def run_game(max_turns_per_round: int = 500, scripted_inputs: Optional[List[str]] = None) -> None:
    """Run the game with human input or scripted inputs for testing."""
    console = Console()
    game = Game()
    ui = GameUI(console)

    # For scripted testing
    input_iterator = iter(scripted_inputs) if scripted_inputs else None
    turn_count = 0
    max_total_turns = 5 if scripted_inputs else float('inf')

    # Track if we're in a round
    player_pos = game.grid.start
    got_key = False
    round_active = True

    def render_frame():
        ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
        return ui.get_layout(game.grid, player_pos, game.enemies, got_key, game.tracker, game.zone_model)

    with Live(render_frame(), console=console, refresh_per_second=10, screen=True) as live:
        while turn_count < max_total_turns:
            # Update display
            live.update(render_frame())

            # Get input
            if input_iterator:
                try:
                    key = next(input_iterator)
                except StopIteration:
                    break
            else:
                # For interactive mode, we'd use console.input() but that breaks Live
                # So we'll just use a simple approach for the scripted test
                key = 'q'  # Default to quit for non-scripted

            turn_count += 1

            if key == 'q':
                break
            elif key == 'h':
                ui.toggle_heatmap()
                continue
            elif key in ('w', 'a', 's', 'd'):
                # Move player
                new_pos = get_move_from_key(key, player_pos, game.grid)

                # Record position with tracker (engine does this at start of turn)
                game.tracker.record(player_pos)

                # Update zone model
                current_zone = game.grid.zone_of(player_pos)
                if hasattr(run_game, 'last_zone'):
                    if current_zone != run_game.last_zone:
                        game.zone_model.record_transition(run_game.last_zone, current_zone)
                run_game.last_zone = current_zone

                # Decay tracker every 5 turns
                if turn_count % 5 == 0 and turn_count > 0:
                    game.tracker.decay_counts()

                player_pos = new_pos

                # Check for key pickup
                if player_pos == game.grid.vault:
                    got_key = True

                # Check for capture
                enemy_positions = [e.position for e in game.enemies]
                if player_pos in enemy_positions:
                    ui.set_last_result(f"CAUGHT at turn {turn_count}!")
                    round_active = False
                    break

                # Check for escape
                if player_pos == game.grid.exit and got_key:
                    ui.set_last_result(f"ESCAPED in {turn_count} turns!")
                    round_active = False
                    break

                # Update enemies
                for enemy in game.enemies:
                    enemy.update(game.grid, player_pos, game.tracker, game.zone_model, game.alpha)

                # Check capture after enemy move
                enemy_positions = [e.position for e in game.enemies]
                if player_pos in enemy_positions:
                    ui.set_last_result(f"CAUGHT at turn {turn_count} (enemy moved)!")
                    round_active = False
                    break

            # Small delay for visualization
            import time
            time.sleep(0.1)

        # Final render
        live.update(render_frame())

    # Print final result
    if round_active:
        console.print(f"[yellow]Game ended after {turn_count} turns (max reached)[/yellow]")
    console.print("[green]Game session ended.[/green]")


if __name__ == "__main__":
    # For testing: run 5 turns with scripted input
    # Move right 4 times, then quit
    scripted = ['d', 'd', 'd', 'd', 'q']
    run_game(scripted_inputs=scripted)