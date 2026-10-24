"""Main entry point for the adaptive enemies game with human player (interactive)."""
from __future__ import annotations
import sys
from typing import List, Tuple, Optional

import readchar
from rich.console import Console
from rich.panel import Panel
from rich.live import Live

from grid import Grid, Position
from game.engine import Game
from game.ui import GameUI
from game.enemy import Enemy


def get_move_from_key(key: str, player_pos: Position, grid: Grid) -> Position:
    """Convert key press to new position."""
    x, y = player_pos
    moves = {
        'w': (x, y - 1),
        'a': (x - 1, y),
        's': (x, y + 1),
        'd': (x, y + 1),
    }
    new_pos = moves.get(key.lower(), player_pos)
    if grid.is_walkable(new_pos):
        return new_pos
    return player_pos


def run_interactive() -> None:
    """Run the game with human input from CLI using Live for in-place rendering.
    The same Game, tracker, and zone_model persist across rounds until the player quits.
    """
    console = Console()
    game = Game()
    ui = GameUI(console)

    # Show controls once at start
    console.print(Panel(
        "[bold]Controls:[/bold] W/A/S/D = move  |  H = toggle heatmap  |  Q = quit\n"
        "[bold]Goal:[/bold] Reach the key (K), then exit (E) without getting caught (X)",
        title="Adaptive Enemies",
        border_style="cyan"
    ))

    # Initial UI state
    player_pos = game.grid.start
    ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
    layout = ui.get_layout(game.grid, player_pos, game.enemies, False, game.tracker, game.zone_model)

    with Live(layout, console=console, refresh_per_second=10) as live:
        while True:
            # Use readchar for single keypress (no Enter needed)
            key = readchar.readkey().lower()

            if key == 'q':
                live.update(Panel("[yellow]Quitting...[/yellow]", border_style="yellow"))
                break
            elif key == 'h':
                ui.toggle_heatmap()
                # Re-render with updated heatmap
                ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
                layout = ui.get_layout(game.grid, player_pos, game.enemies, game.got_key, game.tracker, game.zone_model)
                live.update(layout)
                continue
            elif key not in ('w', 'a', 's', 'd'):
                live.update(Panel("[red]Invalid input. Use WASD to move, H for heatmap, Q to quit.[/red]", border_style="red"))
                continue

            # Compute intended move
            new_pos = get_move_from_key(key, player_pos, game.grid)
            if new_pos == player_pos:
                live.update(Panel("[red]Can't move there (wall or out of bounds).[/red]", border_style="red"))
                continue

            # Step the game (handles tracker, zone_model, enemies, capture, key, escape)
            caught, got_key, escaped = game.step(new_pos)
            player_pos = new_pos

            if caught:
                ui.set_last_result(f"CAUGHT at turn {game.turn_count}!")
                ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
                layout = ui.get_layout(game.grid, player_pos, game.enemies, got_key, game.tracker, game.zone_model)
                live.update(layout)
                live.update(Panel("[bold red]GAME OVER - CAUGHT![/bold red]", border_style="red"))
                # Start new round: reset enemies but keep tracker/zone_model
                game.reset_enemies()
                player_pos = game.grid.start
                ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
                layout = ui.get_layout(game.grid, player_pos, game.enemies, False, game.tracker, game.zone_model)
                live.update(layout)
                continue

            if escaped:
                ui.set_last_result(f"ESCAPED in {game.turn_count} turns!")
                ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
                layout = ui.get_layout(game.grid, player_pos, game.enemies, got_key, game.tracker, game.zone_model)
                live.update(layout)
                live.update(Panel("[bold green]VICTORY - ESCAPED![/bold green]", border_style="green"))
                # Start new round
                game.reset_enemies()
                player_pos = game.grid.start
                ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
                layout = ui.get_layout(game.grid, player_pos, game.enemies, False, game.tracker, game.zone_model)
                live.update(layout)
                continue

            # Normal continue: re-render
            ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
            layout = ui.get_layout(game.grid, player_pos, game.enemies, game.got_key, game.tracker, game.zone_model)
            live.update(layout)


if __name__ == "__main__":
    run_interactive()