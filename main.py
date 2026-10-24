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
        'd': (x + 1, y),
    }
    new_pos = moves.get(key.lower(), player_pos)
    if grid.is_walkable(new_pos):
        return new_pos
    return player_pos


def print_grid(grid: Grid, player_pos: Position, enemies: List[Enemy], got_key: bool) -> None:
    """Print a simple text representation of the grid."""
    console = Console()
    rows = []
    for y in range(grid.height):
        row = []
        for x in range(grid.width):
            pos = (x, y)
            if pos == player_pos:
                row.append("@")
            elif pos == grid.vault and not got_key:
                row.append("K")
            elif pos == grid.exit:
                row.append("E")
            elif any(e.position == pos for e in enemies):
                row.append("X")
            elif grid.is_wall(pos):
                row.append("#")
            else:
                row.append(".")
        rows.append("".join(row))
    console.print("\n".join(rows))
    console.print(f"\n[bold]Position:[/bold] {player_pos}  [bold]Key:[/bold] {'Yes' if got_key else 'No'}  [bold]Enemies:[/bold] {len(enemies)}")


def run_interactive() -> None:
    """Run the game with human input from CLI using Live for in-place rendering."""
    console = Console()
    game = Game()
    ui = GameUI(console)

    player_pos = game.grid.start
    got_key = False
    turn_count = 0

    # Initial UI state
    ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
    layout = ui.get_layout(game.grid, player_pos, game.enemies, got_key, game.tracker, game.zone_model)

    # Show controls once at start
    console.print(Panel(
        "[bold]Controls:[/bold] W/A/S/D = move  |  H = toggle heatmap  |  Q = quit\n"
        "[bold]Goal:[/bold] Reach the key (K), then exit (E) without getting caught (X)",
        title="Adaptive Enemies",
        border_style="cyan"
    ))

    # Use Live for in-place rendering - prevents multiple boards stacking
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
                layout = ui.get_layout(game.grid, player_pos, game.enemies, got_key, game.tracker, game.zone_model)
                live.update(layout)
                continue
            elif key not in ('w', 'a', 's', 'd'):
                live.update(Panel("[red]Invalid input. Use WASD to move, H for heatmap, Q to quit.[/red]", border_style="red"))
                continue

            # Move player
            new_pos = get_move_from_key(key, player_pos, game.grid)
            if new_pos == player_pos:
                live.update(Panel("[red]Can't move there (wall or out of bounds).[/red]", border_style="red"))
                continue

            # Record position with tracker
            game.tracker.record(player_pos)

            # Update zone model
            current_zone = game.grid.zone_of(player_pos)
            if hasattr(run_interactive, 'last_zone'):
                if current_zone != run_interactive.last_zone:
                    game.zone_model.record_transition(run_interactive.last_zone, current_zone)
            run_interactive.last_zone = current_zone

            # Decay tracker every 5 turns
            turn_count += 1
            if turn_count % 5 == 0:
                game.tracker.decay_counts()

            player_pos = new_pos

            # Check for key pickup
            if player_pos == game.grid.vault:
                got_key = True
                live.update(Panel("[green]Got the key![/green]", border_style="green"))
                # Brief pause to show message, then continue
                continue

            # Check for capture
            enemy_positions = [e.position for e in game.enemies]
            if player_pos in enemy_positions:
                ui.set_last_result(f"CAUGHT at turn {turn_count}!")
                ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
                layout = ui.get_layout(game.grid, player_pos, game.enemies, got_key, game.tracker, game.zone_model)
                live.update(layout)
                live.update(Panel("[bold red]GAME OVER - CAUGHT![/bold red]", border_style="red"))
                break

            # Check for escape
            if player_pos == game.grid.exit and got_key:
                ui.set_last_result(f"ESCAPED in {turn_count} turns!")
                ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
                layout = ui.get_layout(game.grid, player_pos, game.enemies, got_key, game.tracker, game.zone_model)
                live.update(layout)
                live.update(Panel("[bold green]VICTORY - ESCAPED![/bold green]", border_style="green"))
                break

            # Update enemies
            for enemy in game.enemies:
                enemy.update(game.grid, player_pos, game.tracker, game.zone_model, game.alpha)

            # Check capture after enemy move
            enemy_positions = [e.position for e in game.enemies]
            if player_pos in enemy_positions:
                ui.set_last_result(f"CAUGHT at turn {turn_count} (enemy moved)!")
                ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
                layout = ui.get_layout(game.grid, player_pos, game.enemies, got_key, game.tracker, game.zone_model)
                live.update(layout)
                live.update(Panel("[bold red]GAME OVER - CAUGHT![/bold red]", border_style="red"))
                break

            # Re-render (update in place)
            ui.update_markov_prediction(game.zone_model, game.grid.zone_of(player_pos))
            layout = ui.get_layout(game.grid, player_pos, game.enemies, got_key, game.tracker, game.zone_model)
            live.update(layout)


if __name__ == "__main__":
    run_interactive()