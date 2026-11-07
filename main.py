"""Main entry point for the adaptive enemies game with human player (interactive)."""
from __future__ import annotations
import sys
from typing import List, Tuple, Optional
import random

import readchar
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.prompt import Prompt

from grid import Grid, Position
from game.engine import Game
from game.ui import GameUI
from game.enemy import Enemy
from game.levels import get_num_levels, get_level_config


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


def select_level(console: Console) -> Tuple[int, int]:
    """Display level selection menu and return selected (level_number, seed)."""
    num_levels = get_num_levels()

    table = Table(title="Select Level", show_header=True, header_style="bold cyan")
    table.add_column("Level", style="yellow", justify="center")
    table.add_column("Archetype", style="green", justify="center")
    table.add_column("Size", justify="center")
    table.add_column("Patrollers", justify="center")
    table.add_column("Flankers", justify="center")
    table.add_column("Alpha", justify="center")
    table.add_column("Sight", justify="center")
    table.add_column("Max Turns", justify="center")

    for level in range(1, num_levels + 1):
        config = get_level_config(level)
        from game.levels import get_archetype_for_level, get_map_size_for_level
        archetype = get_archetype_for_level(level)
        width, height = get_map_size_for_level(level)
        table.add_row(
            str(level),
            archetype,
            f"{width}x{height}",
            str(config.get("num_patrollers", 2)),
            str(config.get("num_flankers", 1)),
            str(config.get("alpha", 1.5)),
            str(config.get("sight_radius", 8)),
            str(config.get("max_turns", 500)),
        )

    console.print(table)
    console.print()

    while True:
        try:
            choice = Prompt.ask(
                f"[bold]Select level (1-{num_levels})[/bold]",
                default="1"
            )
            level = int(choice)
            if 1 <= level <= num_levels:
                break
            console.print(f"[red]Please enter a number between 1 and {num_levels}[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")

    # Seed selection
    console.print()
    while True:
        try:
            seed_input = Prompt.ask(
                "[bold]Enter seed (leave empty for random)[/bold]",
                default=""
            )
            if seed_input == "":
                seed = random.randint(0, 2**31 - 1)
                console.print(f"[dim]Using random seed: {seed}[/dim]")
                return level, seed
            seed = int(seed_input)
            if seed < 0:
                console.print("[red]Seed must be non-negative[/red]")
                continue
            return level, seed
        except ValueError:
            console.print("[red]Please enter a valid integer seed[/red]")


def run_interactive() -> None:
    """Run the game with human input from CLI using Live for in-place rendering.
    The same Game, tracker, and zone_model persist across rounds until the player quits.
    """
    console = Console()

    # Level and seed selection
    level, seed = select_level(console)
    game = Game(level=level, seed=seed)
    ui = GameUI(console)

    # Show controls once at start
    console.print(Panel(
        "[bold]Controls:[/bold] W/A/S/D = move  |  H = toggle heatmap  |  Q = quit\n"
        "[bold]Goal:[/bold] Reach the vault (V), then exit (E) without getting caught (X)\n"
        f"[bold]Level:[/bold] {level}  |  [bold]Seed:[/bold] {seed}  |  [bold]Archetype:[/bold] {game.level_obj.archetype}",
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