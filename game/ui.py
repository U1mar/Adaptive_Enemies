"""Game UI using rich library for terminal rendering."""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich import box

from grid import Grid, Position
from game.enemy import Enemy, EnemyState, Patroller, Flanker
from learning import Tracker, ZoneModel


@dataclass
class UIState:
    """State for the UI."""
    show_heatmap: bool = False
    round_number: int = 1
    last_result: str = ""
    markov_prediction: str = ""


class GameUI:
    """Rich-based UI for the adaptive enemies game."""

    # Enemy state colors
    STATE_COLORS = {
        EnemyState.PATROL: "green",
        EnemyState.CHASE: "red",
        EnemyState.FLANK: "yellow",
    }

    # Enemy labels
    ENEMY_LABELS = ["E1", "E2", "E3"]

    def __init__(self, console: Console = None):
        self.console = console or Console()
        self.ui_state = UIState()

    def render_grid(
        self,
        grid: Grid,
        player_pos: Position,
        enemies: List[Enemy],
        got_key: bool,
        tracker: Tracker,
        zone_model: ZoneModel,
    ) -> Table:
        """Render the game grid as a rich Table."""
        table = Table(show_header=False, show_lines=False, box=box.SIMPLE, padding=0)

        # Add columns
        for _ in range(grid.width):
            table.add_column(width=2, justify="center")

        # Enemy positions and states for quick lookup
        enemy_positions: Dict[Position, Tuple[str, EnemyState]] = {}
        for i, enemy in enumerate(enemies):
            label = self.ENEMY_LABELS[i] if i < len(self.ENEMY_LABELS) else f"E{i+1}"
            enemy_positions[enemy.position] = (label, enemy.state)

        prob_map = tracker.probability_map() if self.ui_state.show_heatmap else {}

        for y in range(grid.height):
            row = []
            for x in range(grid.width):
                pos = (x, y)

                if pos in grid.walls:
                    cell = Text("#", style="bold white")
                elif pos == player_pos:
                    cell = Text("@", style="bold cyan")
                elif pos in enemy_positions:
                    label, state = enemy_positions[pos]
                    color = self.STATE_COLORS.get(state, "white")
                    cell = Text(label, style=f"bold {color}")
                elif pos == grid.vault and not got_key:
                    cell = Text("V", style="bold magenta")
                elif pos == grid.exit:
                    cell = Text("E", style="bold green")
                elif pos == grid.start:
                    cell = Text("S", style="dim white")
                elif self.ui_state.show_heatmap and pos in prob_map:
                    # Heatmap: shade background based on probability
                    prob = prob_map[pos]
                    if prob > 0.7:
                        bg = "red"
                    elif prob > 0.4:
                        bg = "yellow"
                    elif prob > 0.1:
                        bg = "blue"
                    else:
                        bg = "black"
                    cell = Text("·", style=f"white on {bg}")
                else:
                    cell = Text("·", style="dim white")
                row.append(cell)
            table.add_row(*row)

        return table

    def render_status(
        self,
        grid: Grid,
        player_pos: Position,
        enemies: List[Enemy],
        got_key: bool,
        tracker: Tracker,
        zone_model: ZoneModel,
    ) -> Panel:
        """Render status panel with round info, last result, and Markov prediction."""
        lines = []

        # Round number
        lines.append(f"[bold]Round:[/bold] {self.ui_state.round_number}")

        # Player position and key status
        zone = grid.zone_of(player_pos)
        key_status = "Yes" if got_key else "No"
        lines.append(f"[bold]Position:[/bold] {player_pos}  [bold]Zone:[/bold] {zone}  [bold]Key:[/bold] {key_status}")

        # Enemy states
        lines.append("[bold]Enemies:[/bold]")
        for i, enemy in enumerate(enemies):
            label = self.ENEMY_LABELS[i] if i < len(self.ENEMY_LABELS) else f"E{i+1}"
            state_str = enemy.state.value.capitalize()
            color = self.STATE_COLORS.get(enemy.state, "white")
            lines.append(f"  [{color}]{label}[/{color}]: {state_str} at {enemy.position}")

        # Last result
        if self.ui_state.last_result:
            lines.append(f"[bold]Last Round:[/bold] {self.ui_state.last_result}")

        # Markov prediction
        if self.ui_state.markov_prediction:
            lines.append(f"[bold]Markov Prediction:[/bold] {self.ui_state.markov_prediction}")

        # Heatmap toggle hint
        heatmap_status = "ON" if self.ui_state.show_heatmap else "OFF"
        lines.append(f"[dim]Heatmap: {heatmap_status} (press 'h' to toggle)[/dim]")

        content = "\n".join(lines)
        return Panel(content, title="Game Status", border_style="blue", padding=(0, 1))

    def render_controls(self) -> Panel:
        """Render controls help panel."""
        controls = [
            "[bold]Controls:[/bold]",
            "  [bold cyan]w/a/s/d[/bold cyan] - Move up/left/down/right",
            "  [bold yellow]h[/bold yellow] - Toggle heatmap overlay",
            "  [bold red]q[/bold red] - Quit game",
        ]
        content = "\n".join(controls)
        return Panel(content, title="Controls", border_style="dim", padding=(0, 1))

    def get_layout(
        self,
        grid: Grid,
        player_pos: Position,
        enemies: List[Enemy],
        got_key: bool,
        tracker: Tracker,
        zone_model: ZoneModel,
    ) -> Layout:
        """Create the full layout with grid and status panels."""
        layout = Layout()

        layout.split_row(
            Layout(self.render_grid(grid, player_pos, enemies, got_key, tracker, zone_model), name="grid", ratio=3),
            Layout(name="side", ratio=1)
        )

        layout["side"].split_column(
            Layout(self.render_status(grid, player_pos, enemies, got_key, tracker, zone_model), name="status", ratio=2),
            Layout(self.render_controls(), name="controls", ratio=1)
        )

        return layout

    def update_markov_prediction(self, zone_model: ZoneModel, current_zone: str) -> None:
        """Update the Markov prediction display."""
        if zone_model.last_zone is not None:
            predicted = zone_model.predict_next(current_zone)
            self.ui_state.markov_prediction = f"{current_zone} → {predicted}"
        else:
            self.ui_state.markov_prediction = "Insufficient data"

    def set_last_result(self, result: str) -> None:
        """Set the last round result message."""
        self.ui_state.last_result = result

    def next_round(self) -> None:
        """Increment round number."""
        self.ui_state.round_number += 1

    def toggle_heatmap(self) -> None:
        """Toggle heatmap display."""
        self.ui_state.show_heatmap = not self.ui_state.show_heatmap