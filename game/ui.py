"""Game UI using plain text for terminal rendering."""
from __future__ import annotations
import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from grid import Grid, Position
from game.enemy import Enemy, EnemyState, Patroller, Flanker
from learning import Tracker, ZoneModel
from game.engine import Game


# ANSI colour codes
ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_BLUE = "\033[94m"
ANSI_MAGENTA = "\033[95m"
ANSI_WHITE = "\033[97m"
ANSI_DIM = "\033[2m"

# Heatmap gradient using single-width ASCII chars
HEATMAP_CHARS = " .:-=+*#%"

# Enemy state colours
ENEMY_STATE_COLOURS = {
    EnemyState.PATROL: ANSI_GREEN,
    EnemyState.CHASE: ANSI_RED,
    EnemyState.FLANK: ANSI_YELLOW,
}


@dataclass
class UIState:
    """State for the UI."""
    show_heatmap: bool = False
    round_number: int = 1
    last_result: str = ""
    markov_prediction: str = ""


def visible_width(s: str) -> int:
    """Return the visible width of a string, stripping ANSI escape codes."""
    # Strip ANSI escape sequences (CSI sequences like \x1b[31m, \x1b[0m, etc.)
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    clean = ansi_escape.sub("", s)
    return len(clean)


def _colour_wrap(symbol: str, colour: str) -> str:
    """Wrap a single symbol in ANSI colour codes."""
    return f"{colour}{symbol}{ANSI_RESET}"


def _get_enemy_symbol(enemy: Enemy, index: int) -> str:
    """Get single-character symbol for enemy (digit 1-5)."""
    # Enemy index 0->'1', 1->'2', etc. Supports up to 5 enemies.
    return str(index + 1)


def _get_enemy_colour(enemy: Enemy) -> str:
    """Get ANSI colour for enemy based on its state."""
    return ENEMY_STATE_COLOURS.get(enemy.state, ANSI_WHITE)


def render_rows(
    game: Game,
    heatmap_on: bool,
    player_pos: Position,
    got_key: bool,
) -> List[str]:
    """
    Render the game grid as a list of strings.
    Each tile is exactly 2 columns: one symbol char + one space.
    Returns list of rows (without side panel).
    """
    grid = game.grid
    enemies = game.enemies
    tracker = game.tracker

    # Build enemy position map: pos -> (enemy_index, enemy)
    enemy_at: Dict[Position, Tuple[int, Enemy]] = {}
    for i, enemy in enumerate(enemies):
        enemy_at[enemy.position] = (i, enemy)

    prob_map = tracker.probability_map() if heatmap_on else {}

    rows = []
    for y in range(grid.height):
        row_parts = []
        for x in range(grid.width):
            pos = (x, y)

            # Determine symbol and colour for this tile
            symbol = '.'
            colour = ANSI_RESET

            if pos in grid.walls:
                symbol = '#'
                colour = ANSI_DIM
            elif pos == player_pos:
                symbol = '@'
                colour = ANSI_CYAN
            elif pos in enemy_at:
                # Enemy takes priority over other floor features
                enemy_idx, enemy = enemy_at[pos]
                symbol = _get_enemy_symbol(enemy, enemy_idx)
                colour = _get_enemy_colour(enemy)
            elif pos == grid.vault and not got_key:
                symbol = 'V'
                colour = ANSI_YELLOW
            elif pos == grid.exit:
                symbol = 'E'
                colour = ANSI_MAGENTA
            elif pos == grid.start:
                symbol = 'S'
                colour = ANSI_BLUE
            elif heatmap_on and pos in prob_map:
                prob = prob_map[pos]
                idx = int(prob * (len(HEATMAP_CHARS) - 1) + 0.5)
                idx = max(0, min(idx, len(HEATMAP_CHARS) - 1))
                symbol = HEATMAP_CHARS[idx]
                colour = ANSI_DIM
            else:
                symbol = '.'
                colour = ANSI_RESET

            # Each tile: coloured symbol + space (exactly 2 visible columns)
            row_parts.append(_colour_wrap(symbol, colour) + " ")

        rows.append("".join(row_parts))

    return rows


def render_status(
    grid: Grid,
    player_pos: Position,
    enemies: List[Enemy],
    got_key: bool,
    tracker: Tracker,
    zone_model: ZoneModel,
    ui_state: UIState,
) -> List[str]:
    """Render status panel as list of strings (fixed width)."""
    lines = []

    # Round number
    lines.append(f"Round: {ui_state.round_number}")

    # Player position and key status
    zone = grid.zone_of(player_pos)
    key_status = "Yes" if got_key else "No"
    lines.append(f"Position: {player_pos}  Zone: {zone}  Key: {key_status}")

    # Enemy states
    lines.append("Enemies:")
    for i, enemy in enumerate(enemies):
        symbol = _get_enemy_symbol(enemy, i)
        state_str = enemy.state.value.capitalize()
        colour = _get_enemy_colour(enemy)
        coloured_symbol = _colour_wrap(symbol, colour)
        lines.append(f"  {coloured_symbol}: {state_str} at {enemy.position}")

    # Last result
    if ui_state.last_result:
        lines.append(f"Last Round: {ui_state.last_result}")

    # Markov prediction
    if ui_state.markov_prediction:
        lines.append(f"Markov Prediction: {ui_state.markov_prediction}")

    # Heatmap toggle hint
    heatmap_status = "ON" if ui_state.show_heatmap else "OFF"
    lines.append(f"Heatmap: {heatmap_status} (press 'h' to toggle)")

    return lines


def render_controls() -> List[str]:
    """Render controls help as list of strings."""
    return [
        "Controls:",
        "  w/a/s/d - Move up/left/down/right",
        "  h       - Toggle heatmap overlay",
        "  q       - Quit game",
    ]


class GameUI:
    """Plain-text UI for the adaptive enemies game."""

    def __init__(self):
        self.ui_state = UIState()
        self._first_frame = True

    def get_layout(
        self,
        game: Game,
        player_pos: Position,
        got_key: bool,
    ) -> str:
        """Create the full UI as a single string with map and side panel."""
        grid = game.grid
        heatmap_on = self.ui_state.show_heatmap

        # Render map rows (each tile = 2 visible columns)
        map_rows = render_rows(game, heatmap_on, player_pos, got_key)
        map_visible_width = grid.width * 2  # 2 cols per tile

        # Render status panel lines
        status_lines = render_status(
            grid, player_pos, game.enemies, got_key,
            game.tracker, game.zone_model, self.ui_state
        )
        control_lines = render_controls()

        # Side panel configuration
        PANEL_WIDTH = 45
        GAP = "  "  # Two spaces between map and panel

        # Combine map and panel side by side
        # If terminal is too narrow, stack panel below map
        # We'll assume a reasonable minimum terminal width; if not, the user can resize
        output_lines = []

        max_map_lines = len(map_rows)
        max_panel_lines = max(len(status_lines), len(control_lines))

        # First, map + status side by side
        for i in range(max(max_map_lines, len(status_lines))):
            map_part = map_rows[i] if i < max_map_lines else " " * map_visible_width
            # Ensure map_part has exactly map_visible_width visible columns
            # (it should already, but pad if needed)
            mvw = visible_width(map_part)
            if mvw < map_visible_width:
                map_part += " " * (map_visible_width - mvw)

            status_part = status_lines[i] if i < len(status_lines) else ""
            # Truncate or pad status to PANEL_WIDTH
            svw = visible_width(status_part)
            if svw > PANEL_WIDTH:
                # Truncate - find the cut point accounting for ANSI
                status_part = self._truncate_to_visible(status_part, PANEL_WIDTH)
            elif svw < PANEL_WIDTH:
                status_part += " " * (PANEL_WIDTH - svw)

            output_lines.append(map_part + GAP + status_part)

        # Then controls below (full width or just panel area)
        if max_map_lines > len(status_lines):
            # Map is taller than status, add controls below status area
            for i in range(len(control_lines)):
                if len(status_lines) + i < max_map_lines:
                    # There's still map rows, append to those
                    idx = len(status_lines) + i
                    map_part = map_rows[idx] if idx < max_map_lines else " " * map_visible_width
                    mvw = visible_width(map_part)
                    if mvw < map_visible_width:
                        map_part += " " * (map_visible_width - mvw)
                    ctrl_part = control_lines[i]
                    cvw = visible_width(ctrl_part)
                    if cvw > PANEL_WIDTH:
                        ctrl_part = self._truncate_to_visible(ctrl_part, PANEL_WIDTH)
                    elif cvw < PANEL_WIDTH:
                        ctrl_part += " " * (PANEL_WIDTH - cvw)
                    output_lines[idx] = map_part + GAP + ctrl_part
                else:
                    # No more map rows, just print controls indented
                    output_lines.append(" " * map_visible_width + GAP + control_lines[i])
        else:
            # Status is taller or equal, append controls after
            for line in control_lines:
                output_lines.append(" " * map_visible_width + GAP + line)

        return "\n".join(output_lines)

    def _truncate_to_visible(self, s: str, max_visible: int) -> str:
        """Truncate string to max_visible visible characters, preserving ANSI codes."""
        ansi_escape = re.compile(r"(\x1b\[[0-9;]*m)")
        parts = ansi_escape.split(s)
        # parts alternates: text, ansi, text, ansi, ...
        visible_count = 0
        result_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 0:  # text part
                if visible_count + len(part) <= max_visible:
                    result_parts.append(part)
                    visible_count += len(part)
                else:
                    # Need to truncate this part
                    remaining = max_visible - visible_count
                    if remaining > 0:
                        result_parts.append(part[:remaining])
                    break
            else:  # ANSI code
                result_parts.append(part)
        return "".join(result_parts)

    def update_markov_prediction(self, zone_model: ZoneModel, current_zone: str) -> None:
        """Update the Markov prediction display."""
        if zone_model.last_zone is not None:
            predicted = zone_model.predict_next(current_zone)
            self.ui_state.markov_prediction = f"{current_zone} -> {predicted}"
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

    def hide_cursor(self) -> None:
        """Hide terminal cursor."""
        print("\033[?25l", end="", flush=True)

    def show_cursor(self) -> None:
        """Show terminal cursor."""
        print("\033[?25h", end="", flush=True)

    def move_cursor_home(self) -> None:
        """Move cursor to top-left (1,1)."""
        print("\033[H", end="", flush=True)

    def clear_screen_once(self) -> None:
        """Clear screen and move cursor home (for initial setup)."""
        print("\033[2J\033[H", end="", flush=True)

    def erase_below(self) -> None:
        """Erase from cursor to end of screen."""
        print("\033[J", end="", flush=True)

    def draw_frame(self, frame: str) -> None:
        """Draw a complete frame at the top-left without clearing."""
        self.move_cursor_home()
        print(frame, end="", flush=True)
        self.erase_below()


def build_level_table() -> List[str]:
    """Build the level selection table with properly aligned columns."""
    from game.levels import get_num_levels, get_level_config, get_archetype_for_level, get_map_size_for_level

    num_levels = get_num_levels()

    # Column definitions: (header, key_func, align, format_func)
    columns = [
        ("Level", lambda l, c: str(l), ">"),
        ("Archetype", lambda l, c: get_archetype_for_level(l), "<"),
        ("Size", lambda l, c: f"{get_map_size_for_level(l)[0]}x{get_map_size_for_level(l)[1]}", ">"),
        ("Patrollers", lambda l, c: str(c.get("num_patrollers", 2)), ">"),
        ("Flankers", lambda l, c: str(c.get("num_flankers", 1)), ">"),
        ("Alpha", lambda l, c: f"{c.get('alpha', 1.5):.2f}", ">"),
        ("Sight", lambda l, c: str(c.get("sight_radius", 8)), ">"),
        ("Max Turns", lambda l, c: str(c.get("max_turns", 500)), ">"),
    ]

    # Compute column widths
    col_widths = []
    for header, key_func, _ in columns:
        max_w = len(header)
        for level in range(1, num_levels + 1):
            config = get_level_config(level)
            val = key_func(level, config)
            max_w = max(max_w, len(val))
        col_widths.append(max_w)

    # Build format string
    # Two spaces between columns
    fmt_parts = []
    for i, (_, _, align) in enumerate(columns):
        if align == ">":
            fmt_parts.append(f"{{:>{col_widths[i]}}}")
        else:
            fmt_parts.append(f"{{:<{col_widths[i]}}}")
    row_fmt = "  ".join(fmt_parts)

    lines = []
    # Header
    headers = [col[0] for col in columns]
    lines.append(row_fmt.format(*headers))
    # Separator
    sep_parts = ["-" * w for w in col_widths]
    lines.append("  ".join(sep_parts))
    # Data rows
    for level in range(1, num_levels + 1):
        config = get_level_config(level)
        values = [key_func(level, config) for _, key_func, _ in columns]
        lines.append(row_fmt.format(*values))

    return lines