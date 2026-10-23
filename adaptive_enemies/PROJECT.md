# Adaptive Enemies - Project Specification

## Overview
Turn-based grid dungeon game where a player steals a key at vault **V** and reaches exit **E**. Enemies learn the player's habits through a tracker (tile visits with decay) and a Markov chain over zones (west/middle/east) that predicts the next zone. Probabilities modify tile costs so enemy A* paths cover likely routes.

## Shared Interfaces

### Types
```python
Pos = tuple[int, int]  # (x, y)
```

### game/grid.py
```python
class Grid:
    def __init__(self, width: int, height: int, walls: set[Pos])
    def in_bounds(self, p: Pos) -> bool
    def passable(self, p: Pos) -> bool
    def neighbors(self, p: Pos) -> list[Pos]  # 4-directional
    @classmethod
    def from_ascii(cls, text: str) -> Grid  # '#' wall, '.' floor, 'S' start, 'V' vault, 'E' exit
    # attributes: start, vault, exit: Pos
    def zone_of(self, p: Pos) -> str  # "west" | "middle" | "east" by x-third
```

### pathing/
```python
@dataclass
class PathResult:
    path: list[Pos] | None
    explored: int
    cost: float

def bfs(grid: Grid, start: Pos, goal: Pos) -> PathResult
def dijkstra(grid: Grid, start: Pos, goal: Pos, cost_map: dict[Pos, float] | None = None) -> PathResult
def astar(grid: Grid, start: Pos, goal: Pos, cost_map: dict[Pos, float] | None = None) -> PathResult
# cost_map: extra cost per tile (default 0); base step cost = 1
```

### ai/tracker.py
```python
class Tracker:
    def __init__(self, decay: float = 0.95, history_len: int = 50)
    def record(self, pos: Pos) -> None
    def probability_map(self) -> dict[Pos, float]  # sums to 1
```

### ai/markov.py
```python
class ZoneModel:
    def observe(self, prev_zone: str, next_zone: str) -> None
    def predict_next(self, zone: str) -> dict[str, float]  # zone -> probability
```

### ai/costs.py
```python
def build_cost_map(grid: Grid, prob_map: dict[Pos, float], alpha: float = 5.0) -> dict[Pos, float]
```

### game/enemy.py
```python
class Enemy:
    def __init__(self, pos: Pos, role: str, patrol_route: list[Pos])
    def step(self, grid: Grid, player_pos: Pos, cost_map: dict[Pos, float]) -> Pos
    # roles: "patroller" | "flanker"
    # states: patrol / chase / ambush
    def sees(self, grid: Grid, player_pos: Pos, radius: int = 5) -> bool
```

### game/engine.py
```python
@dataclass
class RoundResult:
    caught: bool
    got_key: bool
    escaped: bool
    turns: int
    path: list[Pos]

class Game:
    def __init__(self, grid: Grid, enemies: list[Enemy], tracker: Tracker, model: ZoneModel)
    def play_round(self, player_policy, max_turns: int = 200) -> RoundResult
```

### sim/
```python
# bots.py: bots implement policy(game_state) -> Pos
# run_batch.py: run multiple rounds, save CSV
# plots.py: generate plots from CSV
```

### game/levels.py
Default 40x20 ASCII map with three vertical corridors (west, middle, east) connecting S, V, E.