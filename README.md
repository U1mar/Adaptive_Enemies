# Adaptive Enemies

A roguelike-style game where enemies learn from player behavior across rounds using Markov chains and probability-based pathfinding.

## Overview

This project implements an adaptive enemy AI system for a grid-based game. Enemies track player movement patterns, build probabilistic models of player behavior, and adjust their patrol routes and positioning to intercept the player more effectively over time.

## Features

- **Three enemy types**: Patrollers (follow connected waypoint loops), Flankers (ambush from hottest zones), and learning persistence across rounds
- **Real line-of-sight**: Bresenham algorithm with wall occlusion
- **Markov zone modeling**: Tracks player transitions between west/middle/east corridors with Laplace smoothing
- **Dynamic cost maps**: A* pathfinding weighted by observed player frequency (α≈3 normalization)
- **Persistent learning**: Tracker and ZoneModel survive round resets; only enemy positions reset
- **Symmetric capture**: Detects player-on-enemy, enemy-on-player, and tile-swap collisions

## Game Map

The default map is a 30×15 grid with three parallel vertical corridors connected only at:
- **Start (S)** — top junction row
- **Vault (K)** — middle junction row (key pickup)
- **Exit (E)** — bottom junction row (escape with key)

Corridors are separated by walls, forcing the player to commit to a lane.

## Enemy Behaviors

| Enemy | Behavior |
|-------|----------|
| **Patroller (×2)** | Patrols a connected loop of waypoints in west/middle corridors. When not chasing, samples next waypoint from tracker probability map (biased toward historically hot tiles). Switches to CHASE on line-of-sight. |
| **Flanker (×1)** | Waits on a tile in the hottest zone that lies on the shortest path toward the exit. Only enters CHASE when it gains line-of-sight on the player. |

## Learning System

1. **Tracker**: Records every player position. Decays counts by 10% every 5 turns. Produces a probability map via Laplace-smoothed normalization.
2. **ZoneModel**: First-order Markov chain over {west, middle, east} zones. Records zone transitions. Predicts next zone with `P(next | current)`.
3. **Cost Map**: `cost = clamp(-α × normalized_prob, min_step_cost - 1.0)` where `α = 3.0`, `min_step_cost = 0.1`. Higher player frequency → more negative cost → A* prefers those tiles.

## Batch Results (30 rounds each bot)

| Bot | Captured | Escaped | Avg Turns | Early Capture (1-10) | Late Capture (21-30) |
|-----|----------|---------|-----------|----------------------|----------------------|
| FixedRouteBot | 30/30 | 0/30 | 9.0 | 10/10 | 10/10 |
| SwitchingBot | 30/30 | 0/30 | 18.9 | — | — |
| RandomBot | 30/30 | 0/30 | 11.4 | — | — |

**Key observation**: FixedRouteBot capture rate does not rise from early to late rounds (10/10 → 10/10), indicating the learning system does not yet produce a measurable improvement against deterministic corridor-hugging policies on this map geometry.

## Installation

```bash
pip install -r adaptive_enemies/requirements.txt
```

Dependencies: `numpy`, `matplotlib`, `pytest`, `networkx`, `rich`, `readchar`

## Running

### Interactive (human player)
```bash
python main.py
```
Controls: `W/A/S/D` = move, `H` = toggle heatmap, `Q` = quit

### Batch evaluation
```bash
python run_batch.py
```
Runs 30 rounds each for FixedRouteBot, SwitchingBot, RandomBot. Outputs CSV logs and plots.

### Tests
```bash
python -m pytest test_game.py -v
```
34 tests covering grid, patrol routes, spawning, line-of-sight, cost maps, Game.step, patroller/flanker behavior, play_round, patrol adjacency, and symmetric capture.

## Project Structure

```
adaptive_enemies/
├── main.py              # Interactive game loop (readchar + Rich Live)
├── run_batch.py         # Batch evaluation harness
├── test_game.py         # Test suite (34 tests)
├── grid.py              # Grid, zones, path_distance (BFS)
├── pathing.py           # A* with cost map support
├── learning.py          # Tracker, ZoneModel (Markov)
├── ai/
│   └── costs.py         # build_cost_map, build_flank_targets
├── game/
│   ├── engine.py        # Game, build_patrol_routes, find_spawn_position, Game.step, reset_enemies
│   ├── enemy.py         # Patroller, Flanker, EnemyState, Bresenham LOS
│   └── ui.py            # Rich-based rendering with heatmap overlay
├── bots.py              # FixedRouteBot, SwitchingBot, RandomBot policies
└── requirements.txt
```

## Design Notes

- **Patrol routes** are built as connected loops via A* between waypoints, excluding junction rows (S/V/E rows) so patrollers only traverse vertical corridor segments.
- **Spawn positions** are chosen per-zone with minimum path distance ≥ 8 from start, avoiding junction rows.
- **Line-of-sight** uses Bresenham's algorithm; walls block vision; range limited to `sight_radius=8`.
- **Capture detection** is symmetric: player moves onto enemy, enemy moves onto player, or they swap tiles in the same turn.
- **Learning persistence**: `Game.reset_enemies()` clears enemies but preserves `tracker` and `zone_model` across rounds.