# Report Notes: Adaptive Enemies Batch Evaluation

## Experiment Configuration

- **Map**: 30×15 grid, three parallel vertical corridors (west/middle/east) joined only at Start (row 1), Vault (row 7), Exit (row 13)
- **Enemies per round**: 2 Patrollers (west & middle routes), 1 Flanker (east route)
- **Learning parameters**: Tracker decay 10% every 5 turns, ZoneModel Laplace smoothing (α=1), Cost map α=3.0
- **Rounds per bot**: 30
- **Max turns per round**: 500

## Bots Tested

| Bot | Description |
|-----|-------------|
| **FixedRouteBot** | Always takes the east corridor (shortest path S→K→E) |
| **SwitchingBot** | Alternates between east and middle corridors every 3 rounds |
| **RandomBot** | Chooses a random valid move each turn (no policy) |

## Results Summary

### FixedRouteBot (30 rounds)
- **Captured**: 30/30 (100%)
- **Escaped**: 0/30 (0%)
- **Average turns to capture**: 9.0
- **Early rounds (1-10) capture rate**: 10/10 (100%)
- **Late rounds (21-30) capture rate**: 10/10 (100%)
- **Capture rate change**: No rise (100% → 100%)

### SwitchingBot (30 rounds)
- **Captured**: 30/30 (100%)
- **Escaped**: 0/30 (0%)
- **Average turns to capture**: 18.9

### RandomBot (30 rounds)
- **Captured**: 30/30 (100%)
- **Escaped**: 0/30 (0%)
- **Average turns to capture**: 11.4

## Analysis

### Learning Effectiveness
The FixedRouteBot results show **no improvement in capture rate from early to late rounds** (10/10 → 10/10). This indicates that on the current map geometry, the learning system (Tracker + ZoneModel + cost maps) does not produce a measurable advantage against a deterministic corridor-hugging policy.

**Possible reasons:**
1. **Geometry dominates**: The east corridor is a straight line from Start to Vault to Exit. Patrollers in the middle corridor have line-of-sight onto the east corridor junction rows, capturing the player before zone-model predictions matter.
2. **Capture is too fast**: Average 9 turns leaves little time for the tracker to accumulate meaningful statistics before the round ends.
3. **Zone transitions are deterministic**: FixedRouteBot never transitions zones after the initial commitment (stays in east), so ZoneModel has no transitions to learn from.
4. **Cost map saturation**: With α=3.0, costs clamp at -0.9 quickly, reducing differentiation between hot and cold tiles.

### SwitchingBot vs FixedRouteBot
SwitchingBot survives ~2× longer (18.9 vs 9.0 turns) because it forces patrollers to relearn when the corridor switches. However, capture rate remains 100% — the flanker in the east corridor eventually intercepts.

### RandomBot
RandomBot survives slightly longer than FixedRouteBot (11.4 vs 9.0 turns) due to occasional backtracking, but still gets caught 100% of the time.

## Visualizations Generated

`run_batch.py` produces:
- `markov_accuracy.png` — ZoneModel next-zone prediction accuracy over rounds (predict-before-observe protocol)
- `capture_rate_by_round.png` — Per-bot capture rate over 30 rounds
- `avg_turns_by_round.png` — Per-bot average turns over 30 rounds
- `heatmap_final.png` — Final tracker probability map overlaid on grid
- `batch_results.csv` — Raw per-round data

## Conclusions

1. **Core mechanics work**: Patrol routes connect via A*, line-of-sight blocks correctly, spawns respect distance/zones, capture detection is symmetric, learning persists across rounds.
2. **Learning signal is weak on this map**: The corridor geometry and enemy placement (middle patroller sees east junction) cause early capture before adaptation can demonstrate value.
3. **Recommended map changes for stronger learning signal**:
   - Lengthen corridors (more tiles between junctions)
   - Add occluding walls between corridors at junction rows
   - Reduce patroller sight radius or reposition spawns
   - Increase max turns per round

## Test Coverage

34 unit tests pass, covering:
- Grid pathfinding (BFS distance)
- Patrol route construction (connected, excludes S/V/E, valid waypoints)
- Enemy spawning (min distance ≥8, per-zone, avoids junction rows)
- Line-of-sight (clear, wall-blocked, range-limited, symmetric)
- Cost map normalization (clamping, monotonicity with α, empty tracker)
- Game.step (movement, 3 capture types, key pickup, escape)
- Patroller behavior (1-tile steps, tracker-weighted waypoint selection)
- Flanker behavior (hottest-zone flank target, sight-gated chase)
- play_round (uses step internally)
- Patrol adjacency (A*-connected waypoints yield adjacent steps)
- Symmetric capture (player→enemy, enemy→player, swap, no-capture adjacent)