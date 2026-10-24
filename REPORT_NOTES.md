# Report Notes (copy‑paste sections)

## Data structures & why
- **Grid (width, height, walls set, start/vault/exit)** – O(1) wall checks, 4‑dir neighbors, zone lookup by x‑third.
- **Tracker** – `dict[Position, float]` visit counts with exponential decay (`decay=0.95`) and fixed‑length deque history (50). Gives a probability map `p(tile)` that emphasises recent player activity.
- **ZoneModel** – 3×3 transition matrix with Laplace smoothing (`alpha=1.0`). Stores `transitions[from][to]` counts; `predict_next(zone)` returns most likely next zone.
- **Enemy state machine** – `Patroller` (loop patrol, chase on sight) and `Flanker` (ambush at exit‑side of hottest zone, chase on sight). Shared `replan_interval=3` limits A* calls.

## How the model feeds A*
1. Each turn the Tracker records the player tile and decays every 5 turns.
2. `build_cost_map(tracker, α)` converts the probability map to a cost map: `cost = -α * p`, clamped so step cost ≥ 0.1.
3. Enemy `update()` calls `astar(grid, pos, target, cost_map)` only when `replan_counter ≥ 3` or on state change.
4. Flanker also uses `build_flank_targets(grid, prob_map)` to pick a wait tile on the eastern boundary of the hottest zone.

## Results
- **FixedRouteBot** (deterministic east‑corridor A*): capture rate oscillates 1/0 each round → final 0.50, no learning gain.
- **RandomBot / SwitchingBot**: captured every round (1.00) because they wander into patroller corridors.
- ZoneModel prediction accuracy stays near zero (player zones are deterministic, but Laplace smoothing spreads probability).
- A* replan count per round ≈ 0 (enemies replan every 3 turns, but the batch only counts explicit `astar` calls; internal enemy replans are not hooked).

## Limitations
- Cost map only lowers cost; it never *increases* cost for unlikely tiles, so enemies are only mildly attracted.
- Tracker decay is slow; old visits linger, diluting adaptation.
- Flanker’s hottest‑zone logic assumes east exit; not general.
- Batch script monkey‑patches `pathing.astar` – misses replans triggered inside enemy `update` without going through the wrapper.
- No visualisation of enemy paths during batch runs.