# Adaptive Enemies - Improvement Roadmap

*Generated from code review on 2026-10-25*

---

## Critical Bugs (Must Fix First)

### 1. Game Constructor API Mismatch (game/engine.py:149)
**Problem**: Game.__init__ signature expects level: int but tests pass Grid object directly.

\\python
# Current (broken when Grid passed):
def __init__(self, level: int = 1, grid: Grid = None, alpha: float = None):
    config = get_level_config(level)  # TypeError: '<' not supported between 'Grid' and 'int'
\
**Fix**: Use keyword-only args and detect which parameter was provided.

\\python
def __init__(self, *, level: int = 1, grid: Grid = None, alpha: float = None):
    if grid is not None:
        self.grid = grid
        self.level = 1
        config = get_level_config(self.level)
    else:
        self.level = level
        self.grid = get_level_grid(level)
        config = get_level_config(level)
\
### 2. build_patrol_routes Return Type (game/engine.py:38)
**Problem**: Returns List[List[Position]] (variable length), but tests expect exactly 3 routes.

\\python
return all_routes  # Can be 2, 3, or 4 routes depending on map
\
**Fix**: Return consistent structure or update tests.

### 3. Flanker Target Computation Returns None (game/enemy.py:235)
**Problem**: _compute_flank_target searches prob_map near zone boundary, but test grid east zone has no visits there.

**Fix**: Fallback to any walkable tile in hottest zone.

---

## Code Quality Improvements

### 4. Cost Map Clamping Logic (ai/costs.py:23)
**Problem**: With alpha=3.0, all costs clamp to -0.9, losing differentiation.

**Fix**: Remove hard clamp or use softer scaling.

### 5. Magic Numbers to Constants
| Magic Number | Location | Suggested Constant |
|--------------|----------|-------------------|
| 0.1 | ai/costs.py:18 | MIN_STEP_COST = 0.1 |
| 8 | game/enemy.py:30 | DEFAULT_SIGHT_RADIUS = 8 |
| 3 | game/enemy.py:31 | REPLAN_INTERVAL = 3 |
| 5 | game/engine.py:193 | DECAY_INTERVAL = 5 |
| 0.95 | game/levels.py:112 | DEFAULT_DECAY_RATE = 0.95 |

### 6. Zone Detection Fragility (grid.py:62)
**Problem**: Corridor detection relies on columns >50% walls.
**Fix**: Use explicit zone definitions per level in game/levels.py.

### 7. Patroller Waypoint Selection Ignores Distance (game/enemy.py:178)
**Problem**: _choose_next_waypoint ignores current position.
**Fix**: Weight by inverse distance.

### 8. Flanker State Machine Stale Target (game/enemy.py:214)
**Problem**: Doesnt clear flank_target on FLANK to CHASE.
**Fix**: Clear flank_target on state exit.

---

## Gameplay & Balance

### 9. Alpha Too Aggressive
**Issue**: At alpha=3.0, cost map clamps everything.
**Recommendation**: Reduce default alpha to 1.5.

### 10. Decay Interval Hardcoded (game/engine.py:193)
**Fix**: Make configurable per level via LEVEL_CONFIG[decay_interval].

### 11. Flanker Always Recomputes Target (game/enemy.py:244)
**Fix**: Track last_hottest_zone and only recompute when changed.

### 12. No Difficulty Scaling Per Round
**Recommendation**: After N escapes, increase alpha, add enemies.

### 13. Heatmap Visualization Hard to Read
**Better**: Use block characters: HEATMAP_CHARS = ' ░▒▓█'

---

## Architecture & Extensibility

### 14. Event System (Missing)
Add hooks: on_capture, on_key_pickup, on_escape, on_zone_change, on_round_end

### 15. Save/Load System (Missing)
Add save_state() and load_state() methods.

### 16. Level Selection Menu (Missing)
Hook write_levels.py into main.py.

### 17. Policy Pattern for Bots (bots.py) - Excellent
Already well-designed for ML-based bots.

---

## Testing Improvements

### 18. Fix Test Fixtures
Use game = Game(grid=self.grid) after fixing engine.py.

### 19. Add Integration Tests
Full round tests, learning verification.

### 20. Property-Based Testing
Use hypothesis for A* and LOS symmetry tests.

---

## Performance Optimizations

| Operation | Current | Optimization |
|-----------|---------|--------------|
| build_cost_map | Every enemy update | Cache per turn |
| ZoneModel.predict_next | Every UI frame | Cache prediction |
| grid.zone_of | Frequent | Precompute zone_grid |
| astar replan | Every 3 turns | Incremental A* |

---

## Implementation Order

### Phase 1: Unblock Tests (1-2 hours)
1. Fix Game.__init__ constructor API
2. Fix build_patrol_routes return type
3. Fix flanker _compute_flank_target fallback
4. Fix cost map clamping

### Phase 2: Core Polish (2-4 hours)
5. Extract magic numbers to constants
6. Fix patroller waypoint distance weighting
7. Fix flanker stale target
8. Make decay interval configurable

### Phase 3: Gameplay Balance (1-2 hours)
9. Tune alpha / cost formula
10. Add flanker target caching
11. Improve heatmap visualization
12. Add difficulty scaling

### Phase 4: Architecture (4-8 hours)
13. Event system
14. Save/load
15. Level selection menu
16. Zone definitions per level

### Phase 5: Testing & CI (2-4 hours)
17. Fix all test fixtures
18. Add integration tests
19. Add property-based tests
20. Set up GitHub Actions CI

---

## Quick Wins
- [ ] Fix Game.__init__ signature (5 min)
- [ ] Remove cost map clamp (5 min)
- [ ] Add MIN_STEP_COST constant (5 min)
- [ ] Clear flank_target on state change (5 min)
- [ ] Use HEATMAP_CHARS gradient (10 min)
- [ ] Make decay interval configurable (10 min)
- [ ] Add level select to main.py (15 min)

---

## Metrics to Track
| Metric | Target |
|--------|--------|
| Test pass rate | 100% |
| Capture rate (round 1) | ~30% |
| Capture rate (round 10) | ~70% |
| Avg turns to escape | 50-150 |
| Frame rate (Live UI) | >30 FPS |
| Memory growth per round | <1 MB |

---

*End of improvement roadmap*
