"""Cost map building for adaptive enemies."""
from __future__ import annotations
from typing import Dict, Tuple, List, Optional

from grid import Grid, Position
from learning import Tracker
from game.constants import MIN_STEP_COST
from pathing import StepCostMap

CostMap = Dict[Position, float]


def build_cost_map(tracker: Tracker, alpha: float, min_step_cost: float = MIN_STEP_COST) -> CostMap:
    """
    Build a cost map from tracker probabilities.

    For patrollers: cost = -alpha * normalized_prob so that likely tiles become cheaper to walk through.
    Normalize probabilities by maximum so costs are in reasonable range.
    Clamp so total step cost stays at least min_step_cost.
    Note: With reduced default alpha (1.5), clamping is less aggressive and preserves more differentiation.
    """
    prob_map = tracker.probability_map()
    cost_map = {}

    if not prob_map:
        return cost_map

    # Normalize by maximum probability
    max_prob = max(prob_map.values()) if prob_map.values() else 1.0

    # Clamp so total step cost (1.0 + cost) stays at least min_step_cost
    # This means cost >= min_step_cost - 1.0
    min_cost = min_step_cost - 1.0

    for pos, prob in prob_map.items():
        # Normalize probability to [0, 1] range
        normalized_prob = prob / max_prob if max_prob > 0 else 0.0
        # cost = -alpha * normalized_prob, making high-probability tiles cheaper
        cost = -alpha * normalized_prob
        # Hard clamp to ensure positive step costs
        if cost < min_cost:
            cost = min_cost
        cost_map[pos] = cost

    return cost_map


_step_cache: dict = {}


def get_step_costs(grid: Grid, tracker: Tracker, alpha: float) -> StepCostMap:
    """StepCostMap for the tracker's current state, built once and shared by every enemy.

    The cache key is (grid, tracker, tracker.version, alpha); tracker.version changes on every
    record()/decay_counts(), so the map is rebuilt at most once per turn instead of once per enemy
    per step. If the tracker has no `version`, nothing is cached (always correct, just slower).
    """
    ver = getattr(tracker, 'version', None)
    ent = _step_cache.get('last')
    if ver is not None and ent is not None:
        g, t, v, a, sc = ent
        if g is grid and t is tracker and v == ver and a == alpha:
            return sc
    sc = StepCostMap(grid, build_cost_map(tracker, alpha))
    if ver is not None:
        _step_cache['last'] = (grid, tracker, ver, alpha, sc)
    return sc


def build_flank_targets(grid: Grid, prob_map: Dict[Position, float]) -> Optional[Position]:
    """
    Return the most likely tile near the exit-side of the hottest zone.

    The hottest zone is the zone with the highest total probability.
    The exit-side means the boundary tile of that zone closest to the exit zone (east).
    """
    if not prob_map:
        return None

    # Calculate heat per zone
    zone_heat: Dict[str, float] = {'west': 0.0, 'middle': 0.0, 'east': 0.0}
    for pos, prob in prob_map.items():
        zone = grid.zone_of(pos)
        zone_heat[zone] += prob

    # Find hottest zone
    hottest_zone = max(zone_heat, key=zone_heat.get)

    # Determine exit-side boundary of the hottest zone
    # Exit is in the east, so we want the eastern boundary of the hottest zone
    third = grid.width // 3

    if hottest_zone == 'west':
        # East boundary of west zone
        target_x = third - 1
    elif hottest_zone == 'middle':
        # East boundary of middle zone
        target_x = 2 * third - 1
    else:  # east zone
        # East boundary of east zone (near exit)
        target_x = grid.width - 2

    # Find the most probable walkable tile near the exit-side boundary
    best_pos = None
    best_prob = -1.0

    # Search in a small window around the boundary
    search_range = 3
    for y in range(grid.height):
        for x in range(max(0, target_x - search_range), min(grid.width, target_x + search_range + 1)):
            pos = (x, y)
            if grid.is_walkable(pos) and pos in prob_map:
                if prob_map[pos] > best_prob:
                    best_prob = prob_map[pos]
                    best_pos = pos

    # If no tile found in prob_map near boundary, find any walkable tile near boundary
    if best_pos is None:
        for y in range(grid.height):
            for x in range(max(0, target_x - search_range), min(grid.width, target_x + search_range + 1)):
                pos = (x, y)
                if grid.is_walkable(pos):
                    return pos

    return best_pos