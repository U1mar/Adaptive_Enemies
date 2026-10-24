"""Cost map building for adaptive enemies."""
from __future__ import annotations
from typing import Dict, Tuple, List, Optional

from grid import Grid, Position
from learning import Tracker

CostMap = Dict[Position, float]


def build_cost_map(tracker: Tracker, alpha: float, min_step_cost: float = 0.1) -> CostMap:
    """
    Build a cost map from tracker probabilities.

    For patrollers: cost = -alpha * p so that likely tiles become cheaper to walk through.
    Clamp so total step cost stays at least min_step_cost.
    """
    prob_map = tracker.probability_map()
    cost_map = {}

    for pos, prob in prob_map.items():
        # cost = -alpha * prob, making high-probability tiles cheaper
        cost = -alpha * prob
        # Clamp so total step cost (1.0 + cost) stays at least min_step_cost
        # This means cost >= min_step_cost - 1.0
        min_cost = min_step_cost - 1.0
        if cost < min_cost:
            cost = min_cost
        cost_map[pos] = cost

    return cost_map


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