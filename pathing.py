"""Pathfinding algorithms: BFS, Dijkstra, A*  (Member 2: tasks 2.2 and 2.3).

Public API is unchanged: bfs/dijkstra/astar(grid, start, goal, cost_map=None) -> (path, explored_count).
New in this version
  * flat per-tile cost arrays (StepCostMap) instead of grid.neighbors()/dict lookups per node
  * the cost map can be built ONCE per turn and shared (see ai.costs.get_step_costs);
    pass the StepCostMap instead of a dict as `cost_map`
  * A* heuristic = Manhattan * (lowest step cost): admissible and consistent even when the
    learned cost map makes tiles cheaper than 1 (task 2.3)
  * optional `margin` search window, with fallback to the full map
  * astar_result()/dijkstra_result()/bfs_result() return a SearchResult (cost, explored tiles)
  * distance_map(grid, source): one BFS for all tile distances (replaces N path_distance calls)
Step cost of a tile = max(MIN_STEP_COST, 1 + cost_map[tile]) - same rule as ai.costs.build_cost_map.
"""
from __future__ import annotations
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from grid import Grid

Position = Tuple[int, int]
CostMap = Dict[Position, float]

MIN_STEP_COST = 0.1
INF = float("inf")
_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class SearchResult:
    path: List[Position] = field(default_factory=list)
    cost: float = INF
    nodes_explored: int = 0
    explored: List[Position] = field(default_factory=list)   # expansion order, only if record=True

    @property
    def found(self) -> bool:
        return bool(self.path)


class StepCostMap:
    """Flattened step costs for one grid + one cost map. Build once, reuse for every search."""

    def __init__(self, grid: Grid, cost_map: Optional[CostMap] = None,
                 min_step_cost: float = MIN_STEP_COST):
        self.grid = grid
        self.w, self.h = grid.width, grid.height
        n = self.w * self.h
        self.walkable = bytearray(n)
        self.cost = [1.0] * n
        walls = grid.walls
        lowest = INF
        get = cost_map.get if cost_map else None
        for y in range(self.h):
            for x in range(self.w):
                if not grid.in_bounds((x, y)) or (x, y) in walls:
                    continue
                i = y * self.w + x
                self.walkable[i] = 1
                if get is not None:
                    c = 1.0 + get((x, y), 0.0)
                    if c < min_step_cost:
                        c = min_step_cost
                    self.cost[i] = c
                    if c < lowest:
                        lowest = c
        # Every step costs at least `lowest` -> Manhattan * lowest never overestimates.
        self.lowest = 1.0 if (lowest is INF or lowest > 1.0) else lowest


_plain_cache: dict = {}


def _plain_costs(grid: Grid) -> StepCostMap:
    ent = _plain_cache.get(id(grid))
    if ent is not None and ent[0] is grid and ent[1] == len(grid.walls):
        return ent[2]
    sc = StepCostMap(grid)
    _plain_cache[id(grid)] = (grid, len(grid.walls), sc)
    return sc


def _costs(grid: Grid, cost_map) -> StepCostMap:
    if isinstance(cost_map, StepCostMap):
        return cost_map
    if not cost_map:
        return _plain_costs(grid)
    return StepCostMap(grid, cost_map)


def heuristic(a: Position, b: Position) -> float:
    """Manhattan distance (kept for compatibility; astar scales it by the lowest step cost)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _bbox(costs: StepCostMap, start: Position, goal: Position, margin: Optional[int]):
    if margin is None:
        return 0, 0, costs.w - 1, costs.h - 1
    return (max(0, min(start[0], goal[0]) - margin), max(0, min(start[1], goal[1]) - margin),
            min(costs.w - 1, max(start[0], goal[0]) + margin),
            min(costs.h - 1, max(start[1], goal[1]) + margin))


def _search(costs: StepCostMap, start: Position, goal: Position, scale: float,
            bbox, record: bool) -> SearchResult:
    w = costs.w
    s, t = start[1] * w + start[0], goal[1] * w + goal[0]
    if not (costs.walkable[s] and costs.walkable[t]):
        return SearchResult()
    x0, y0, x1, y1 = bbox
    gx, gy = goal
    g = {s: 0.0}
    parent: Dict[int, int] = {}
    closed = set()
    explored: List[Position] = []
    h = costs.h
    # Heap entries are (f, g, x*h+y, index): ties break by lower g, then by (x, y) - exactly like the
    # original (priority, cost, (x, y)) tuples, so levels/routes generated with a given seed do not change.
    heap = [(scale * (abs(start[0] - gx) + abs(start[1] - gy)), 0.0, start[0] * h + start[1], s)]
    cost, walk = costs.cost, costs.walkable
    while heap:
        _, _, _, i = heapq.heappop(heap)
        if i in closed:
            continue
        closed.add(i)
        y, x = divmod(i, w)
        if record:
            explored.append((x, y))
        if i == t:
            idx = [i]
            while idx[-1] != s:
                idx.append(parent[idx[-1]])
            idx.reverse()
            return SearchResult([(p % w, p // w) for p in idx], g[t], len(closed), explored)
        gi = g[i]
        for dx, dy in _DIRS:
            nx, ny = x + dx, y + dy
            if nx < x0 or nx > x1 or ny < y0 or ny > y1:
                continue
            j = ny * w + nx
            if not walk[j] or j in closed:
                continue
            ng = gi + cost[j]
            if ng < g.get(j, INF):
                g[j] = ng
                parent[j] = i
                heapq.heappush(heap, (ng + scale * (abs(nx - gx) + abs(ny - gy)), ng, nx * h + ny, j))
    return SearchResult(nodes_explored=len(closed), explored=explored)


def astar_result(grid: Grid, start: Position, goal: Position, cost_map=None,
                 margin: Optional[int] = None, record: bool = False) -> SearchResult:
    if start == goal:
        return SearchResult([start], 0.0, 1, [start] if record else [])
    c = _costs(grid, cost_map)
    if margin is not None:
        res = _search(c, start, goal, c.lowest, _bbox(c, start, goal, margin), record)
        if res.found:
            return res
    return _search(c, start, goal, c.lowest, _bbox(c, start, goal, None), record)


def dijkstra_result(grid: Grid, start: Position, goal: Position, cost_map=None,
                    record: bool = False) -> SearchResult:
    if start == goal:
        return SearchResult([start], 0.0, 1, [start] if record else [])
    c = _costs(grid, cost_map)
    return _search(c, start, goal, 0.0, _bbox(c, start, goal, None), record)


def bfs_result(grid: Grid, start: Position, goal: Position, cost_map=None,
               record: bool = False) -> SearchResult:
    """Unweighted shortest path (ignores costs); cost = number of steps."""
    if start == goal:
        return SearchResult([start], 0.0, 1, [start] if record else [])
    c = _costs(grid, None)
    w = c.w
    s, t = start[1] * w + start[0], goal[1] * w + goal[0]
    if not (c.walkable[s] and c.walkable[t]):
        return SearchResult()
    parent = {s: -1}
    q = deque([s])
    explored: List[Position] = []
    count = 0
    while q:
        i = q.popleft()
        count += 1
        y, x = divmod(i, w)
        if record:
            explored.append((x, y))
        if i == t:
            idx = [i]
            while parent[idx[-1]] != -1:
                idx.append(parent[idx[-1]])
            idx.reverse()
            return SearchResult([(p % w, p // w) for p in idx], float(len(idx) - 1), count, explored)
        for dx, dy in _DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < c.w and 0 <= ny < c.h:
                j = ny * w + nx
                if c.walkable[j] and j not in parent:
                    parent[j] = i
                    q.append(j)
    return SearchResult(nodes_explored=count, explored=explored)


# ---- original tuple API --------------------------------------------------
def bfs(grid: Grid, start: Position, goal: Position, cost_map: CostMap = None) -> Tuple[List[Position], int]:
    r = bfs_result(grid, start, goal, cost_map)
    return r.path, r.nodes_explored


def dijkstra(grid: Grid, start: Position, goal: Position, cost_map: CostMap = None) -> Tuple[List[Position], int]:
    r = dijkstra_result(grid, start, goal, cost_map)
    return r.path, r.nodes_explored


def astar(grid: Grid, start: Position, goal: Position, cost_map: CostMap = None,
          margin: Optional[int] = None) -> Tuple[List[Position], int]:
    """A* search. cost_map may be a dict (delta per tile) or a prebuilt StepCostMap."""
    r = astar_result(grid, start, goal, cost_map, margin)
    return r.path, r.nodes_explored


# ---- all-distances BFS ---------------------------------------------------
_dist_cache: dict = {}


def distance_map(grid: Grid, source: Position) -> Dict[Position, int]:
    """Steps from `source` to every reachable tile in ONE BFS (cached per grid + source)."""
    key = (id(grid), source)
    ent = _dist_cache.get(key)
    if ent is not None and ent[0] is grid and ent[1] == len(grid.walls):
        return ent[2]
    dist: Dict[Position, int] = {}
    if grid.is_walkable(source):
        dist[source] = 0
        q = deque([source])
        while q:
            cur = q.popleft()
            d = dist[cur] + 1
            for nb in grid.neighbors(cur):
                if nb not in dist:
                    dist[nb] = d
                    q.append(nb)
    _dist_cache[key] = (grid, len(grid.walls), dist)
    return dist
