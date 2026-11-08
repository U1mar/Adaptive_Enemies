"""Member 2 tests (tasks 2.1-2.7) written against the real project classes."""
import os
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grid import Grid
from pathing import (astar, astar_result, bfs, dijkstra, distance_map, StepCostMap)
from ai.costs import get_step_costs, build_cost_map
from learning import Tracker
from game.engine import Game
from game.enemy import Enemy, EnemyState, Flanker, Patroller
from game.enemy_rng import enemy_rng
from game.enemy_overlay import enemy_overlay_data
from game.levels import generate_level
from game.replay import ReplayPlayer, ReplayStore, RoundRecorder, RoundRecording
from bots import FixedRouteBot

MIN_STEP = 0.1


def random_grid(seed, n=14):
    rng = random.Random(seed)
    walls = {(x, y) for x in range(n) for y in range(n) if rng.random() < 0.2} - {(0, 0), (n - 1, n - 1)}
    cm = {(x, y): -rng.random() * 0.95 for x in range(n) for y in range(n) if rng.random() < 0.4}
    return Grid(n, n, walls, (0, 0), (0, 0), (n - 1, n - 1)), cm


def path_cost(path, cm):
    return sum(max(MIN_STEP, 1.0 + cm.get(p, 0.0)) for p in path[1:])


def reference_astar(grid, start, goal, cost_map=None):
    """The ORIGINAL pathing.astar (unscaled Manhattan, same tie-breaking) - regression oracle."""
    import heapq
    cost_map = cost_map or {}
    h = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
    pq = [(h(start, goal), 0.0, start)]
    came, best = {start: None}, {start: 0.0}
    while pq:
        _, c, cur = heapq.heappop(pq)
        if cur == goal:
            path = [cur]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return path[::-1]
        if c > best.get(cur, float("inf")):
            continue
        for nb in grid.neighbors(cur):
            nc = c + 1.0 + cost_map.get(nb, 0.0)
            if nc < best.get(nb, float("inf")):
                best[nb] = nc
                came[nb] = cur
                heapq.heappush(pq, (nc + h(nb, goal), nc, nb))
    return []


# ---------------- 2.1 RNG ----------------
def test_enemy_rng_reproducible_and_independent_of_global_random():
    a = [enemy_rng(5, 2, "patroller", 0).random() for _ in range(3)]
    random.seed(999)
    assert a == [enemy_rng(5, 2, "patroller", 0).random() for _ in range(3)]


def test_enemy_rng_streams_differ():
    base = enemy_rng(5, 2, "patroller", 0).random()
    assert base != enemy_rng(5, 2, "patroller", 1).random()
    assert base != enemy_rng(5, 3, "patroller", 0).random()
    assert base != enemy_rng(6, 2, "patroller", 0).random()


def test_enemy_module_has_no_global_random_calls():
    src = (ROOT / "game" / "enemy.py").read_text()
    bad = re.findall(r"(?<![\w.])random\.(random|choice|choices|randint|shuffle|sample|uniform)\(", src)
    assert not bad, f"global random used in game/enemy.py: {bad}"


def _play(seed_global):
    random.seed(seed_global)                       # must not matter any more
    g = Game(level=1, seed=11)
    bot = FixedRouteBot()
    log = []
    for _ in range(3):
        res = g.play_round(bot, max_turns=40)
        log.append((res.caught, res.turns, tuple(res.path), tuple(e.position for e in g.enemies)))
        g.reset_enemies()
    return log


def test_game_is_reproducible_by_seed_regardless_of_global_random():
    assert _play(1) == _play(2)


def test_enemies_get_own_rngs():
    g = Game(level=1, seed=3)
    rngs = [e.rng for e in g.enemies]
    assert len({id(r) for r in rngs}) == len(rngs)


# ---------------- 2.2 / 2.3 pathing ----------------
@pytest.mark.parametrize("seed", range(120))
def test_astar_is_optimal_with_discounted_tiles(seed):
    grid, cm = random_grid(seed)
    pd, _ = dijkstra(grid, (0, 0), (13, 13), cm)
    pa, _ = astar(grid, (0, 0), (13, 13), cm)
    assert bool(pa) == bool(pd)
    if pd:
        assert path_cost(pa, cm) == pytest.approx(path_cost(pd, cm))
        assert all(b in grid.neighbors(a) for a, b in zip(pa, pa[1:]))


def test_scaled_heuristic_fixes_inadmissible_original():
    """Count maps where the ORIGINAL A* returned a costlier path than optimal; the new one never does."""
    worse_old = 0
    for seed in range(120):
        grid, cm = random_grid(seed)
        best, _ = dijkstra(grid, (0, 0), (13, 13), cm)
        if not best:
            continue
        old = reference_astar(grid, (0, 0), (13, 13), cm)
        new, _ = astar(grid, (0, 0), (13, 13), cm)
        assert path_cost(new, cm) == pytest.approx(path_cost(best, cm))
        if path_cost(old, cm) > path_cost(best, cm) + 1e-9:
            worse_old += 1
    assert worse_old > 0          # documents WHY the fix matters (the old heuristic overestimates)


@pytest.mark.parametrize("seed", range(40))
def test_same_paths_as_original_when_costs_are_not_discounts(seed):
    """Level generation uses positive penalty maps: paths (and tie-breaks) must not change."""
    grid, _ = random_grid(seed, 16)
    rng = random.Random(seed)
    cm = {(x, y): 1.0 + rng.randint(0, 3) * 2.0 for x in range(16) for y in range(16) if rng.random() < 0.3}
    new, _ = astar(grid, (0, 0), (15, 15), cm)
    assert new == reference_astar(grid, (0, 0), (15, 15), cm)
    new, _ = astar(grid, (0, 0), (15, 15))
    assert new == reference_astar(grid, (0, 0), (15, 15))


def test_tuple_api_and_edge_cases():
    grid, _ = random_grid(1)
    assert astar(grid, (0, 0), (0, 0)) == ([(0, 0)], 1)
    assert bfs(grid, (0, 0), (0, 0)) == ([(0, 0)], 1)
    blocked = Grid(5, 5, {(2, y) for y in range(5)}, (0, 0), (0, 0), (4, 4))
    for algo in (bfs, dijkstra, astar):
        path, explored = algo(blocked, (0, 0), (4, 4))
        assert path == [] 
        explored > 0


def test_lowest_step_cost_scales_heuristic():
    g = Grid(6, 6, set(), (0, 0), (0, 0), (5, 5))
    assert StepCostMap(g).lowest == 1.0
    assert StepCostMap(g, {(2, 2): -0.9}).lowest == pytest.approx(0.1)
    assert StepCostMap(g, {(2, 2): -5.0}).lowest == pytest.approx(0.1)     # clamped like build_cost_map
    assert StepCostMap(g, {(2, 2): 4.0}).lowest == 1.0


def test_margin_falls_back_to_full_map():
    g = Grid(10, 4, {(x, 1) for x in range(0, 9)}, (0, 0), (0, 0), (9, 3))
    tight, _ = astar(g, (0, 0), (0, 2), None, margin=0)
    full, _ = astar(g, (0, 0), (0, 2))
    assert tight 
    len(tight) == len(full)


def test_distance_map_matches_path_distance():
    g = generate_level(1, 0).grid
    dm = g.distance_map(g.start)
    for pos in [(x, y) for x in range(g.width) for y in range(g.height)]:
        assert dm.get(pos) == g.path_distance(g.start, pos)
    assert distance_map(g, g.exit) == g.distance_map(g.exit)


def test_zone_of_matches_bruteforce_definition():
    """Original definition: nearest route by path distance, ties -> lowest route index."""
    g = generate_level(1, 0).grid
    for x in range(g.width):
        for y in range(g.height):
            p = (x, y)
            if not g.is_walkable(p):
                continue
            best, idx = None, None
            for i, route in enumerate(g.routes):
                ds = [d for d in (g.path_distance(p, t) for t in route) if d is not None]
                if ds and (best is None or min(ds) < best):
                    best, idx = min(ds), i
            assert g.zone_of(p) == f"route_{idx}"


# ---------------- 2.2 cost map cache ----------------
def test_step_costs_built_once_per_tracker_state():
    g = Game(level=1, seed=1)
    t = Tracker()
    t.record(g.grid.start)
    a = get_step_costs(g.grid, t, 1.5)
    assert get_step_costs(g.grid, t, 1.5) is a
    t.record((2, 1))
    assert get_step_costs(g.grid, t, 1.5) is not a
    assert get_step_costs(g.grid, t, 2.0) is not get_step_costs(g.grid, t, 1.5)


def test_step_costs_without_tracker_version_is_never_stale():
    g = Game(level=1, seed=1)

    class T:                       # a tracker that has no `version`
        def __init__(self): self.c = {}
        def probability_map(self):
            tot = sum(self.c.values()) or 1
            return {k: v / tot for k, v in self.c.items()}
    t = T()
    t.c[g.grid.start] = 1.0
    a = get_step_costs(g.grid, t, 1.5)
    t.c[(2, 1)] = 3.0
    assert get_step_costs(g.grid, t, 1.5) is not a


def test_game_init_is_fast_on_level_2():
    import time
    t0 = time.perf_counter()
    Game(level=2, seed=1)
    assert time.perf_counter() - t0 < 5.0       # was ~31 s (zone_of ran ~124k BFS)


# ---------------- flank target ----------------
def test_flank_target_matches_original_astar_per_tile_version():
    g = Game(level=1, seed=4)
    fl = next(e for e in g.enemies if isinstance(e, Flanker))
    prob = {(x, y): random.Random(x * 31 + y).random() for x in range(g.grid.width)
            for y in range(g.grid.height) if g.grid.is_walkable((x, y)) and (x + y) % 3 == 0}
    for zone in g.grid.get_zone_names():
        got = fl._compute_flank_target(g.grid, prob, zone)
        cands = []
        for y in range(g.grid.height):
            for x in range(g.grid.width):
                pos = (x, y)
                if g.grid.is_walkable(pos) and g.grid.zone_of(pos) == zone:
                    pe = reference_astar(g.grid, pos, g.grid.exit)
                    if pe and len(pe) > 1:
                        cands.append((pos, prob.get(pos, 0.0), len(pe)))
        cps = [(cp, prob.get(cp, 0.0), len(reference_astar(g.grid, cp, g.grid.exit)))
               for cp in fl.chokepoints if g.grid.zone_of(cp) == zone
               and len(reference_astar(g.grid, cp, g.grid.exit)) > 1]
        pool = cps or cands
        want = max(pool, key=lambda c: c[1] - 0.01 * c[2])[0] if pool else None
        assert got == want


# ---------------- 2.4 overlay data ----------------
def test_overlay_data_from_real_enemies():
    g = Game(level=1, seed=2)
    for _ in range(3):
        g.step(g.grid.start)
    data = enemy_overlay_data(g.enemies)
    assert len(data) == len(g.enemies)
    for d, e in zip(data, g.enemies):
        assert d["pos"] == e.position 
        d["letter"] in "PCF"
    e = g.enemies[0]
    e.state = EnemyState.CHASE
    assert enemy_overlay_data([e])[0]["letter"] == "C"


def test_explored_nodes_recorded_only_when_requested():
    g = Game(level=1, seed=2)
    e = g.enemies[0]
    e.record_explored = True
    e._move_one_step_toward(g.grid, g.grid.exit)
    assert e.last_explored 
    e.path_preview
    res = astar_result(g.grid, g.grid.start, g.grid.exit, record=True)
    assert res.explored 
    res.nodes_explored == len(res.explored)


def test_overlay_draw_headless():
    pygame = pytest.importorskip("pygame")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from game.enemy_overlay import draw_enemy_overlay
    from game.replay import draw_timeline
    pygame.init()
    surf = pygame.Surface((300, 200))
    data = [{"id": 0, "pos": (2, 2), "state": "CHASE", "letter": "C", "colour": (225, 60, 60),
             "path": [(2, 2), (3, 2)], "explored": [(2, 3)]}]
    draw_enemy_overlay(surf, data, tile=24, show_explored=True)
    draw_timeline(surf, (10, 150, 280, 14), ReplayPlayer(_rec()))


# ---------------- 2.5 replay ----------------
def _rec(n=6):
    r = RoundRecorder(round_no=3, seed=42)
    for t in range(n):
        r.capture(t, (t, 0), [(5, 5)], ["PATROL"], caught=(t == n - 1))
    return r.finish("caught", heatmap={(1, 1): 0.5})


def test_record_a_real_round_and_replay_it():
    g = Game(level=1, seed=5)
    rec = RoundRecorder(round_no=1, seed=5)
    pos = g.grid.start
    bot = FixedRouteBot()
    for t in range(8):
        pos = bot(g.grid, pos, g.got_key, [e.position for e in g.enemies])
        caught, _, _ = g.step(pos)
        rec.capture(t, pos, [e.position for e in g.enemies], [e.state.name for e in g.enemies], caught)
        if caught:
            break
    done = rec.finish("caught" if caught else "timeout", g.tracker.probability_map())
    pl = ReplayPlayer(done)
    pl.handle_key("end")
    assert pl.frame.player == done.player_path[-1] 
    pl.progress == 1.0


def test_player_scrub_and_playback():
    pl = ReplayPlayer(_rec())
    pl.step(2); pl.step(-1); assert pl.index == 1
    pl.handle_key("end"); assert pl.index == 5
    pl.handle_key("home"); assert pl.index == 0
    pl.step(-5); assert pl.index == 0
    pl.seek_fraction(0.5); assert pl.index in (2, 3)
    pl = ReplayPlayer(_rec(), fps=10)
    pl.handle_key("p"); pl.update(0.25); assert pl.index == 2
    pl.update(10); assert pl.index == 5 
    not pl.playing
    pl.handle_key("p"); assert pl.index == 0 
    pl.playing


def test_recording_json_roundtrip_with_tuple_keyed_heatmap(tmp_path):
    rec = _rec()
    f = tmp_path / "r.json"
    rec.save(str(f))
    back = RoundRecording.load(str(f))
    assert back.player_path == rec.player_path 
    back.frames[-1].caught
    assert back.heatmap == {(1, 1): 0.5}


def test_store_keeps_last_n():
    s = ReplayStore(keep=2)
    for i in range(4):
        s.add(RoundRecording(round_no=i))
    assert [r.round_no for r in s.rounds] == [2, 3]
