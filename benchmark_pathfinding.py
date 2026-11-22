"""benchmark_pathfinding.py - task 2.6: BFS vs Dijkstra vs A* (nodes explored + runtime).

    python benchmark_pathfinding.py            # writes results/member2_pathfinding_comparison.png + .csv
    python benchmark_pathfinding.py --levels   # same comparison on the 4 real game levels
Uses the project's real Grid and pathing.py. Learned-style cost discounts are applied to 30% of tiles.
"""
from __future__ import annotations
import argparse, csv, os, random, statistics, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid import Grid
from pathing import StepCostMap, astar_result, bfs_result, dijkstra_result

ALGOS = ("BFS", "Dijkstra", "A*", "Dijkstra (no discounts)", "A* (no discounts)")


def random_case(n: int, seed: int):
    rng = random.Random(seed)
    walls = {(x, y) for x in range(n) for y in range(n) if rng.random() < 0.22} - {(0, 0), (n - 1, n - 1)}
    return Grid(n, n, walls, (0, 0), (0, 0), (n - 1, n - 1))


def discounts(grid, seed):
    rng = random.Random(seed + 1)
    return {(x, y): -rng.random() * 0.9 for x in range(grid.width) for y in range(grid.height)
            if rng.random() < 0.3}


def timed(fn, *a):
    t0 = time.perf_counter()
    r = fn(*a)
    return r, (time.perf_counter() - t0) * 1000


def measure(grid, start, goal, cm, acc):
    costs, plain = StepCostMap(grid, cm), StepCostMap(grid)
    b, tb = timed(bfs_result, grid, start, goal)
    if not b.found:
        return False
    d, td = timed(dijkstra_result, grid, start, goal, costs)
    a, ta = timed(astar_result, grid, start, goal, costs)
    d2, td2 = timed(dijkstra_result, grid, start, goal, plain)
    a2, ta2 = timed(astar_result, grid, start, goal, plain)
    assert abs(d.cost - a.cost) < 1e-9 and abs(d2.cost - a2.cost) < 1e-9, "A* must match Dijkstra"
    for k, r, t in zip(ALGOS, (b, d, a, d2, a2), (tb, td, ta, td2, ta2)):
        acc[k]["nodes"].append(r.nodes_explored)
        acc[k]["ms"].append(t)
    return True


def run_sizes(sizes=(15, 30, 45, 60, 90), trials=8):
    rows = []
    for n in sizes:
        acc = {k: {"nodes": [], "ms": []} for k in ALGOS}
        done, seed = 0, 0
        while done < trials:
            seed += 1
            g = random_case(n, seed * 1000 + n)
            done += measure(g, (0, 0), (n - 1, n - 1), discounts(g, seed), acc)
        rows += [{"size": n, "algorithm": k, "nodes": statistics.mean(v["nodes"]),
                  "ms": statistics.mean(v["ms"])} for k, v in acc.items()]
    return rows


def run_levels(seeds=range(5)):
    from game.levels import generate_level
    rows = []
    for lvl in (1, 2, 3, 4):
        acc = {k: {"nodes": [], "ms": []} for k in ALGOS}
        for sd in seeds:
            lv = generate_level(lvl, sd)
            g = lv.grid
            measure(g, lv.start, lv.exit, discounts(g, sd), acc)
        rows += [{"size": f"L{lvl} ({g.width}x{g.height})", "algorithm": k,
                  "nodes": statistics.mean(v["nodes"]), "ms": statistics.mean(v["ms"])}
                 for k, v in acc.items()]
    return rows


def plot(rows, png, xlabel):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colours = {"BFS": "#7b6fd6", "Dijkstra": "#e08a2c", "A*": "#2a9d6f",
               "Dijkstra (no discounts)": "#e0b02c", "A* (no discounts)": "#1b6b4a"}
    for k, c in colours.items():
        sub = [r for r in rows if r["algorithm"] == k]
        xs = [str(r["size"]) for r in sub]
        style = "s--" if "no disc" in k else "o-"
        axes[0].plot(xs, [r["nodes"] for r in sub], style, label=k, color=c)
        axes[1].plot(xs, [r["ms"] for r in sub], style, label=k, color=c)
    axes[0].set(title="Nodes explored", xlabel=xlabel, ylabel="nodes (mean)")
    axes[1].set(title="Runtime", xlabel=xlabel, ylabel="ms (mean)")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("BFS vs Dijkstra vs A* (learned cost discounts on 30% of tiles)")
    fig.tight_layout(); fig.savefig(png, dpi=160)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", action="store_true")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    tag = "levels" if a.levels else "sizes"
    rows = run_levels() if a.levels else run_sizes()
    with open(f"results/member2_pathfinding_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["size", "algorithm", "nodes", "ms"])
        w.writeheader(); w.writerows(rows)
    plot(rows, f"results/member2_pathfinding_{tag}.png", "level" if a.levels else "map side (tiles)")
    for r in rows:
        print(f"{str(r['size']):>12} {r['algorithm']:<24} nodes={r['nodes']:>8.0f} ms={r['ms']:>8.2f}")
