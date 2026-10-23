"""Benchmark pathing algorithms on different grid sizes."""
from __future__ import annotations
import time
import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from adaptive_enemies.game.grid import Grid, Pos
from adaptive_enemies.pathing import bfs, dijkstra, astar


def make_random_grid(width: int, height: int, wall_prob: float = 0.2, seed: int = None) -> Grid:
    """Generate a random grid with walls."""
    if seed is not None:
        random.seed(seed)

    walls: set[Pos] = set()
    for y in range(height):
        for x in range(width):
            if random.random() < wall_prob:
                walls.add((x, y))

    start = (0, 0)
    goal = (width - 1, height - 1)
    walls.discard(start)
    walls.discard(goal)

    return Grid(width, height, walls, start, goal, goal)


def benchmark_algorithm(algo_func, grid: Grid, start: Pos, goal: Pos, cost_map=None, runs: int = 10) -> dict:
    """Run algorithm multiple times and return avg time and explored."""
    times = []
    explored_counts = []

    for _ in range(runs):
        t0 = time.perf_counter()
        if cost_map is not None:
            result = algo_func(grid, start, goal, cost_map)
        else:
            result = algo_func(grid, start, goal)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        explored_counts.append(result.explored)

    return {
        'avg_time': sum(times) / len(times),
        'avg_explored': sum(explored_counts) / len(explored_counts),
        'times': times,
        'explored': explored_counts
    }


def main():
    random.seed(42)
    sizes = [20, 40, 80]
    algorithms = [
        ('BFS', bfs, None),
        ('Dijkstra', dijkstra, None),
        ('A*', astar, None),
    ]

    results = {size: {} for size in sizes}

    print("Running benchmarks...")
    for size in sizes:
        print(f"\nGrid size: {size}x{size}")
        # Generate a few random grids and average
        num_grids = 5
        for name, func, _ in algorithms:
            total_time = 0.0
            total_explored = 0
            valid_runs = 0

            for grid_seed in range(num_grids):
                grid = make_random_grid(size, size, wall_prob=0.2, seed=grid_seed * 100 + size)

                # Skip if start/goal unreachable (quick check)
                result = func(grid, grid.start, grid.vault)
                if result.path is None:
                    continue

                bench = benchmark_algorithm(func, grid, grid.start, grid.vault, runs=5)
                total_time += bench['avg_time']
                total_explored += bench['avg_explored']
                valid_runs += 1

            if valid_runs > 0:
                avg_time = total_time / valid_runs
                avg_explored = total_explored / valid_runs
                results[size][name] = {'time': avg_time, 'explored': avg_explored}
                print(f"  {name}: {avg_time*1000:.2f} ms, {avg_explored:.0f} nodes explored")
            else:
                results[size][name] = {'time': 0, 'explored': 0}
                print(f"  {name}: No valid grids (all unreachable)")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    algo_names = ['BFS', 'Dijkstra', 'A*']
    x = range(len(sizes))
    width = 0.25

    # Time plot
    for i, name in enumerate(algo_names):
        times = [results[size][name]['time'] * 1000 for size in sizes]  # Convert to ms
        offset = (i - 1) * width
        ax1.bar([xi + offset for xi in x], times, width, label=name, alpha=0.8)

    ax1.set_xlabel('Grid Size')
    ax1.set_ylabel('Average Time (ms)')
    ax1.set_title('Pathfinding Time by Grid Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Explored nodes plot
    for i, name in enumerate(algo_names):
        explored = [results[size][name]['explored'] for size in sizes]
        offset = (i - 1) * width
        ax2.bar([xi + offset for xi in x], explored, width, label=name, alpha=0.8)

    ax2.set_xlabel('Grid Size')
    ax2.set_ylabel('Average Nodes Explored')
    ax2.set_title('Nodes Explored by Grid Size')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('adaptive_enemies/results/path_benchmark.png', dpi=150)
    print("\nSaved benchmark chart to adaptive_enemies/results/path_benchmark.png")


if __name__ == '__main__':
    main()