"""Batch runner: runs multiple rounds, saves results and plots."""
from __future__ import annotations
import csv
import os
from typing import List, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grid import Grid, Position, DEFAULT_MAP
from game.engine import Game, RoundResult
from bots import FixedRouteBot, SwitchingBot, RandomBot
from pathing import bfs, dijkstra, astar
from learning import Tracker, ZoneModel


def run_bot_rounds(bot_class, num_rounds: int = 30) -> List[RoundResult]:
    """Run multiple rounds with a bot, returning results.
    Creates Game ONCE and reuses it, calling reset_enemies() between rounds
    so learning (tracker, zone_model) persists across rounds.
    """
    results = []
    bot = bot_class()
    game = Game()  # Create ONCE
    for round_num in range(num_rounds):
        # For SwitchingBot, call new_round
        if hasattr(bot, 'new_round'):
            bot.new_round()
        result = game.play_round(bot)
        results.append(result)
        # Reset enemies but keep tracker/zone_model for learning persistence
        game.reset_enemies()
    return results


def run_all_bots(num_rounds: int = 30) -> Dict[str, List[RoundResult]]:
    """Run all bots and return results."""
    bots = {
        'FixedRouteBot': FixedRouteBot,
        'SwitchingBot': SwitchingBot,
        'RandomBot': RandomBot,
    }
    all_results = {}
    for name, bot_class in bots.items():
        print(f"Running {name} for {num_rounds} rounds...")
        all_results[name] = run_bot_rounds(bot_class, num_rounds)
    return all_results


def save_results_csv(results: Dict[str, List[RoundResult]], filepath: str) -> None:
    """Save results to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['bot', 'round', 'caught', 'got_key', 'escaped', 'turns', 'path_len'])
        for bot_name, round_results in results.items():
            for i, r in enumerate(round_results):
                writer.writerow([bot_name, i, r.caught, r.got_key, r.escaped, r.turns, len(r.path)])


def plot_capture_rate(results: Dict[str, List[RoundResult]], filepath: str) -> None:
    """Plot capture rate per round for each bot."""
    plt.figure(figsize=(10, 6))
    for bot_name, round_results in results.items():
        capture_rates = []
        caught_count = 0
        for i, r in enumerate(round_results):
            if r.caught:
                caught_count += 1
            capture_rates.append(caught_count / (i + 1))
        plt.plot(range(1, len(round_results) + 1), capture_rates, label=bot_name, marker='o', markersize=3)
    plt.xlabel('Round')
    plt.ylabel('Capture Rate')
    plt.title('Capture Rate per Round')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_markov_accuracy(results: Dict[str, List[RoundResult]], filepath: str) -> None:
    """Plot Markov prediction accuracy per round.

    For each round, predict each zone change BEFORE observing it,
    using only data from PREVIOUS rounds. Count only zone changes (transitions).
    """
    plt.figure(figsize=(10, 6))
    for bot_name, round_results in results.items():
        accuracies = []
        # ZoneModel trained on all previous rounds
        zm = ZoneModel()

        for round_idx, r in enumerate(round_results):
            # Predict each zone transition in this round using model trained on previous rounds
            round_correct = 0
            round_total = 0

            zones = r.zones_visited
            for i in range(len(zones) - 1):
                current_zone = zones[i]
                next_zone = zones[i + 1]

                # Only count zone CHANGES
                if current_zone != next_zone:
                    round_total += 1
                    predicted = zm.predict_next(current_zone)
                    if predicted == next_zone:
                        round_correct += 1

            # Record accuracy for this round (only if there were zone changes)
            if round_total > 0:
                accuracies.append(round_correct / round_total)
            else:
                accuracies.append(0.0)

            # NOW update the model with this round's transitions (for future rounds)
            for i in range(len(zones) - 1):
                if zones[i] != zones[i + 1]:
                    zm.record_transition(zones[i], zones[i + 1])

        # Running average of per-round accuracies
        running_avg = [sum(accuracies[:i+1])/(i+1) for i in range(len(accuracies))]
        plt.plot(range(1, len(round_results) + 1), running_avg, label=bot_name, marker='o', markersize=3)

    plt.xlabel('Round')
    plt.ylabel('Markov Prediction Accuracy (on zone changes)')
    plt.title('Markov Model Prediction Accuracy (predict before observing)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_tracker_heatmap(results: Dict[str, List[RoundResult]], filepath: str, grid: Grid) -> None:
    """Plot tracker heatmap from last round of each bot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (bot_name, round_results) in enumerate(results.items()):
        if not round_results:
            continue
        # Use the last round's tracker data
        last_result = round_results[-1]
        # Build heatmap from path
        heatmap = np.zeros((grid.height, grid.width))
        for pos in last_result.path:
            x, y = pos
            if 0 <= x < grid.width and 0 <= y < grid.height:
                heatmap[y, x] += 1
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        im = axes[idx].imshow(heatmap, cmap='hot', interpolation='nearest')
        axes[idx].set_title(f'{bot_name} - Last Round')
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        plt.colorbar(im, ax=axes[idx])
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_pathfinding_comparison(filepath: str) -> None:
    """Bar chart comparing nodes explored by BFS, Dijkstra, A* on 3 grid sizes."""
    # Create 3 test grids of different sizes
    sizes = [(15, 10), (30, 15), (45, 20)]
    algorithms = [('BFS', bfs), ('Dijkstra', dijkstra), ('A*', astar)]

    results = {name: [] for name, _ in algorithms}

    for w, h in sizes:
        # Create a simple open grid
        ascii_map = ['#' * w]
        for y in range(h - 2):
            ascii_map.append('#' + '.' * (w - 2) + '#')
        ascii_map.append('#' * w)
        ascii_map = '\n'.join(ascii_map)

        # Add S and E
        lines = ascii_map.split('\n')
        lines[1] = '#S' + '.' * (w - 3) + '#'
        lines[-2] = '#' + '.' * (w - 3) + 'E#'
        ascii_map = '\n'.join(lines)

        grid = Grid.from_ascii(ascii_map)
        start = grid.start
        goal = grid.exit

        for name, algo in algorithms:
            _, explored = algo(grid, start, goal)
            results[name].append(explored)

    # Plot
    x = np.arange(len(sizes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, _) in enumerate(algorithms):
        offset = (i - 1) * width
        ax.bar(x + offset, results[name], width, label=name)

    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Nodes Explored')
    ax.set_title('Pathfinding Algorithm Comparison: Nodes Explored')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{w}x{h}' for w, h in sizes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def main():
    """Main batch run function."""
    os.makedirs('results', exist_ok=True)

    # Run all bots
    results = run_all_bots(30)

    # Save CSV
    save_results_csv(results, 'results/results.csv')

    # Generate plots
    grid = Grid.from_ascii(DEFAULT_MAP)
    plot_capture_rate(results, 'results/capture_rate.png')
    plot_markov_accuracy(results, 'results/markov_accuracy.png')
    plot_tracker_heatmap(results, 'results/heatmap.png', grid)
    plot_pathfinding_comparison('results/pathfinding_comparison.png')

    # Print summary
    print("\n=== Batch Run Complete ===")
    for bot_name, round_results in results.items():
        caught = sum(1 for r in round_results if r.caught)
        escaped = sum(1 for r in round_results if r.escaped)
        avg_turns = sum(r.turns for r in round_results) / len(round_results)
        print(f"{bot_name}: Captured {caught}/30, Escaped {escaped}/30, Avg turns: {avg_turns:.1f}")

    # Check if FixedRouteBot capture rate rises (rounds 1-10 vs 21-30)
    fr_results = results['FixedRouteBot']
    early_caught = sum(1 for r in fr_results[:10] if r.caught)
    late_caught = sum(1 for r in fr_results[20:30] if r.caught)  # rounds 21-30 (indices 20-29)
    print(f"\nFixedRouteBot: Early rounds (1-10) capture rate: {early_caught}/10")
    print(f"FixedRouteBot: Late rounds (21-30) capture rate: {late_caught}/10")
    if late_caught > early_caught:
        print(">>> FixedRouteBot capture rate RISES across rounds (adaptive enemies learning)")
    else:
        print(">>> FixedRouteBot capture rate does not clearly rise")

    print("\nResults saved to results/")


if __name__ == '__main__':
    main()