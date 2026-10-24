"""Generate plots from batch simulation results."""
from __future__ import annotations
import sys
from pathlib import Path
import csv
from typing import List, Dict
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

from grid import Grid, Position
from game.engine import Game
from sim.bots import fixed_route_bot
from learning import Tracker

RESULTS_CSV = Path(__file__).parent.parent / "results" / "results.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_results() -> List[Dict]:
    """Load metrics from CSV."""
    with open(RESULTS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row['round'] = int(row['round'])
            row['caught'] = int(row['caught'])
            row['escaped'] = int(row['escaped'])
            row['prediction_accuracy'] = float(row['prediction_accuracy'])
            row['replan_count'] = int(row['replan_count'])
            rows.append(row)
    return rows

def rolling_average(data: List[float], window: int = 5) -> List[float]:
    """Compute rolling average."""
    if len(data) < window:
        return data
    return [np.mean(data[i-window+1:i+1]) for i in range(window-1, len(data))]

def plot_capture_rate(rows: List[Dict]):
    """Plot capture rate per round (rolling avg over 5) for each bot."""
    bots = sorted(set(r['bot'] for r in rows))
    plt.figure(figsize=(10, 6))
    for bot in bots:
        bot_data = [r for r in rows if r['bot'] == bot]
        bot_data.sort(key=lambda x: x['round'])
        rounds = [r['round'] for r in bot_data]
        caught = [r['caught'] for r in bot_data]
        # cumulative capture rate per round
        cum_rate = [sum(caught[:i+1]) / (i+1) for i in range(len(caught))]
        # rolling average of capture rate over 5 rounds
        rolling = rolling_average(cum_rate, 5)
        # align rounds with rolling (starting at round 5)
        plt.plot(rounds[4:], rolling, label=bot, marker='o')
    plt.xlabel('Round')
    plt.ylabel('Capture Rate (Rolling 5-round Avg)')
    plt.title('Capture Rate per Round by Bot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'capture_rate.png', dpi=150)
    plt.close()

def plot_prediction_accuracy(rows: List[Dict]):
    """Plot Markov prediction accuracy over rounds for each bot."""
    bots = sorted(set(r['bot'] for r in rows))
    plt.figure(figsize=(10, 6))
    for bot in bots:
        bot_data = [r for r in rows if r['bot'] == bot]
        bot_data.sort(key=lambda x: x['round'])
        rounds = [r['round'] for r in bot_data]
        acc = [r['prediction_accuracy'] for r in bot_data]
        plt.plot(rounds, acc, label=bot, marker='s')
    plt.xlabel('Round')
    plt.ylabel('Prediction Accuracy')
    plt.title('Markov Prediction Accuracy Over Rounds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'prediction_accuracy.png', dpi=150)
    plt.close()

def plot_heatmap():
    """Run a FixedRouteBot game and plot tracker heatmap."""
    game = Game()
    # Run 30 rounds to accumulate tracker data
    for _ in range(30):
        game.play_round(fixed_route_bot, max_turns=500)
        game.reset_enemies()

    tracker = game.tracker
    prob_map = tracker.probability_map()

    # Create grid for heatmap
    grid = game.grid
    heatmap_data = np.zeros((grid.height, grid.width))
    for (x, y), prob in prob_map.items():
        if 0 <= x < grid.width and 0 <= y < grid.height:
            heatmap_data[y, x] = prob

    plt.figure(figsize=(10, 8))
    # Mask walls
    mask = np.zeros_like(heatmap_data, dtype=bool)
    for (x, y) in grid.walls:
        if 0 <= x < grid.width and 0 <= y < grid.height:
            mask[y, x] = True
    masked_data = np.ma.masked_where(mask, heatmap_data)

    plt.imshow(masked_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Visit Probability')
    # Mark special positions
    plt.scatter([game.grid.start[0]], [game.grid.start[1]], c='cyan', s=100, marker='s', label='Start')
    plt.scatter([game.grid.vault[0]], [game.grid.vault[1]], c='magenta', s=100, marker='*', label='Vault')
    plt.scatter([game.grid.exit[0]], [game.grid.exit[1]], c='green', s=100, marker='D', label='Exit')
    plt.title('Tracker Heatmap after FixedRouteBot (30 rounds)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tracker_heatmap.png', dpi=150)
    plt.close()

def main():
    rows = load_results()
    plot_capture_rate(rows)
    plot_prediction_accuracy(rows)
    plot_heatmap()
    print(f"Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()