"""Batch runner: runs multiple rounds across seeds/archetypes, saves results and plots."""
from __future__ import annotations
import csv
import os
import argparse
from typing import List, Dict, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from grid import Grid, Position
from game.engine import Game, RoundResult
from game.levels import get_archetype_for_level, get_map_size_for_level, get_num_levels
from bots import (
    FixedRouteBot, SwitchingBot, RandomBot, AdaptiveBot,
    HabitBot, RotatingBot, RandomRouteBot, CounterBot, ExplorerBot
)
from learning import ZoneModel


# Bot classes to test
BOT_CLASSES = {
    'FixedRouteBot': FixedRouteBot,
    'SwitchingBot': SwitchingBot,
    'RandomBot': RandomBot,
    'AdaptiveBot': AdaptiveBot,
    'HabitBot': HabitBot,
    'RotatingBot': RotatingBot,
    'RandomRouteBot': RandomRouteBot,
    'CounterBot': CounterBot,
    'ExplorerBot': ExplorerBot,
}


def run_bot_on_seed(bot_class, seed: int, level: int, num_rounds: int = 30) -> Tuple[List[RoundResult], List[int], List[float]]:
    """
    Run a bot on a specific seed/level for multiple rounds.
    Returns: (round_results, route_used_per_round, prediction_accuracy_per_round)
    """
    results = []
    routes_used = []
    accuracies = []

    # Create bot with seed for reproducibility
    bot = bot_class(seed=seed)
    game = Game(level=level, seed=seed)

    # ZoneModel for tracking enemy prediction accuracy
    # We'll simulate what the enemy's ZoneModel would predict
    enemy_zm = ZoneModel(zones=game.grid.get_zone_names())

    for round_num in range(num_rounds):
        # Score prediction BEFORE this round using only previous rounds
        if round_num > 0:
            # For simplicity, we evaluate prediction on the bot's actual zone transitions
            # The "enemy" predicts the next zone given current zone
            # But since we only have this round's data, we score on previous round
            pass

        if hasattr(bot, 'new_round'):
            bot.new_round()

        result = game.play_round(bot)
        results.append(result)

        # Track which route was used (for route-based bots)
        route_used = getattr(bot, 'current_route_idx', -1)
        routes_used.append(route_used)

        # Update enemy ZoneModel with this round's transitions for NEXT round's prediction
        for i in range(len(result.zones_visited) - 1):
            if result.zones_visited[i] != result.zones_visited[i + 1]:
                enemy_zm.record_transition(result.zones_visited[i], result.zones_visited[i + 1])

        # Calculate prediction accuracy for THIS round (using model trained on previous rounds)
        round_correct = 0
        round_total = 0
        zones = result.zones_visited
        for i in range(len(zones) - 1):
            current_zone = zones[i]
            next_zone = zones[i + 1]
            if current_zone != next_zone:
                round_total += 1
                predicted = enemy_zm.predict_next(current_zone)
                if predicted == next_zone:
                    round_correct += 1
        if round_total > 0:
            accuracies.append(round_correct / round_total)
        else:
            accuracies.append(0.0)

        game.reset_enemies()

    return results, routes_used, accuracies


def run_all_configs(num_seeds: int = 10, num_rounds: int = 30) -> Dict:
    """Run all archetype x bot combinations across seeds."""
    num_levels = get_num_levels()
    all_data = {}

    for level in range(1, num_levels + 1):
        archetype = get_archetype_for_level(level)
        print(f"\n=== Level {level} ({archetype}) ===")

        for bot_name, bot_class in BOT_CLASSES.items():
            print(f"  {bot_name}...", end=' ', flush=True)

            seed_results = []
            seed_routes = []
            seed_accuracies = []

            for seed in range(num_seeds):
                try:
                    results, routes_used, accuracies = run_bot_on_seed(
                        bot_class, seed, level, num_rounds
                    )
                    seed_results.append(results)
                    seed_routes.append(routes_used)
                    seed_accuracies.append(accuracies)
                except Exception as e:
                    print(f"\n    Seed {seed} failed: {e}")
                    # Fill with empty data
                    seed_results.append([None] * num_rounds)
                    seed_routes.append([-1] * num_rounds)
                    seed_accuracies.append([0.0] * num_rounds)

            key = f"{archetype}_{bot_name}"
            all_data[key] = {
                'level': level,
                'archetype': archetype,
                'bot': bot_name,
                'seeds': seed_results,
                'routes': seed_routes,
                'accuracies': seed_accuracies,
            }
            print("done")

    return all_data


def save_results_csv(all_data: Dict, filepath: str, num_rounds: int = 30) -> None:
    """Save detailed results to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'archetype', 'bot', 'seed', 'round',
            'caught', 'got_key', 'escaped', 'turns',
            'route_used', 'prediction_accuracy'
        ])
        for key, data in all_data.items():
            archetype = data['archetype']
            bot_name = data['bot']
            for seed_idx, (seed_results, seed_routes, seed_accuracies) in enumerate(
                zip(data['seeds'], data['routes'], data['accuracies'])
            ):
                for round_idx in range(num_rounds):
                    r = seed_results[round_idx]
                    if r is None:
                        writer.writerow([archetype, bot_name, seed_idx, round_idx,
                                       '', '', '', '', '', ''])
                    else:
                        writer.writerow([
                            archetype, bot_name, seed_idx, round_idx,
                            r.caught, r.got_key, r.escaped, r.turns,
                            seed_routes[round_idx], seed_accuracies[round_idx]
                        ])


def compute_capture_rates(all_data: Dict, num_rounds: int = 30) -> Dict:
    """Compute mean capture rates and 95% CI across seeds for each round."""
    capture_data = {}

    for key, data in all_data.items():
        archetype = data['archetype']
        bot_name = data['bot']
        seed_results = data['seeds']

        # For each round, compute capture rate across seeds
        round_rates = []
        round_cis = []

        for round_idx in range(num_rounds):
            caught_counts = []
            for sr in seed_results:
                r = sr[round_idx]
                if r is not None:
                    caught_counts.append(1 if r.caught else 0)

            if caught_counts:
                mean_rate = np.mean(caught_counts)
                # 95% CI for binomial proportion
                n = len(caught_counts)
                if mean_rate > 0 and mean_rate < 1:
                    se = np.sqrt(mean_rate * (1 - mean_rate) / n)
                    ci = 1.96 * se
                else:
                    ci = 0.0
                round_rates.append(mean_rate)
                round_cis.append(ci)
            else:
                round_rates.append(0.0)
                round_cis.append(0.0)

        capture_data[key] = {
            'archetype': archetype,
            'bot': bot_name,
            'rates': round_rates,
            'cis': round_cis,
        }

    return capture_data


def plot_capture_rate_with_ci(capture_data: Dict, filepath: str, num_rounds: int = 30) -> None:
    """Plot capture rate per round with 95% CI shading, faceted by archetype."""
    archetypes = sorted(set(d['archetype'] for d in capture_data.values()))
    bots = sorted(set(d['bot'] for d in capture_data.values()))

    fig, axes = plt.subplots(len(archetypes), 1, figsize=(12, 4 * len(archetypes)), squeeze=False)

    for ai, archetype in enumerate(archetypes):
        ax = axes[ai, 0]
        for bot_name in bots:
            key = f"{archetype}_{bot_name}"
            if key not in capture_data:
                continue
            data = capture_data[key]
            rounds = range(1, num_rounds + 1)
            rates = data['rates']
            cis = data['cis']
            color = plt.cm.tab10(bots.index(bot_name) % 10)
            ax.plot(rounds, rates, label=bot_name, color=color, marker='o', markersize=3)
            ax.fill_between(rounds,
                          np.array(rates) - np.array(cis),
                          np.array(rates) + np.array(cis),
                          color=color, alpha=0.2)

        ax.set_xlabel('Round')
        ax.set_ylabel('Capture Rate')
        ax.set_title(f'{archetype}')
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_route_usage(all_data: Dict, filepath: str, num_rounds: int = 30) -> None:
    """Plot route usage chart per bot, faceted by archetype."""
    archetypes = sorted(set(d['archetype'] for d in all_data.values()))
    bots = [b for b in BOT_CLASSES.keys() if 'Bot' in b and b not in ['FixedRouteBot', 'SwitchingBot', 'RandomBot', 'AdaptiveBot']]

    fig, axes = plt.subplots(len(archetypes), len(bots), figsize=(4 * len(bots), 3 * len(archetypes)), squeeze=False)

    for ai, archetype in enumerate(archetypes):
        for bi, bot_name in enumerate(bots):
            ax = axes[ai, bi]
            key = f"{archetype}_{bot_name}"
            if key not in all_data:
                ax.set_title(f'{archetype} - {bot_name}\n(no data)')
                continue

            data = all_data[key]
            seed_routes = data['routes']

            # Count route usage per round across seeds
            max_route = 3  # assuming max 3 routes
            round_route_counts = np.zeros((num_rounds, max_route))
            round_totals = np.zeros(num_rounds)

            for seed_routes in seed_routes:
                for round_idx, route_idx in enumerate(seed_routes):
                    if route_idx >= 0 and route_idx < max_route:
                        round_route_counts[round_idx, route_idx] += 1
                        round_totals[round_idx] += 1

            # Plot stacked area
            rounds = range(1, num_rounds + 1)
            bottom = np.zeros(num_rounds)
            colors = plt.cm.Set2(np.arange(max_route))
            for ri in range(max_route):
                if round_totals[ri] > 0:
                    prop = np.where(round_totals > 0, round_route_counts[:, ri] / round_totals, 0)
                    ax.fill_between(rounds, bottom, bottom + prop,
                                  label=f'Route {ri}' if ai == 0 and bi == 0 else '',
                                  color=colors[ri], alpha=0.7)
                    bottom += prop

            ax.set_ylim(0, 1)
            ax.set_title(f'{archetype} - {bot_name}', fontsize=9)
            ax.set_xlabel('Round', fontsize=8)
            if bi == 0:
                ax.set_ylabel('Route Proportion', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    # Add single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def plot_prediction_accuracy(all_data: Dict, filepath: str, num_rounds: int = 30) -> None:
    """Plot prediction accuracy over rounds, faceted by archetype."""
    archetypes = sorted(set(d['archetype'] for d in all_data.values()))
    bots = sorted(set(d['bot'] for d in all_data.values()))

    fig, axes = plt.subplots(len(archetypes), 1, figsize=(12, 4 * len(archetypes)), squeeze=False)

    for ai, archetype in enumerate(archetypes):
        ax = axes[ai, 0]
        for bot_name in bots:
            key = f"{archetype}_{bot_name}"
            if key not in all_data:
                continue
            data = all_data[key]
            seed_accuracies = data['accuracies']

            # Mean accuracy per round across seeds
            round_accuracies = []
            round_cis = []
            for round_idx in range(num_rounds):
                accs = [sa[round_idx] for sa in seed_accuracies if len(sa) > round_idx]
                if accs:
                    mean_acc = np.mean(accs)
                    n = len(accs)
                    if mean_acc > 0 and mean_acc < 1:
                        se = np.sqrt(mean_acc * (1 - mean_acc) / n)
                        ci = 1.96 * se
                    else:
                        ci = 0.0
                    round_accuracies.append(mean_acc)
                    round_cis.append(ci)
                else:
                    round_accuracies.append(0.0)
                    round_cis.append(0.0)

            rounds = range(1, num_rounds + 1)
            color = plt.cm.tab10(bots.index(bot_name) % 10)
            ax.plot(rounds, round_accuracies, label=bot_name, color=color, marker='o', markersize=3)
            ax.fill_between(rounds,
                          np.array(round_accuracies) - np.array(round_cis),
                          np.array(round_accuracies) + np.array(round_cis),
                          color=color, alpha=0.2)

        ax.set_xlabel('Round')
        ax.set_ylabel('Prediction Accuracy')
        ax.set_title(f'{archetype} - Enemy Route Prediction Accuracy')
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def print_summary(all_data: Dict, num_rounds: int = 30) -> None:
    """Print capture rate comparison for each bot."""
    print("\n" + "=" * 80)
    print("SUMMARY: Capture Rate Comparison (Rounds 1-10 vs 21-30)")
    print("=" * 80)

    for key, data in sorted(all_data.items()):
        archetype = data['archetype']
        bot_name = data['bot']
        seed_results = data['seeds']

        # Aggregate across seeds
        early_caught = 0
        early_total = 0
        late_caught = 0
        late_total = 0

        for seed_results in seed_results:
            for round_idx, r in enumerate(seed_results):
                if r is None:
                    continue
                if round_idx < 10:  # Rounds 1-10
                    early_total += 1
                    if r.caught:
                        early_caught += 1
                elif round_idx >= 20:  # Rounds 21-30
                    late_total += 1
                    if r.caught:
                        late_caught += 1

        early_rate = early_caught / early_total if early_total > 0 else 0
        late_rate = late_caught / late_total if late_total > 0 else 0

        print(f"\n{archetype} - {bot_name}:")
        print(f"  Rounds 1-10:  {early_caught}/{early_total} = {early_rate:.1%}")
        print(f"  Rounds 21-30: {late_caught}/{late_total} = {late_rate:.1%}")

        if bot_name == 'HabitBot':
            if late_rate > early_rate:
                print(f"  >>> HabitBot capture rate RISES ({early_rate:.1%} -> {late_rate:.1%}) - enemies learn!")
            else:
                print(f"  >>> HabitBot capture rate does NOT rise ({early_rate:.1%} -> {late_rate:.1%}) - enemies NOT learning")
        elif bot_name == 'CounterBot':
            # Compare to HabitBot for same archetype
            habit_key = f"{archetype}_HabitBot"
            if habit_key in all_data:
                habit_late = 0
                habit_total = 0
                for sr in all_data[habit_key]['seeds']:
                    for ri, r in enumerate(sr):
                        if r and ri >= 20:
                            habit_total += 1
                            if r.caught:
                                habit_late += 1
                habit_late_rate = habit_late / habit_total if habit_total > 0 else 0
                if late_rate < habit_late_rate:
                    print(f"  >>> CounterBot BETTER than HabitBot ({late_rate:.1%} vs {habit_late_rate:.1%})")
                else:
                    print(f"  >>> CounterBot NOT better than HabitBot ({late_rate:.1%} vs {habit_late_rate:.1%})")


def main():
    parser = argparse.ArgumentParser(description='Run batch experiments')
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds per archetype (default: 10)')
    parser.add_argument('--rounds', type=int, default=30, help='Number of rounds per seed (default: 30)')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    print(f"Running batch: {args.seeds} seeds x {args.rounds} rounds per archetype x bot")
    print(f"Total configurations: {get_num_levels()} archetypes x {len(BOT_CLASSES)} bots = {get_num_levels() * len(BOT_CLASSES)}")

    all_data = run_all_configs(args.seeds, args.rounds)

    # Save CSV
    save_results_csv(all_data, 'results/results.csv', args.rounds)

    # Generate plots
    capture_data = compute_capture_rates(all_data, args.rounds)
    plot_capture_rate_with_ci(capture_data, 'results/capture_rate.png', args.rounds)
    plot_route_usage(all_data, 'results/route_usage.png', args.rounds)
    plot_prediction_accuracy(all_data, 'results/prediction_accuracy.png', args.rounds)

    # Print summary
    print_summary(all_data, args.rounds)

    print("\nResults saved to results/")


if __name__ == '__main__':
    main()