"""Run batch simulations for all bots and collect metrics."""
from __future__ import annotations
import csv
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from grid import Grid, Position
from game.engine import Game, RoundResult, Policy
from sim.bots import fixed_route_bot, random_bot, switching_bot
from learning import ZoneModel
import pathing

# Monkey-patch astar to count calls
_original_astar = pathing.astar
_astar_call_count = 0

def counting_astar(grid: Grid, start: Position, goal: Position, cost_map: Dict[Position, float] = None):
    global _astar_call_count
    _astar_call_count += 1
    return _original_astar(grid, start, goal, cost_map)

pathing.astar = counting_astar

BOTS = {
    "FixedRouteBot": fixed_route_bot,
    "RandomBot": random_bot,
    "SwitchingBot": switching_bot,
}

ROUNDS_PER_BOT = 30
RESULTS_CSV = Path(__file__).parent.parent / "results" / "results.csv"

def run_bot(bot_name: str, bot_policy: Policy) -> List[Dict[str, Any]]:
    """Run 30 rounds for a single bot, return list of round metrics."""
    game = Game()
    metrics = []
    for round_num in range(1, ROUNDS_PER_BOT + 1):
        global _astar_call_count
        _astar_call_count = 0

        # Notify bot of new round if it has new_round method
        if hasattr(bot_policy, 'new_round'):
            bot_policy.new_round()

        result: RoundResult = game.play_round(bot_policy, max_turns=500)

        # Prediction accuracy for this round
        pred_acc = game.zone_model.prediction_accuracy(result.zones_visited)

        metrics.append({
            "bot": bot_name,
            "round": round_num,
            "caught": int(result.caught),
            "escaped": int(result.escaped),
            "prediction_accuracy": pred_acc,
            "replan_count": _astar_call_count,
        })

        # Reset enemies for next round (tracker and zone_model persist)
        game.reset_enemies()

    return metrics

def main():
    all_metrics = []
    for bot_name, bot_policy in BOTS.items():
        print(f"Running {bot_name} for {ROUNDS_PER_BOT} rounds...")
        metrics = run_bot(bot_name, bot_policy)
        all_metrics.extend(metrics)
        # Print capture rate trend
        caught_rates = [sum(m['caught'] for m in metrics[:i]) / i for i in range(1, len(metrics)+1)]
        print(f"  Final capture rate: {caught_rates[-1]:.3f}")
        # Check if rising
        if caught_rates[-1] > caught_rates[0]:
            print(f"  Capture rate increased from {caught_rates[0]:.3f} to {caught_rates[-1]:.3f}")

    # Write CSV
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["bot", "round", "caught", "escaped", "prediction_accuracy", "replan_count"])
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"Results saved to {RESULTS_CSV}")

if __name__ == "__main__":
    main()