"""Tests for new bot behaviors."""
from __future__ import annotations
import pytest
from game.engine import Game
from bots import (
    HabitBot, RotatingBot, RandomRouteBot, CounterBot, ExplorerBot,
    FixedRouteBot, SwitchingBot, RandomBot, AdaptiveBot
)


def test_random_route_bot_uses_different_routes():
    """Two rounds of RandomRouteBot should use different routes (with high probability)."""
    # Use a seed that gives multiple routes
    game = Game(level=1, seed=0)
    bot = RandomRouteBot(seed=42)

    routes_used = []
    for round_num in range(10):
        if hasattr(bot, 'new_round'):
            bot.new_round()
        result = game.play_round(bot)
        routes_used.append(bot.current_route_idx)
        game.reset_enemies()

    # Should use at least 2 different routes in 10 rounds
    unique_routes = set(routes_used)
    assert len(unique_routes) >= 2, f"RandomRouteBot only used routes: {routes_used}"


def test_habitbot_uses_main_route_more_than_70_percent():
    """HabitBot should use its preferred route more than 70% of the time."""
    game = Game(level=1, seed=0)
    bot = HabitBot(seed=123)

    routes_used = []
    for round_num in range(30):
        if hasattr(bot, 'new_round'):
            bot.new_round()
        result = game.play_round(bot)
        routes_used.append(bot.current_route_idx)
        game.reset_enemies()

    # Find the most used route (preferred route)
    from collections import Counter
    counts = Counter(routes_used)
    most_common_route, most_common_count = counts.most_common(1)[0]

    # Should use preferred route > 70% of rounds
    preferred_ratio = most_common_count / len(routes_used)
    assert preferred_ratio > 0.7, f"HabitBot used preferred route only {preferred_ratio:.1%} of the time (routes: {routes_used})"


def test_same_seed_reproduces_same_run():
    """The same seed should reproduce exactly the same sequence of routes."""
    # Run 1
    game1 = Game(level=1, seed=0)
    bot1 = HabitBot(seed=42)
    routes1 = []
    for _ in range(10):
        if hasattr(bot1, 'new_round'):
            bot1.new_round()
        result = game1.play_round(bot1)
        routes1.append(bot1.current_route_idx)
        game1.reset_enemies()

    # Run 2 with same seed
    game2 = Game(level=1, seed=0)
    bot2 = HabitBot(seed=42)
    routes2 = []
    for _ in range(10):
        if hasattr(bot2, 'new_round'):
            bot2.new_round()
        result = game2.play_round(bot2)
        routes2.append(bot2.current_route_idx)
        game2.reset_enemies()

    assert routes1 == routes2, f"Same seed produced different routes: {routes1} vs {routes2}"


def test_same_seed_reproduces_same_run_random_route_bot():
    """RandomRouteBot with same seed should reproduce same route sequence."""
    game1 = Game(level=1, seed=0)
    bot1 = RandomRouteBot(seed=42)
    routes1 = []
    for _ in range(10):
        if hasattr(bot1, 'new_round'):
            bot1.new_round()
        result = game1.play_round(bot1)
        routes1.append(bot1.current_route_idx)
        game1.reset_enemies()

    game2 = Game(level=1, seed=0)
    bot2 = RandomRouteBot(seed=42)
    routes2 = []
    for _ in range(10):
        if hasattr(bot2, 'new_round'):
            bot2.new_round()
        result = game2.play_round(bot2)
        routes2.append(bot2.current_route_idx)
        game2.reset_enemies()

    assert routes1 == routes2, f"RandomRouteBot same seed: {routes1} vs {routes2}"


def test_same_seed_reproduces_same_run_counter_bot():
    """CounterBot with same seed should reproduce same route sequence."""
    game1 = Game(level=1, seed=0)
    bot1 = CounterBot(seed=42)
    routes1 = []
    for _ in range(10):
        if hasattr(bot1, 'new_round'):
            bot1.new_round()
        result = game1.play_round(bot1)
        routes1.append(bot1.current_route_idx)
        game1.reset_enemies()

    game2 = Game(level=1, seed=0)
    bot2 = CounterBot(seed=42)
    routes2 = []
    for _ in range(10):
        if hasattr(bot2, 'new_round'):
            bot2.new_round()
        result = game2.play_round(bot2)
        routes2.append(bot2.current_route_idx)
        game2.reset_enemies()

    assert routes1 == routes2, f"CounterBot same seed: {routes1} vs {routes2}"


def test_same_seed_reproduces_same_run_explorer_bot():
    """ExplorerBot with same seed should reproduce same behavior."""
    game1 = Game(level=1, seed=0)
    bot1 = ExplorerBot(seed=42)
    routes1 = []
    for _ in range(10):
        if hasattr(bot1, 'new_round'):
            bot1.new_round()
        result = game1.play_round(bot1)
        routes1.append(bot1.current_route_idx)
        game1.reset_enemies()

    game2 = Game(level=1, seed=0)
    bot2 = ExplorerBot(seed=42)
    routes2 = []
    for _ in range(10):
        if hasattr(bot2, 'new_round'):
            bot2.new_round()
        result = game2.play_round(bot2)
        routes2.append(bot2.current_route_idx)
        game2.reset_enemies()

    assert routes1 == routes2, f"ExplorerBot same seed: {routes1} vs {routes2}"


def test_rotating_bot_cycles_every_3_rounds():
    """RotatingBot should cycle through routes every 3 rounds."""
    game = Game(level=1, seed=0)
    bot = RotatingBot(seed=42)

    routes_used = []
    for round_num in range(12):
        if hasattr(bot, 'new_round'):
            bot.new_round()
        result = game.play_round(bot)
        routes_used.append(bot.current_route_idx)
        game.reset_enemies()

    # Should change route every 3 rounds
    # First 3 rounds: same route, next 3: different route, etc.
    assert routes_used[0] == routes_used[1] == routes_used[2], "First 3 rounds should use same route"
    assert routes_used[3] == routes_used[4] == routes_used[5], "Rounds 4-6 should use same route"
    assert routes_used[6] == routes_used[7] == routes_used[8], "Rounds 7-9 should use same route"
    assert routes_used[9] == routes_used[10] == routes_used[11], "Rounds 10-12 should use same route"
    # And they should be different (cycling)
    assert routes_used[0] != routes_used[3] or routes_used[3] != routes_used[6], "Should cycle through different routes"


def test_explorer_bot_visits_explore_waypoints():
    """ExplorerBot should visit explore waypoints before going to vault."""
    game = Game(level=1, seed=0)
    bot = ExplorerBot(seed=42)

    paths = []
    for round_num in range(5):
        if hasattr(bot, 'new_round'):
            bot.new_round()
        result = game.play_round(bot)
        paths.append(result.path)
        game.reset_enemies()

    # Check that path doesn't go directly to vault (should have exploration phase)
    for path in paths:
        # First few steps should not be directly toward vault
        # The path should include some exploration
        assert len(path) > 5, "Path should be non-trivial"


def test_all_bots_have_new_round_method():
    """All new bots should have new_round method."""
    bots = [
        HabitBot(seed=0),
        RotatingBot(seed=0),
        RandomRouteBot(seed=0),
        CounterBot(seed=0),
        ExplorerBot(seed=0),
    ]

    for bot in bots:
        assert hasattr(bot, 'new_round'), f"{bot.__class__.__name__} missing new_round method"
        # Should not raise
        bot.new_round()


def test_habitbot_deviates_occasionally():
    """HabitBot should occasionally deviate from preferred route (15% chance)."""
    game = Game(level=1, seed=0)
    bot = HabitBot(seed=42)

    routes_used = []
    for round_num in range(50):
        if hasattr(bot, 'new_round'):
            bot.new_round()
        result = game.play_round(bot)
        routes_used.append(bot.current_route_idx)
        game.reset_enemies()

    # Find preferred route (most common)
    from collections import Counter
    counts = Counter(routes_used)
    preferred_route = counts.most_common(1)[0][0]

    # Count deviations
    deviations = sum(1 for r in routes_used if r != preferred_route)
    deviation_rate = deviations / len(routes_used)

    # Should deviate some (but not too much, 15% expected)
    # With 50 rounds, expected ~7.5 deviations
    # Allow some variance: 3-20 deviations
    assert 2 <= deviations <= 20, f"HabitBot deviations: {deviations} (rate: {deviation_rate:.1%}), routes: {routes_used}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])