"""Tests for AI components: Tracker and ZoneModel."""
from __future__ import annotations
import pytest
from adaptive_enemies.ai.tracker import Tracker
from adaptive_enemies.ai.markov import ZoneModel
from adaptive_enemies.game.grid import Pos


def test_tracker_decay_recent_outweighs_old():
    """Decay makes recent tiles have higher probability than old ones."""
    tracker = Tracker(decay=0.5, history_len=10)

    # Record position A many times
    for _ in range(10):
        tracker.record((0, 0))

    # Record position B once (recent)
    tracker.record((1, 0))

    prob_map = tracker.probability_map()

    # Recent position B should have higher probability than old position A
    # because A was decayed 10 times (0.5^10 ≈ 0.001) while B has count 1
    assert prob_map[(1, 0)] > prob_map[(0, 0)]


def test_tracker_probabilities_sum_to_one():
    """Probability map should sum to 1."""
    tracker = Tracker(decay=0.9, history_len=20)

    positions = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)]
    for pos in positions:
        tracker.record(pos)

    prob_map = tracker.probability_map()
    total = sum(prob_map.values())
    assert abs(total - 1.0) < 1e-9


def test_tracker_empty_returns_empty_dict():
    """Empty tracker returns empty probability map."""
    tracker = Tracker()
    assert tracker.probability_map() == {}


def test_markov_empty_returns_uniform():
    """Empty ZoneModel returns uniform probabilities over west/middle/east."""
    model = ZoneModel()
    probs = model.predict_next("west")

    # Should have all three zones
    assert set(probs.keys()) == {"west", "middle", "east"}
    # Should be uniform (1/3 each)
    for p in probs.values():
        assert abs(p - 1/3) < 1e-9


def test_markov_learns_east_to_east_pattern():
    """Markov model learns repeated east->east transitions."""
    model = ZoneModel()

    # Observe east -> east many times
    for _ in range(100):
        model.observe("east", "east")

    probs = model.predict_next("east")

    # east should have highest probability
    assert probs["east"] > probs["west"]
    assert probs["east"] > probs["middle"]
    # Should be close to 1 (but not exactly 1 due to Laplace smoothing)
    assert probs["east"] > 0.9


def test_markov_probabilities_sum_to_one():
    """ZoneModel predict_next probabilities should sum to 1."""
    model = ZoneModel()

    # Add some observations
    model.observe("west", "middle")
    model.observe("west", "east")
    model.observe("middle", "west")
    model.observe("east", "east")

    for zone in ["west", "middle", "east"]:
        probs = model.predict_next(zone)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])