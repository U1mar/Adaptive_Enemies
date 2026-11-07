"""Learning module: Tracker for visit counts and ZoneModel for Markov chain."""
from __future__ import annotations
from collections import deque
from typing import Dict, List, Tuple, Optional
import random

Position = Tuple[int, int]
Zone = str  # 'west', 'middle', 'east'


class Tracker:
    """Tracks player visit counts with decay and maintains move history."""

    def __init__(self, decay: float = 0.95, history_len: int = 50):
        self.decay = decay
        self.visit_counts: Dict[Position, float] = {}
        self.history: deque[Position] = deque(maxlen=history_len)

    def record(self, pos: Position) -> None:
        """Record a visit to a position."""
        self.visit_counts[pos] = self.visit_counts.get(pos, 0.0) + 1.0
        self.history.append(pos)

    def decay_counts(self) -> None:
        """Apply decay factor to all visit counts."""
        for pos in self.visit_counts:
            self.visit_counts[pos] *= self.decay
        # Remove very small counts
        self.visit_counts = {p: c for p, c in self.visit_counts.items() if c > 0.01}

    def probability_map(self) -> Dict[Position, float]:
        """Return normalized probability map."""
        total = sum(self.visit_counts.values())
        if total == 0:
            return {}
        return {p: c / total for p, c in self.visit_counts.items()}

    def get_history(self) -> List[Position]:
        """Return history as list."""
        return list(self.history)


class ZoneModel:
    """Markov chain over zones with Laplace smoothing."""

    def __init__(self, zones: List[Zone] = None, alpha: float = 1.0):
        self.zones = zones or ['west', 'middle', 'east']
        self.alpha = alpha
        self.transitions: Dict[Zone, Dict[Zone, int]] = {z: {z2: 0 for z2 in self.zones} for z in self.zones}
        self.last_zone: Optional[Zone] = None

    def set_zones(self, zones: List[Zone]) -> None:
        """Update the zone list (e.g., when loading a new level)."""
        # Preserve existing transition data for zones that still exist
        old_transitions = self.transitions
        self.zones = zones
        self.transitions = {z: {z2: 0 for z2 in self.zones} for z in self.zones}
        # Restore transitions for zones that exist in both old and new
        for from_zone in self.zones:
            if from_zone in old_transitions:
                for to_zone in self.zones:
                    if to_zone in old_transitions[from_zone]:
                        self.transitions[from_zone][to_zone] = old_transitions[from_zone][to_zone]
        # Reset last_zone if it's not in new zones
        if self.last_zone not in self.zones:
            self.last_zone = None

    def record_transition(self, from_zone: Zone, to_zone: Zone) -> None:
        """Record a zone transition."""
        if from_zone in self.transitions and to_zone in self.transitions[from_zone]:
            self.transitions[from_zone][to_zone] += 1
        self.last_zone = to_zone

    def predict_next(self, current_zone: Zone = None) -> Zone:
        """Predict next zone based on current zone (or last observed)."""
        zone = current_zone or self.last_zone
        if zone is None or zone not in self.transitions:
            return random.choice(self.zones)

        counts = self.transitions[zone]
        # Laplace smoothing
        total = sum(counts.values()) + self.alpha * len(self.zones)
        probs = {z: (counts[z] + self.alpha) / total for z in self.zones}
        return max(probs, key=probs.get)

    def prediction_accuracy(self, actual_sequence: List[Zone]) -> float:
        """Calculate prediction accuracy for a sequence of zones."""
        if len(actual_sequence) < 2:
            return 0.0
        correct = 0
        for i in range(len(actual_sequence) - 1):
            predicted = self.predict_next(actual_sequence[i])
            if predicted == actual_sequence[i + 1]:
                correct += 1
        return correct / (len(actual_sequence) - 1)

    def get_transition_matrix(self) -> Dict[Zone, Dict[Zone, float]]:
        """Get normalized transition probabilities."""
        matrix = {}
        for from_zone in self.zones:
            counts = self.transitions[from_zone]
            total = sum(counts.values()) + self.alpha * len(self.zones)
            matrix[from_zone] = {z: (counts[z] + self.alpha) / total for z in self.zones}
        return matrix