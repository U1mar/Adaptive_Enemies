"""game/replay.py - round replay system (task 2.5).

Record  : RoundRecorder.capture(...) once per turn (call it right after Game.step()).
          heatmap = tracker.probability_map() (a {(x, y): p} dict) is snapshotted at round end.
Store   : ReplayStore keeps the last N rounds.
Playback: ReplayPlayer - P play/pause, LEFT/RIGHT scrub, HOME/END jump, timeline bar.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Frame:
    turn: int
    player: tuple
    guards: list                      # [(x, y), ...]
    states: list                      # ["PATROL", "CHASE", ...]
    caught: bool = False


@dataclass
class RoundRecording:
    round_no: int = 0
    seed: Optional[int] = None
    frames: list = field(default_factory=list)
    heatmap: object = None            # snapshot of tracker.probability_map() at round end
    result: Optional[str] = None      # "caught" | "escaped"

    def __len__(self):
        return len(self.frames)

    @property
    def player_path(self) -> list:
        return [f.player for f in self.frames]

    def to_dict(self) -> dict:
        d = asdict(self)
        hm = self.heatmap
        if isinstance(hm, dict):                  # tracker.probability_map(): {(x, y): p}
            d["heatmap"] = {"tiles": [[x, y, v] for (x, y), v in hm.items()]}
        elif hasattr(hm, "tolist"):
            d["heatmap"] = hm.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RoundRecording":
        frames = [Frame(f["turn"], tuple(f["player"]), [tuple(g) for g in f["guards"]],
                        list(f["states"]), f.get("caught", False)) for f in d["frames"]]
        hm = d.get("heatmap")
        if isinstance(hm, dict) and "tiles" in hm:
            hm = {(int(x), int(y)): v for x, y, v in hm["tiles"]}
        return cls(d["round_no"], d.get("seed"), frames, hm, d.get("result"))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path: str) -> "RoundRecording":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def _snapshot(hm):
    if hm is None:
        return None
    return hm.copy() if hasattr(hm, "copy") and not isinstance(hm, (list, dict)) else copy.deepcopy(hm)


class RoundRecorder:
    def __init__(self, round_no: int = 0, seed: Optional[int] = None):
        self.rec = RoundRecording(round_no=round_no, seed=seed)

    def capture(self, turn: int, player_pos, guard_positions, guard_states, caught=False):
        self.rec.frames.append(Frame(turn, tuple(player_pos), [tuple(g) for g in guard_positions],
                                     [str(s) for s in guard_states], bool(caught)))

    def finish(self, result: str, heatmap=None) -> RoundRecording:
        self.rec.result = result
        self.rec.heatmap = _snapshot(heatmap)
        return self.rec


class ReplayStore:
    def __init__(self, keep: int = 20):
        self.keep, self.rounds = keep, []

    def add(self, rec: RoundRecording) -> None:
        self.rounds.append(rec)
        del self.rounds[:-self.keep]

    @property
    def latest(self) -> Optional[RoundRecording]:
        return self.rounds[-1] if self.rounds else None


class ReplayPlayer:
    def __init__(self, rec: RoundRecording, fps: float = 4.0):
        self.rec, self.fps = rec, fps
        self.index, self.playing, self._acc = 0, False, 0.0

    @property
    def last(self) -> int:
        return max(0, len(self.rec) - 1)

    @property
    def frame(self) -> Optional[Frame]:
        return self.rec.frames[self.index] if self.rec.frames else None

    @property
    def progress(self) -> float:
        return self.index / self.last if self.last else 0.0

    def toggle(self) -> None:
        if not self.playing and self.index >= self.last:
            self.index = 0                       # replay from the start when finished
        self.playing = not self.playing

    def seek(self, i: int) -> None:
        self.index = min(max(0, i), self.last)

    def seek_fraction(self, f: float) -> None:
        self.seek(round(min(max(f, 0.0), 1.0) * self.last))

    def step(self, delta: int) -> None:
        self.playing = False                     # manual scrub pauses playback
        self.seek(self.index + delta)

    def update(self, dt: float) -> None:
        if not self.playing:
            return
        self._acc += dt * self.fps
        while self._acc >= 1.0:
            self._acc -= 1.0
            if self.index >= self.last:
                self.playing = False
                break
            self.index += 1

    def handle_key(self, key: str) -> bool:
        key = key.lower()
        actions = {"p": self.toggle, "left": lambda: self.step(-1), "right": lambda: self.step(1),
                   "home": lambda: self.seek(0), "end": lambda: self.seek(self.last)}
        if key in actions:
            actions[key]()
            return True
        return False


def draw_timeline(surface, rect, player: ReplayPlayer, font=None):
    """Timeline bar: track, filled progress, a tick per turn, caught marker, handle."""
    import pygame
    x, y, w, h = rect
    pygame.draw.rect(surface, (45, 45, 55), rect, border_radius=4)
    pygame.draw.rect(surface, (90, 140, 230), (x, y, int(w * player.progress), h), border_radius=4)
    n = len(player.rec)
    for k, f in enumerate(player.rec.frames):
        tx = x + int(w * (k / player.last if player.last else 0))
        if f.caught:
            pygame.draw.line(surface, (225, 60, 60), (tx, y - 3), (tx, y + h + 3), 3)
        elif n <= 60:
            pygame.draw.line(surface, (120, 120, 135), (tx, y + h - 4), (tx, y + h), 1)
    hx = x + int(w * player.progress)
    pygame.draw.circle(surface, (255, 255, 255), (hx, y + h // 2), h // 2 + 2)
    if font:
        label = f"Round {player.rec.round_no}  turn {player.index + 1}/{max(n, 1)}  " \
                f"{'PLAY' if player.playing else 'PAUSE'}"
        surface.blit(font.render(label, True, (230, 230, 230)), (x, y - 22))
