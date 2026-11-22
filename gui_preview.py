"""gui_preview.py - standalone Pygame preview for Member 2's GUI parts (tasks 2.4 and 2.5).

Lets you test the enemy overlay and the replay system WITHOUT waiting for Umar's gui.py.
When gui.py exists, copy the marked hook lines from here into it.

    python gui_preview.py [level] [seed]

PLAY    : WASD / arrows move, E = explored-node overlay, P = path overlay, R = new round, V = replay, Q = quit
REPLAY  : P play/pause, LEFT/RIGHT scrub, HOME/END jump, ESC back to play
"""
from __future__ import annotations
import sys

import pygame

from game.engine import Game
from game.enemy_overlay import enemy_overlay_data, draw_enemy_overlay, style_for
from game.replay import RoundRecorder, ReplayStore, ReplayPlayer, draw_timeline

TILE = 24
BG, FLOOR, WALL = (18, 18, 24), (52, 54, 66), (28, 28, 36)
START, VAULT, EXIT, PLAYER = (90, 180, 120), (230, 200, 70), (90, 150, 230), (245, 245, 245)
KEYS = {pygame.K_w: (0, -1), pygame.K_UP: (0, -1), pygame.K_s: (0, 1), pygame.K_DOWN: (0, 1),
        pygame.K_a: (-1, 0), pygame.K_LEFT: (-1, 0), pygame.K_d: (1, 0), pygame.K_RIGHT: (1, 0)}
REPLAY_KEYS = {pygame.K_p: "p", pygame.K_LEFT: "left", pygame.K_RIGHT: "right",
               pygame.K_HOME: "home", pygame.K_END: "end"}


def draw_map(screen, grid):
    for y in range(grid.height):
        for x in range(grid.width):
            p = (x, y)
            colour = WALL if grid.is_wall(p) else FLOOR
            if p == grid.start: colour = START
            elif p == grid.vault: colour = VAULT
            elif p == grid.exit: colour = EXIT
            pygame.draw.rect(screen, colour, (x * TILE, y * TILE, TILE - 1, TILE - 1))


def draw_player(screen, pos):
    pygame.draw.circle(screen, PLAYER, (pos[0] * TILE + TILE // 2, pos[1] * TILE + TILE // 2), TILE // 2 - 3)


def draw_frame_guards(screen, frame, font):
    """Replay: draw the recorded guard positions with the same letters/colours as live play."""
    for pos, state in zip(frame.guards, frame.states):
        letter, colour = style_for(state.upper())
        rect = pygame.Rect(pos[0] * TILE + 1, pos[1] * TILE + 1, TILE - 2, TILE - 2)
        pygame.draw.rect(screen, colour, rect, border_radius=5)
        label = font.render(letter, True, (255, 255, 255))
        screen.blit(label, label.get_rect(center=rect.center))


def run(level=1, seed=0, auto_frames=None):
    pygame.init()
    game = Game(level=level, seed=seed)
    w, h = game.grid.width * TILE, game.grid.height * TILE + 60
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f"Adaptive Enemies preview - level {level}, seed {seed}")
    font = pygame.font.SysFont("consolas", TILE - 8, bold=True)
    small = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()

    # ---- HOOK: replay storage (put these in gui.py) ----
    store = ReplayStore(keep=20)
    recorder = RoundRecorder(game.round_number, game.level_obj.seed)
    # ----------------------------------------------------
    player_pos, mode, banner = game.grid.start, "play", ""
    show_explored, show_paths, player = False, True, None
    auto_bot = None
    if auto_frames:                      # headless smoke test: a bot plays, then we replay
        from bots import FixedRouteBot
        auto_bot = FixedRouteBot()
    frame_no = 0

    while True:
        dt = clock.tick(30) / 1000.0
        frame_no += 1
        events = list(pygame.event.get())
        if auto_bot and mode == "play" and frame_no <= auto_frames:
            move = auto_bot(game.grid, player_pos, game.got_key, [e.position for e in game.enemies])
            dx, dy = move[0] - player_pos[0], move[1] - player_pos[1]
            events.append(pygame.event.Event(pygame.KEYDOWN, key={(0, -1): pygame.K_w, (0, 1): pygame.K_s,
                          (-1, 0): pygame.K_a, (1, 0): pygame.K_d, (0, 0): 0}.get((dx, dy), 0)))
        if auto_bot and frame_no == auto_frames + 1:
            if recorder.rec.frames and not banner:   # round still running: close it so there is something to replay
                store.add(recorder.finish("timeout", game.tracker.probability_map()))
            events.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v))
        if auto_bot and frame_no == auto_frames + 2:
            events.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
        if auto_bot and frame_no > auto_frames + 6:
            return {"replay_frames": len(player.rec) if player else 0, "rounds_stored": len(store.rounds)}

        for ev in events:
            if ev.type == pygame.QUIT:
                return
            if ev.type != pygame.KEYDOWN:
                continue
            if ev.key == pygame.K_q:
                return
            if mode == "play":
                if ev.key in KEYS and not banner:
                    dx, dy = KEYS[ev.key]
                    new_pos = (player_pos[0] + dx, player_pos[1] + dy)
                    if game.grid.is_walkable(new_pos):
                        # ---- HOOK: one step + record (put these in gui.py) ----
                        caught, got_key, escaped = game.step(new_pos)
                        recorder.capture(game.turn_count, new_pos, [e.position for e in game.enemies],
                                         [e.state.name for e in game.enemies], caught)
                        # -------------------------------------------------------
                        player_pos = new_pos
                        if caught or escaped:
                            banner = "CAUGHT" if caught else "ESCAPED"
                            store.add(recorder.finish("caught" if caught else "escaped",
                                                      game.tracker.probability_map()))
                elif ev.key == pygame.K_e:
                    show_explored = not show_explored
                    for e in game.enemies:
                        e.record_explored = show_explored
                elif ev.key == pygame.K_p:
                    show_paths = not show_paths
                elif ev.key == pygame.K_r:
                    if recorder.rec.frames and not banner:        # abandon a half-played round, keep it
                        store.add(recorder.finish("abandoned", game.tracker.probability_map()))
                    game.reset_enemies()
                    recorder = RoundRecorder(game.round_number, game.level_obj.seed)
                    player_pos, banner = game.grid.start, ""
                    for e in game.enemies:
                        e.record_explored = show_explored
                elif ev.key == pygame.K_v and store.latest:
                    player, mode = ReplayPlayer(store.latest), "replay"
            else:
                if ev.key == pygame.K_ESCAPE:
                    mode = "play"
                elif ev.key in REPLAY_KEYS:
                    player.handle_key(REPLAY_KEYS[ev.key])

        screen.fill(BG)
        draw_map(screen, game.grid)
        if mode == "play":
            draw_enemy_overlay(screen, enemy_overlay_data(game.enemies), tile=TILE, font=font,
                               show_paths=show_paths, show_explored=show_explored)
            draw_player(screen, player_pos)
            msg = banner + "  (R = new round, V = replay)" if banner else \
                f"round {game.round_number}  turn {game.turn_count}   E explored  P paths  V replay"
            screen.blit(small.render(msg, True, (230, 230, 230)), (6, game.grid.height * TILE + 8))
        else:
            player.update(dt)
            f = player.frame
            if f:
                draw_frame_guards(screen, f, font)
                draw_player(screen, f.player)
            draw_timeline(screen, (10, game.grid.height * TILE + 30, w - 20, 14), player, small)
        pygame.display.flip()


if __name__ == "__main__":
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run(lvl, sd)
