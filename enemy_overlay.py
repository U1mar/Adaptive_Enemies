"""game/enemy_overlay.py - GUI enemy overlay (task 2.4 + A* explored overlay, week 2).

Data functions are pure (testable without pygame); draw functions import pygame lazily.
Enemy attributes used (game.enemy.Enemy):
    enemy.position ; enemy.state (EnemyState: PATROL, CHASE, FLANK) ; enemy.path (planned A* tiles)
    enemy.last_explored (tiles A* expanded) is filled when enemy.record_explored = True
"""
from __future__ import annotations

STATE_STYLE = {            # prefix -> (letter, RGB)
    "PATROL": ("P", (70, 130, 230)),
    "CHASE":  ("C", (225, 60, 60)),
    "FLANK":  ("F", (240, 160, 40)),
    "AMBUSH": ("F", (240, 160, 40)),
    "WAIT":   ("F", (240, 160, 40)),
}
DEFAULT_STYLE = ("?", (170, 170, 170))


def _pos(e):
    p = getattr(e, "position", None)
    return tuple(p) if p is not None else (e.x, e.y)


def _state_name(e) -> str:
    s = getattr(e, "state", "")
    return str(getattr(s, "name", s)).upper()


def style_for(state_name: str):
    for prefix, style in STATE_STYLE.items():
        if state_name.startswith(prefix):
            return style
    return DEFAULT_STYLE


def enemy_overlay_data(enemies) -> list[dict]:
    out = []
    for i, e in enumerate(enemies):
        name = _state_name(e)
        letter, colour = style_for(name)
        path = [tuple(p) for p in (getattr(e, "path", None) or [])]
        if not path:                       # patrol/flank: the A* leg planned on the last step
            path = [tuple(p) for p in (getattr(e, "path_preview", None) or [])]
        out.append({"id": i, "pos": _pos(e), "state": name, "letter": letter,
                    "colour": colour, "path": path,
                    "explored": [tuple(p) for p in (getattr(e, "last_explored", None) or [])]})
    return out


def draw_enemy_overlay(surface, data, tile: int = 24, origin=(0, 0), font=None,
                       show_paths=True, show_explored=False):
    import pygame
    ox, oy = origin
    font = font or pygame.font.SysFont("consolas", tile - 6, bold=True)
    for d in data:
        r, g, b = d["colour"]
        if show_explored and d["explored"]:
            s = pygame.Surface((tile, tile), pygame.SRCALPHA)
            s.fill((r, g, b, 45))
            for (x, y) in d["explored"]:
                surface.blit(s, (ox + x * tile, oy + y * tile))
        if show_paths and len(d["path"]) > 1:
            pts = [(ox + x * tile + tile // 2, oy + y * tile + tile // 2) for x, y in d["path"]]
            pygame.draw.lines(surface, (r, g, b), False, pts, 2)
    for d in data:                      # markers last so they sit on top of paths
        x, y = d["pos"]
        rect = pygame.Rect(ox + x * tile + 1, oy + y * tile + 1, tile - 2, tile - 2)
        pygame.draw.rect(surface, d["colour"], rect, border_radius=5)
        pygame.draw.rect(surface, (20, 20, 20), rect, 2, border_radius=5)
        label = font.render(d["letter"], True, (255, 255, 255))
        surface.blit(label, label.get_rect(center=rect.center))
