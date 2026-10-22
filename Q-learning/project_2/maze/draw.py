"""SVG drawing of a maze, the greedy arrows, and the agent."""

from __future__ import annotations

from maze.model import ACTIONS, Cell, Maze, edge

_OFFSETS = {"N": (0.0, -1.0), "E": (1.0, 0.0), "S": (0.0, 1.0), "W": (-1.0, 0.0)}


def render_svg(
    maze: Maze,
    agent: Cell | None = None,
    policy: dict[Cell, str] | None = None,
    trail: list[Cell] | None = None,
    q: dict[Cell, dict[str, float]] | None = None,
) -> str:
    """Return an SVG string. `trail` is the greedy path revealed so far."""
    side = max(maze.rows, maze.cols)
    cell = max(22, min(46, 580 // side))
    pad = 16
    width = maze.cols * cell + pad * 2
    height = maze.rows * cell + pad * 2
    agent = maze.start if agent is None else agent
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Laberinto">',
        "<title>Laberinto</title>",
        f'<rect width="{width}" height="{height}" rx="16" fill="#0f172a"/>',
        '<defs><marker id="ah" markerWidth="7" markerHeight="7" refX="5.5" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="#fbbf24"/></marker></defs>',
    ]

    for r in range(maze.rows):
        for c in range(maze.cols):
            x = pad + c * cell
            y = pad + r * cell
            here = (r, c)
            if here == maze.goal:
                fill = "#9a3412"
            elif here == maze.start:
                fill = "#166534"
            else:
                fill = "#162033"
            title = _title(here, q)
            parts.append(f"<g><title>{title}</title>")
            inset = 1.2
            parts.append(
                f'<rect x="{x + inset:.1f}" y="{y + inset:.1f}" '
                f'width="{cell - 2 * inset:.1f}" height="{cell - 2 * inset:.1f}" '
                f'rx="4" fill="{fill}"/>'
            )
            parts.append("</g>")

    parts.extend(_walls(maze, cell, pad))
    if trail and len(trail) > 1:
        points = " ".join(
            f"{pad + c * cell + cell / 2:.1f},{pad + r * cell + cell / 2:.1f}" for r, c in trail
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#38bdf8" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round" opacity="0.7"/>'
        )
    if policy:
        parts.extend(_arrows(maze, policy, agent, cell, pad))
    parts.extend(_letters(maze, agent, cell, pad))
    ax = pad + agent[1] * cell + cell / 2
    ay = pad + agent[0] * cell + cell / 2
    radius = cell * 0.2
    parts.append(
        f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="{radius:.1f}" '
        'fill="#38bdf8" stroke="#e0f2fe" stroke-width="2"/>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _title(cell: Cell, q: dict[Cell, dict[str, float]] | None) -> str:
    if not q or cell not in q:
        return f"{cell[0]},{cell[1]}"
    bits = "  ".join(f"{action} {q[cell][action]:.2f}" for action in ACTIONS)
    return f"{cell[0]},{cell[1]}  {bits}"


def _walls(maze: Maze, cell: int, pad: int) -> list[str]:
    lines: list[str] = []

    def segment(x1: float, y1: float, x2: float, y2: float) -> None:
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#e2e8f0" stroke-width="3.5" stroke-linecap="square"/>'
        )

    for r in range(maze.rows):
        for c in range(maze.cols):
            x = pad + c * cell
            y = pad + r * cell
            here = (r, c)
            if c == 0:
                segment(x, y, x, y + cell)
            if r == 0:
                segment(x, y, x + cell, y)
            east = (r, c + 1)
            if c == maze.cols - 1 or edge(here, east) not in maze.passages:
                segment(x + cell, y, x + cell, y + cell)
            south = (r + 1, c)
            if r == maze.rows - 1 or edge(here, south) not in maze.passages:
                segment(x, y + cell, x + cell, y + cell)
    return lines


def _arrows(
    maze: Maze,
    policy: dict[Cell, str],
    agent: Cell,
    cell: int,
    pad: int,
) -> list[str]:
    lines: list[str] = []
    span = cell * 0.26
    for here, action in policy.items():
        if here == agent or here == maze.goal or action not in _OFFSETS:
            continue
        dx, dy = _OFFSETS[action]
        cx = pad + here[1] * cell + cell / 2
        cy = pad + here[0] * cell + cell / 2
        x1 = cx - dx * span * 0.15
        y1 = cy - dy * span * 0.15
        x2 = cx + dx * span
        y2 = cy + dy * span
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#fbbf24" stroke-width="1.6" marker-end="url(#ah)"/>'
        )
    return lines


def _letters(maze: Maze, agent: Cell, cell: int, pad: int) -> list[str]:
    texts: list[str] = []
    size = cell * 0.34
    for here, letter in ((maze.start, "S"), (maze.goal, "G")):
        if here == agent:
            continue
        cx = pad + here[1] * cell + cell / 2
        cy = pad + here[0] * cell + cell / 2
        texts.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" dominant-baseline="central" '
            f'fill="#f8fafc" font-size="{size:.1f}" font-family="ui-sans-serif,system-ui,sans-serif" '
            f'font-weight="600">{letter}</text>'
        )
    return texts
