"""Random perfect mazes, plus optional extra openings."""

from __future__ import annotations

import random

from maze.model import DELTA, Cell, Maze, edge, is_reachable, validate_ends

MIN_SIDE = 5
MAX_SIDE = 25


def generate_maze(
    rows: int,
    cols: int,
    seed: int,
    extra_passages: int = 0,
    start: Cell = (0, 0),
    goal: Cell | None = None,
    name: str = "maze",
) -> Maze:
    """Carve a maze with recursive backtracking, then open `extra_passages` walls.

    The result is always connected, so a path from start to goal exists.
    """
    rows, cols = int(rows), int(cols)
    if not (MIN_SIDE <= rows <= MAX_SIDE and MIN_SIDE <= cols <= MAX_SIDE):
        raise ValueError(f"rows and cols must be between {MIN_SIDE} and {MAX_SIDE}")
    goal = (rows - 1, cols - 1) if goal is None else (int(goal[0]), int(goal[1]))
    start = (int(start[0]), int(start[1]))
    validate_ends(rows, cols, start, goal)

    rng = random.Random(int(seed))
    passages: set[tuple[Cell, Cell]] = set()
    visited = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        r, c = stack[-1]
        options: list[Cell] = []
        for dr, dc in DELTA.values():
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and nxt not in visited:
                options.append(nxt)
        if not options:
            stack.pop()
            continue
        nxt = rng.choice(options)
        passages.add(edge((r, c), nxt))
        visited.add(nxt)
        stack.append(nxt)

    if len(visited) != rows * cols:
        raise RuntimeError("backtracker did not visit every cell")

    closed: list[tuple[Cell, Cell]] = []
    for r in range(rows):
        for c in range(cols):
            for nxt in ((r, c + 1), (r + 1, c)):
                if nxt[0] >= rows or nxt[1] >= cols:
                    continue
                passage = edge((r, c), nxt)
                if passage not in passages:
                    closed.append(passage)
    rng.shuffle(closed)
    opened = 0
    for passage in closed:
        if opened >= max(0, int(extra_passages)):
            break
        passages.add(passage)
        opened += 1

    maze = Maze(
        name=name,
        rows=rows,
        cols=cols,
        start=start,
        goal=goal,
        passages=passages,
        seed=int(seed),
        algorithm="recursive_backtracker",
        extra_passages=opened,
    )
    if not is_reachable(maze):
        raise RuntimeError("generated maze has no path from start to goal")
    return maze
