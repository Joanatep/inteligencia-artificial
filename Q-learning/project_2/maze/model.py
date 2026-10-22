"""Grid maze: cells, passages, and one action step."""

from __future__ import annotations

from dataclasses import dataclass, field

ACTIONS: tuple[str, ...] = ("N", "E", "S", "W")
DELTA: dict[str, tuple[int, int]] = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}

Cell = tuple[int, int]
Edge = tuple[Cell, Cell]


def edge(a: Cell, b: Cell) -> Edge:
    """Undirected passage, ordered so (a, b) is stable."""
    return (a, b) if a <= b else (b, a)


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(rows: int, cols: int, cell: Cell) -> bool:
    r, c = cell
    return 0 <= r < rows and 0 <= c < cols


@dataclass
class StepResult:
    nxt: Cell
    hit_wall: bool
    reached_goal: bool


@dataclass
class Maze:
    name: str
    rows: int
    cols: int
    start: Cell
    goal: Cell
    passages: set[Edge] = field(default_factory=set)
    seed: int = 0
    algorithm: str = "recursive_backtracker"
    extra_passages: int = 0

    def neighbors(self, cell: Cell) -> list[Cell]:
        found: list[Cell] = []
        r, c = cell
        for dr, dc in DELTA.values():
            nxt = (r + dr, c + dc)
            if in_bounds(self.rows, self.cols, nxt) and edge(cell, nxt) in self.passages:
                found.append(nxt)
        return found


def step(maze: Maze, pos: Cell, action: str) -> StepResult:
    """Move one cell. A wall or the border leaves the agent in place."""
    if action not in DELTA:
        raise ValueError(f"unknown action {action!r}")
    dr, dc = DELTA[action]
    nxt = (pos[0] + dr, pos[1] + dc)
    if not in_bounds(maze.rows, maze.cols, nxt) or edge(pos, nxt) not in maze.passages:
        return StepResult(pos, True, False)
    return StepResult(nxt, False, nxt == maze.goal)


def is_reachable(maze: Maze) -> bool:
    """Breadth-first search from the start. True when the goal is reachable."""
    if maze.start == maze.goal:
        return True
    seen = {maze.start}
    queue = [maze.start]
    while queue:
        cur = queue.pop(0)
        for nxt in maze.neighbors(cur):
            if nxt in seen:
                continue
            if nxt == maze.goal:
                return True
            seen.add(nxt)
            queue.append(nxt)
    return False


def validate_ends(rows: int, cols: int, start: Cell, goal: Cell) -> None:
    if not in_bounds(rows, cols, start):
        raise ValueError(f"start {start} is outside the {rows}×{cols} grid")
    if not in_bounds(rows, cols, goal):
        raise ValueError(f"goal {goal} is outside the {rows}×{cols} grid")
    if start == goal:
        raise ValueError("start and goal must be different cells")
