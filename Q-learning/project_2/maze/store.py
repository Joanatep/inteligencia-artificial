"""Save and load a maze plus its reward expression as YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from maze.model import ACTIONS, Cell, Maze, edge, in_bounds, is_reachable, validate_ends
from maze.reward import DEFAULT_REWARD

MAZES_DIR = Path(__file__).resolve().parent.parent / "mazes"


class _Flow(list):
    """List written in YAML flow style, so a cell is `[row, col]`."""


def _flow_cell(cell: Cell) -> _Flow:
    return _Flow([int(cell[0]), int(cell[1])])


def _represent_flow(dumper: yaml.SafeDumper, data: _Flow):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(_Flow, _represent_flow, Dumper=yaml.SafeDumper)


def safe_stem(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name.strip())
    cleaned = cleaned.strip("-_")
    return (cleaned or "laberinto")[:80]


def maze_path(stem: str, directory: Path | None = None) -> Path:
    root = (directory or MAZES_DIR).resolve()
    path = (root / f"{safe_stem(stem)}.yaml").resolve()
    if path.parent != root:
        raise ValueError("invalid maze name")
    return path


def to_dict(maze: Maze, reward: str) -> dict[str, Any]:
    passages = sorted(maze.passages)
    return {
        "name": maze.name,
        "rows": maze.rows,
        "cols": maze.cols,
        "start": _flow_cell(maze.start),
        "goal": _flow_cell(maze.goal),
        "seed": maze.seed,
        "algorithm": maze.algorithm,
        "extra_passages": maze.extra_passages,
        "reward": reward.strip(),
        "passages": [_Flow([_flow_cell(a), _flow_cell(b)]) for a, b in passages],
    }


def dumps(maze: Maze, reward: str) -> str:
    return yaml.safe_dump(
        to_dict(maze, reward),
        allow_unicode=True,
        sort_keys=False,
    )


def save(maze: Maze, reward: str, stem: str, directory: Path | None = None) -> Path:
    if not is_reachable(maze):
        raise ValueError("no hay camino de la salida a la meta")
    path = maze_path(stem, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(maze, reward), encoding="utf-8")
    return path


def _cell(value: Any, label: str) -> Cell:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be [row, col]")
    return (int(value[0]), int(value[1]))


def parse(raw: dict[str, Any], source: Path | None = None) -> tuple[Maze, str]:
    if not isinstance(raw, dict):
        raise ValueError("maze file must be a mapping")
    rows = int(raw.get("rows") or 0)
    cols = int(raw.get("cols") or 0)
    if rows < 2 or cols < 2:
        raise ValueError("rows and cols must be at least 2")
    start = _cell(raw.get("start") or [0, 0], "start")
    goal = _cell(raw.get("goal") or [rows - 1, cols - 1], "goal")
    validate_ends(rows, cols, start, goal)

    passages: set[tuple[Cell, Cell]] = set()
    for item in raw.get("passages") or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("each passage must be a pair of cells")
        a = _cell(item[0], "passage")
        b = _cell(item[1], "passage")
        if not in_bounds(rows, cols, a) or not in_bounds(rows, cols, b):
            raise ValueError(f"passage {a}–{b} leaves the grid")
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        if (dr, dc) not in ((0, 1), (1, 0)):
            raise ValueError(f"passage {a}–{b} is not an orthogonal step")
        passages.add(edge(a, b))

    # A stored action list, if present, must stay the four compass moves.
    actions = raw.get("actions")
    if actions is not None and tuple(str(a) for a in actions) != ACTIONS:
        raise ValueError(f"actions must be {list(ACTIONS)}")

    name = str(raw.get("name") or (source.stem if source else "maze"))
    reward = str(raw.get("reward") or DEFAULT_REWARD)
    maze = Maze(
        name=name,
        rows=rows,
        cols=cols,
        start=start,
        goal=goal,
        passages=passages,
        seed=int(raw.get("seed") or 0),
        algorithm=str(raw.get("algorithm") or "recursive_backtracker"),
        extra_passages=int(raw.get("extra_passages") or 0),
    )
    return maze, reward


def load(path: str | Path) -> tuple[Maze, str]:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return parse(raw, source=path)


def list_stems(directory: Path | None = None) -> list[str]:
    root = directory or MAZES_DIR
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.yaml"))
