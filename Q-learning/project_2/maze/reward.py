"""Reward expressions: arithmetic over step, goal, hit_wall, and progress."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

NAMES = ("step", "goal", "hit_wall", "progress")
_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div)


PRESETS: dict[str, str] = {
    "Escasa": "10 * goal",
    "Coste por paso": "-1 + 10 * goal",
    "Penalización de muro": "-1 + 10 * goal - 5 * hit_wall",
    "Moldeada": "-0.1 + 10 * goal + progress",
}

DEFAULT_REWARD = PRESETS["Coste por paso"]


class RewardError(ValueError):
    """The expression is empty, unsafe, or not arithmetic."""


def _check(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _check(node.body)
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, _BIN):
        _check(node.left)
        _check(node.right)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _check(node.operand)
        return
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in NAMES:
        return
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return
    raise RewardError(
        "solo se permiten números, step, goal, hit_wall, progress y + - * / ()"
    )


def compile_reward(expr: str) -> ast.Expression:
    text = expr.strip()
    if not text:
        raise RewardError("la función de recompensa está vacía")
    if len(text) > 200:
        raise RewardError("la expresión es demasiado larga")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise RewardError(f"expresión inválida: {exc.msg}") from exc
    _check(tree)
    return tree


@dataclass
class RewardFunction:
    """Callable reward. `expr` is the source the UI and the YAML file share."""

    expr: str
    _code: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        tree = compile_reward(self.expr)
        self.expr = self.expr.strip()
        self._code = compile(tree, "<reward>", "eval")

    def __call__(
        self,
        *,
        step: float,
        goal: float,
        hit_wall: float,
        progress: float,
    ) -> float:
        env = {
            "step": float(step),
            "goal": float(goal),
            "hit_wall": float(hit_wall),
            "progress": float(progress),
        }
        try:
            value = eval(self._code, {"__builtins__": {}}, env)  # noqa: S307
        except ZeroDivisionError as exc:
            raise RewardError("división entre cero en la recompensa") from exc
        return float(value)


def preview(expr: str) -> str:
    """One line: reward for a normal step, a wall bump, and entering the goal."""
    fn = RewardFunction(expr)
    normal = fn(step=1, goal=0, hit_wall=0, progress=0)
    wall = fn(step=1, goal=0, hit_wall=1, progress=0)
    arrived = fn(step=1, goal=1, hit_wall=0, progress=1)
    return f"Paso {normal:g} · muro {wall:g} · meta {arrived:g}"
