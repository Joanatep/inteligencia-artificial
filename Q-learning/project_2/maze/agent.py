"""Tabular Watkins Q-learning on a maze.

The update is the same one as Q-learning/project (`qlearn/agent.py`):

    Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]

with ε-greedy behaviour. While training, ties for the best action are broken
at random so an all-zero table does not always walk north. The greedy policy
that the UI draws breaks ties in action order N, E, S, W, as in project 1.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from maze.model import ACTIONS, Cell, Maze, manhattan, step
from maze.reward import RewardFunction

QTable = dict[Cell, dict[str, float]]


@dataclass
class TrainConfig:
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon: float = 0.25
    episodes: int = 500
    max_steps: int = 300
    seed: int = 0


@dataclass
class TrainResult:
    q: QTable
    returns: list[float] = field(default_factory=list)
    stopped: bool = False


def empty_q(maze: Maze) -> QTable:
    table: QTable = {}
    for r in range(maze.rows):
        for c in range(maze.cols):
            cell = (r, c)
            if cell == maze.goal:
                continue
            table[cell] = {a: 0.0 for a in ACTIONS}
    return table


def argmax_action(row: dict[str, float], rng: random.Random | None = None) -> str:
    best = max(row[a] for a in ACTIONS)
    tied = [a for a in ACTIONS if row[a] == best]
    if rng is None:
        return tied[0]
    return rng.choice(tied)


def epsilon_greedy(
    row: dict[str, float],
    epsilon: float,
    rng: random.Random,
) -> str:
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    return argmax_action(row, rng)


def _reward_of(maze: Maze, reward: RewardFunction, pos: Cell, nxt: Cell, hit_wall: bool, reached: bool) -> float:
    return reward(
        step=1.0,
        goal=1.0 if reached else 0.0,
        hit_wall=1.0 if hit_wall else 0.0,
        progress=float(manhattan(pos, maze.goal) - manhattan(nxt, maze.goal)),
    )


def train(
    maze: Maze,
    reward: RewardFunction,
    cfg: TrainConfig,
    on_batch: Callable[[int, list[float]], None] | None = None,
    batch: int = 25,
    should_stop: Callable[[], bool] | None = None,
) -> TrainResult:
    if not (0.0 <= cfg.epsilon <= 1.0):
        raise ValueError("epsilon must be between 0 and 1")
    if cfg.episodes < 1:
        raise ValueError("episodes must be at least 1")
    if cfg.max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    rng = random.Random(cfg.seed)
    q = empty_q(maze)
    returns: list[float] = []
    every = max(1, batch)
    stopped = False

    for episode in range(cfg.episodes):
        if should_stop and should_stop():
            stopped = True
            break
        state = maze.start
        total = 0.0
        for _ in range(cfg.max_steps):
            if state not in q:
                break
            action = epsilon_greedy(q[state], cfg.epsilon, rng)
            moved = step(maze, state, action)
            gained = _reward_of(maze, reward, state, moved.nxt, moved.hit_wall, moved.reached_goal)
            future = 0.0 if moved.reached_goal else max(q[moved.nxt].values())
            target = gained + cfg.gamma * future
            old = q[state][action]
            q[state][action] = old + cfg.alpha * (target - old)
            total += gained
            state = moved.nxt
            if moved.reached_goal:
                break
        returns.append(total)
        done = episode + 1
        if on_batch and (done % every == 0 or done == cfg.episodes):
            on_batch(done, list(returns))

    return TrainResult(q=q, returns=returns, stopped=stopped)


@dataclass
class Rollout:
    path: list[Cell]
    total: float
    reached: bool


def greedy_rollout(
    maze: Maze,
    q: QTable,
    reward: RewardFunction,
    max_steps: int,
) -> Rollout:
    """Follow the greedy policy from the start. Ties use action order."""
    state = maze.start
    path = [state]
    seen = {state}
    total = 0.0
    reached = state == maze.goal
    for _ in range(max_steps):
        if state not in q:
            break
        action = argmax_action(q[state])
        moved = step(maze, state, action)
        total += _reward_of(maze, reward, state, moved.nxt, moved.hit_wall, moved.reached_goal)
        state = moved.nxt
        path.append(state)
        if moved.reached_goal:
            reached = True
            break
        if state in seen:
            break
        seen.add(state)
    return Rollout(path=path, total=total, reached=reached)


def greedy_policy(maze: Maze, q: QTable) -> dict[Cell, str]:
    return {cell: argmax_action(row) for cell, row in q.items() if cell != maze.goal}
