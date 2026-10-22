"""Quick check: generation, reward safety, save/load, and a solved maze."""

from __future__ import annotations

import tempfile
from pathlib import Path

from maze.agent import TrainConfig, greedy_rollout, train
from maze.draw import render_svg
from maze.generate import generate_maze
from maze.model import is_reachable, step
from maze.reward import PRESETS, RewardError, RewardFunction
from maze.store import load, save


def main() -> None:
    maze = generate_maze(8, 8, seed=7, name="demo")
    assert len(maze.passages) == maze.rows * maze.cols - 1
    assert is_reachable(maze)

    opened = generate_maze(8, 8, seed=7, extra_passages=4)
    assert len(opened.passages) == len(maze.passages) + 4
    assert is_reachable(opened)

    blocked = step(maze, maze.start, "N")
    assert blocked.hit_wall and blocked.nxt == maze.start

    for expr in PRESETS.values():
        RewardFunction(expr)
    for bad in ("().__class__", "__import__('os')", "goal ** 2", "open('/tmp')", ""):
        try:
            RewardFunction(bad)
        except RewardError:
            pass
        else:
            raise AssertionError(f"accepted unsafe reward: {bad}")
    try:
        RewardFunction("1 / (step - step)")(step=1, goal=0, hit_wall=0, progress=0)
    except RewardError:
        pass
    else:
        raise AssertionError("division by zero should fail")

    reward = RewardFunction(PRESETS["Coste por paso"])
    result = train(
        maze,
        reward,
        TrainConfig(alpha=0.2, gamma=0.95, epsilon=0.25, episodes=500, max_steps=300, seed=0),
    )
    rollout = greedy_rollout(maze, result.q, reward, max_steps=300)
    if not rollout.reached:
        raise AssertionError(
            f"greedy policy missed the goal after {len(result.returns)} episodes "
            f"(last return {result.returns[-1]:.2f})"
        )

    sparse = RewardFunction(PRESETS["Escasa"])
    small = generate_maze(5, 5, seed=1, name="small")
    sparse_result = train(
        small,
        sparse,
        TrainConfig(alpha=0.4, gamma=0.98, epsilon=0.4, episodes=800, max_steps=200, seed=1),
    )
    sparse_path = greedy_rollout(small, sparse_result.q, sparse, max_steps=200)
    if not sparse_path.reached:
        raise AssertionError("sparse reward did not solve the 5×5 maze")

    svg = render_svg(maze, policy=None, q=result.q)
    assert svg.startswith("<svg") and "Laberinto" in svg

    broken = generate_maze(5, 5, seed=2, name="broken")
    broken.passages.clear()
    assert not is_reachable(broken)

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        try:
            save(broken, reward.expr, "broken", root)
        except ValueError:
            pass
        else:
            raise AssertionError("saved an unsolvable maze")
        path = save(maze, reward.expr, "demo maze", root)
        assert path.name == "demo-maze.yaml"
        loaded, loaded_reward = load(path)
        assert loaded.rows == maze.rows
        assert loaded.passages == maze.passages
        assert loaded.start == maze.start and loaded.goal == maze.goal
        assert loaded_reward == reward.expr
        assert is_reachable(loaded)

    print(
        f"ok  8×8 in {len(rollout.path) - 1} steps  "
        f"return {rollout.total:.2f}  "
        f"5×5 sparse in {len(sparse_path.path) - 1} steps"
    )


if __name__ == "__main__":
    main()
