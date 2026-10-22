"""Maze world and tabular Q-learning for project_2."""

from maze.agent import TrainConfig, train
from maze.generate import generate_maze
from maze.model import Maze
from maze.reward import PRESETS, RewardFunction

__all__ = [
    "Maze",
    "PRESETS",
    "RewardFunction",
    "TrainConfig",
    "generate_maze",
    "train",
]
