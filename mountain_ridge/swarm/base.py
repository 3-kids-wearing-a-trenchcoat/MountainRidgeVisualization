"""Abstract base classes for swarm agents and swarm optimizers."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


Position = tuple[float, ...]
SearchSpace = Callable[[Position], float]


class Agent(ABC):
    """A single member of a swarm."""

    @property
    @abstractmethod
    def position(self) -> Position:
        """Return the agent's current position."""

    @property
    @abstractmethod
    def score(self) -> float:
        """Return the objective score at the current position."""

    @abstractmethod
    def update(self, shared_memory: dict[str, Any]) -> None:
        """Advance the agent by one iteration.

        Reads shared_memory but must not write to it.
        """


class Swarm(ABC):
    """A collection of agents under a shared optimization rule."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return the current positions of all agents."""

    @abstractmethod
    def get_scores(self) -> list[float]:
        """Return the current objective score of every agent."""

    @abstractmethod
    def get_best_position(self) -> Position:
        """Return the position with the lowest objective score."""

    @abstractmethod
    def update(self) -> None:
        """Advance the simulation by one iteration.

        All agents update their state, then shared memory is updated.
        """
