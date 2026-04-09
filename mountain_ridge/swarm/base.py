"""Abstract base classes for swarm agents and swarm optimizers."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

Position = tuple[float, ...]
SearchSpace = Callable[[Position], float]


def _make_noisy(
    space_fn: SearchSpace,
    amplitude: float,
    rng: np.random.Generator,
) -> SearchSpace:
    """Wrap *space_fn* so every evaluation adds uniform noise.

    The noise is sampled independently on each call from
    Uniform(-amplitude, amplitude) using *rng*.
    """
    def noisy(pos: Position) -> float:
        return float(space_fn(pos)) + rng.uniform(-amplitude, amplitude)
    return noisy


@dataclass(frozen=True)
class AttractionVector:
    """An attraction from an agent toward one of its influence sources."""

    target: Position  # position of the attractor
    weight: float     # influence strength, normalised to [0, 1]
    kind: str         # "pbest" | "gbest" | "firefly"


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

    def get_attractions(self) -> list[list[AttractionVector]]:
        """Return per-agent attraction vectors for the current state.

        Returns an empty inner list per agent by default; override in
        subclasses that support attraction-arrow visualisation.
        """
        return [[] for _ in self.get_positions()]
