"""Simulated Annealing."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mountain_ridge.swarm.base import (
    Agent, AttractionVector, Position, SearchSpace, Swarm, _make_noisy,
)

_T_MIN: float = 1e-4   # temperature floor; cooling halts here


class SAAgent(Agent):
    """A single SA agent."""

    def __init__(
        self,
        position: NDArray[np.float64],
        search_space: SearchSpace,
        bounds: tuple[int, int],
        t0: float,
        cooling_rate: float,
        step_size: float,
        rng: np.random.Generator,
    ) -> None:
        self._pos = position.copy()
        self._search_space = search_space
        self._lo = np.array([0.0, 0.0])
        self._hi = np.array(
            [float(bounds[0] - 1), float(bounds[1] - 1)]
        )
        self._temperature = t0
        self._cooling_rate = cooling_rate
        self._step_size = step_size
        self._rng = rng
        self._best_pos = position.copy()
        self._best_score: float = search_space(self._as_position())

    def _as_position(self) -> Position:
        return (float(self._pos[0]), float(self._pos[1]))

    @property
    def position(self) -> Position:
        return self._as_position()

    @property
    def score(self) -> float:
        return float(self._search_space(self._as_position()))

    def update(self, shared_memory: dict[str, Any]) -> None:
        # Step 1 — propose a neighbour
        eps = self._rng.uniform(-0.5, 0.5, size=2)
        candidate = self._pos + self._step_size * eps
        np.clip(candidate, self._lo, self._hi, out=candidate)

        # Step 2 — compute change in objective
        current_score = self.score
        candidate_score = float(
            self._search_space(
                (float(candidate[0]), float(candidate[1]))
            )
        )
        delta_f = candidate_score - current_score

        # Step 3 — accept or reject (Metropolis criterion)
        if delta_f < 0:
            self._pos = candidate
        else:
            p_accept = (
                float(np.exp(-delta_f / self._temperature))
                if self._temperature > 0 else 0.0
            )
            if self._rng.uniform() < p_accept:
                self._pos = candidate

        # Step 4 — cool temperature
        self._temperature = max(
            _T_MIN, self._cooling_rate * self._temperature
        )

        # Step 5 — update personal best
        new_score = self.score
        if new_score < self._best_score:
            self._best_score = new_score
            self._best_pos = self._pos.copy()


class SASwarm(Swarm):
    """Simulated Annealing swarm.

    Agents are fully independent — shared_memory is always empty.
    """

    def __init__(
        self,
        n_agents: int,
        search_space: SearchSpace,
        dimensions: tuple[int, int],
        seed: int,
        t0: float,
        cooling_rate: float = 0.95,
        step_size: float | None = None,
        noise: float = 0.0,
    ) -> None:
        w_dim, h_dim = dimensions
        resolved_step = (
            step_size if step_size is not None
            else 0.1 * min(w_dim, h_dim)
        )

        rng = np.random.default_rng(seed)
        positions = rng.uniform(
            [0.0, 0.0],
            [float(w_dim - 1), float(h_dim - 1)],
            size=(n_agents, 2),
        )
        agent_rngs = rng.spawn(n_agents)

        self._agents = [
            SAAgent(
                position=positions[i],
                search_space=(
                    _make_noisy(search_space, noise, agent_rngs[i])
                    if noise > 0 else search_space
                ),
                bounds=dimensions,
                t0=t0,
                cooling_rate=cooling_rate,
                step_size=resolved_step,
                rng=agent_rngs[i],
            )
            for i in range(n_agents)
        ]

    def get_positions(self) -> list[Position]:
        return [a.position for a in self._agents]

    def get_scores(self) -> list[float]:
        return [a.score for a in self._agents]

    def get_best_position(self) -> Position:
        best = min(self._agents, key=lambda a: a._best_score)
        return (float(best._best_pos[0]), float(best._best_pos[1]))

    def update(self) -> None:
        shared: dict[str, Any] = {}
        for agent in self._agents:
            agent.update(shared)

    def get_attractions(self) -> list[list[AttractionVector]]:
        return [[] for _ in self._agents]
