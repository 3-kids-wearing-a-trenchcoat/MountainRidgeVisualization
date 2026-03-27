"""Particle Swarm Optimization."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mountain_ridge.swarm.base import Agent, Position, SearchSpace, Swarm

_G_KEY = "g"


class PSOAgent(Agent):
    """A single PSO particle."""

    def __init__(
        self,
        position: NDArray[np.float64],
        search_space: SearchSpace,
        bounds: tuple[int, int],
        c1: float,
        c2: float,
        w: float,
        v_max: float,
        rng: np.random.Generator,
    ) -> None:
        self._pos = position.copy()
        self._vel = np.zeros(2, dtype=np.float64)
        self._best_pos = position.copy()
        self._best_score: float = search_space(self._as_position())
        self._search_space = search_space
        self._lo = np.array([0.0, 0.0])
        self._hi = np.array(
            [float(bounds[0] - 1), float(bounds[1] - 1)]
        )
        self._c1 = c1
        self._c2 = c2
        self._w = w
        self._v_max = v_max
        self._rng = rng

    def _as_position(self) -> Position:
        return (float(self._pos[0]), float(self._pos[1]))

    @property
    def position(self) -> Position:
        return self._as_position()

    @property
    def score(self) -> float:
        return float(self._search_space(self._as_position()))

    @property
    def personal_best_position(self) -> Position:
        return (float(self._best_pos[0]), float(self._best_pos[1]))

    @property
    def personal_best_score(self) -> float:
        return self._best_score

    def update(self, shared_memory: dict[str, Any]) -> None:
        g: NDArray[np.float64] = shared_memory[_G_KEY]
        r1 = self._rng.uniform(0.0, 1.0, size=2)
        r2 = self._rng.uniform(0.0, 1.0, size=2)

        # Step 2: velocity update
        self._vel = (
            self._w * self._vel
            + self._c1 * r1 * (self._best_pos - self._pos)
            + self._c2 * r2 * (g - self._pos)
        )

        # Step 3: velocity clamping
        np.clip(self._vel, -self._v_max, self._v_max, out=self._vel)

        # Step 4: position update + boundary handling
        self._pos += self._vel
        oob = (self._pos < self._lo) | (self._pos > self._hi)
        self._vel[oob] = 0.0
        np.clip(self._pos, self._lo, self._hi, out=self._pos)

        # Step 5: personal best update
        new_score = self.score
        if new_score < self._best_score:
            self._best_score = new_score
            self._best_pos = self._pos.copy()


class PSOSwarm(Swarm):
    """PSO swarm optimizer."""

    def __init__(
        self,
        n_agents: int,
        search_space: SearchSpace,
        dimensions: tuple[int, int],
        seed: int,
        c1: float = 1.5,
        c2: float = 1.5,
        w: float = 0.0,
        v_max: float | None = None,
    ) -> None:
        w_dim, h_dim = dimensions
        if v_max is None:
            v_max = min(w_dim, h_dim) * 0.2

        rng = np.random.default_rng(seed)
        positions = rng.uniform(
            [0.0, 0.0],
            [float(w_dim - 1), float(h_dim - 1)],
            size=(n_agents, 2),
        )
        agent_rngs = rng.spawn(n_agents)

        self._agents = [
            PSOAgent(
                position=positions[i],
                search_space=search_space,
                bounds=dimensions,
                c1=c1,
                c2=c2,
                w=w,
                v_max=v_max,
                rng=agent_rngs[i],
            )
            for i in range(n_agents)
        ]

        best = min(self._agents, key=lambda a: a.personal_best_score)
        self._shared_memory: dict[str, Any] = {
            _G_KEY: np.array(best.personal_best_position),
        }

    def get_positions(self) -> list[Position]:
        return [a.position for a in self._agents]

    def get_best_position(self) -> Position:
        g = self._shared_memory[_G_KEY]
        return (float(g[0]), float(g[1]))

    def update(self) -> None:
        for agent in self._agents:
            agent.update(self._shared_memory)

        # Step 6: update global best after all agents have stepped
        best = min(self._agents, key=lambda a: a.personal_best_score)
        self._shared_memory[_G_KEY] = np.array(
            best.personal_best_position
        )
