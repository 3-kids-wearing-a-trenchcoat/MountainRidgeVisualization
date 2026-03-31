"""Firefly Algorithm."""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mountain_ridge.swarm.base import (
    Agent, AttractionVector, Position, SearchSpace, Swarm,
)

_SNAPSHOT_KEY = "snapshot"


class FAAgent(Agent):
    """A single FA firefly."""

    def __init__(
        self,
        position: NDArray[np.float64],
        search_space: SearchSpace,
        bounds: tuple[int, int],
        beta0: float,
        gamma: float,
        alpha: float,
        variant: str,
        levy_exp: float,
        rng: np.random.Generator,
    ) -> None:
        self._pos = position.copy()
        self._search_space = search_space
        self._lo = np.array([0.0, 0.0])
        self._hi = np.array(
            [float(bounds[0] - 1), float(bounds[1] - 1)]
        )
        self._beta0 = beta0
        self._gamma = gamma
        self._alpha = alpha
        self._variant = variant
        self._levy_exp = levy_exp
        self._rng = rng
        self._last_attractions: list[tuple[Position, float]] = []

        if variant == "levy":
            lam = levy_exp
            num = (
                math.gamma(1 + lam) * math.sin(math.pi * lam / 2)
            )
            den = (
                math.gamma((1 + lam) / 2)
                * lam
                * 2 ** ((lam - 1) / 2)
            )
            self._sigma_u: float = (num / den) ** (1 / lam)

    def _as_position(self) -> Position:
        return (float(self._pos[0]), float(self._pos[1]))

    @property
    def position(self) -> Position:
        return self._as_position()

    @property
    def score(self) -> float:
        return float(self._search_space(self._as_position()))

    def _levy_step(self) -> NDArray[np.float64]:
        """Generate a 2-D Lévy step via Mantegna's algorithm.

        Each dimension gets an independent random sign so the
        distribution is symmetric around zero (sign(rand-0.5) ⊕ L).
        """
        u = self._rng.normal(0.0, self._sigma_u, size=2)
        v = self._rng.standard_normal(size=2)
        L = u / np.abs(v) ** (1.0 / self._levy_exp)
        s = np.sign(self._rng.uniform(size=2) - 0.5)
        return s * L

    def update(self, shared_memory: dict[str, Any]) -> None:
        snapshot: list[tuple[Position, float]] = (
            shared_memory[_SNAPSHOT_KEY]
        )
        my_score = self.score

        # For each brighter firefly: move toward it and add one random
        # walk step immediately (once per attractor, not once per
        # iteration).  Position is updated after each step so later
        # attractions use the already-moved location.
        self._last_attractions = []
        moved = False
        for pos_j, score_j in snapshot:
            if score_j < my_score:
                diff = np.array(pos_j, dtype=np.float64) - self._pos
                r_sq = float(np.dot(diff, diff))
                beta = self._beta0 * math.exp(-self._gamma * r_sq)
                self._last_attractions.append((pos_j, beta))
                self._pos += beta * diff
                if self._variant == "levy":
                    eps = self._levy_step()
                else:
                    eps = self._rng.uniform(-0.5, 0.5, size=2)
                self._pos += self._alpha * eps
                moved = True

        # Clamp once after all steps; the brightest firefly never moves.
        if moved:
            np.clip(self._pos, self._lo, self._hi, out=self._pos)


class FASwarm(Swarm):
    """Firefly Algorithm swarm."""

    def __init__(
        self,
        n_agents: int,
        search_space: SearchSpace,
        dimensions: tuple[int, int],
        seed: int,
        beta0: float = 1.0,
        gamma: float | None = None,
        alpha: float | None = None,
        variant: str = "brownian",
        levy_exp: float = 1.5,
    ) -> None:
        w_dim, h_dim = dimensions
        min_dim = min(w_dim, h_dim)
        resolved_gamma = (
            gamma if gamma is not None else 1.0 / (min_dim ** 2)
        )
        resolved_alpha = (
            alpha if alpha is not None else 0.05 * min_dim
        )

        rng = np.random.default_rng(seed)
        positions = rng.uniform(
            [0.0, 0.0],
            [float(w_dim - 1), float(h_dim - 1)],
            size=(n_agents, 2),
        )
        agent_rngs = rng.spawn(n_agents)

        self._agents = [
            FAAgent(
                position=positions[i],
                search_space=search_space,
                bounds=dimensions,
                beta0=beta0,
                gamma=resolved_gamma,
                alpha=resolved_alpha,
                variant=variant,
                levy_exp=levy_exp,
                rng=agent_rngs[i],
            )
            for i in range(n_agents)
        ]

        self._shared_memory: dict[str, Any] = {
            _SNAPSHOT_KEY: [
                (a.position, a.score) for a in self._agents
            ],
        }

    def get_positions(self) -> list[Position]:
        return [a.position for a in self._agents]

    def get_scores(self) -> list[float]:
        return [a.score for a in self._agents]

    def get_best_position(self) -> Position:
        snapshot: list[tuple[Position, float]] = (
            self._shared_memory[_SNAPSHOT_KEY]
        )
        return min(snapshot, key=lambda entry: entry[1])[0]

    def update(self) -> None:
        for agent in self._agents:
            agent.update(self._shared_memory)

        self._shared_memory[_SNAPSHOT_KEY] = [
            (a.position, a.score) for a in self._agents
        ]

    def get_attractions(self) -> list[list[AttractionVector]]:
        # Min-max stretch: map beta from [beta_min, beta0] → [0, 1].
        # beta_min is the smallest possible beta, reached at the maximum
        # distance (the grid diagonal).  This fills the full [0, 1] range
        # instead of compressing everything into a narrow band near 1.0
        # (which happens because default gamma keeps attraction nearly flat
        # across the whole grid).
        result: list[list[AttractionVector]] = []
        for agent in self._agents:
            diag_sq = float(np.dot(agent._hi, agent._hi))
            beta_min = agent._beta0 * math.exp(-agent._gamma * diag_sq)
            beta_span = agent._beta0 - beta_min
            vectors: list[AttractionVector] = []
            for target, beta in agent._last_attractions:
                if beta_span > 0:
                    weight = (beta - beta_min) / beta_span
                else:
                    weight = 1.0
                weight = max(0.0, min(1.0, weight))
                if weight >= 0.01:
                    vectors.append(AttractionVector(
                        target=target,
                        weight=weight,
                        kind="firefly",
                    ))
            result.append(vectors)
        return result
