"""Steepest Descent with optional random perturbation."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mountain_ridge.swarm.base import (
    Agent, AttractionVector, Position, SearchSpace, Swarm, _make_noisy,
)

# --- Fixed implementation constants (not user-facing) ---
_DELTA: float = 1.0       # finite-difference step (grid units)
_TAU: float = 0.5         # backtracking reduction factor
_STEP_MIN: float = 1e-4   # descent move cancelled below this step size
_GRAD_TOL: float = 1e-6   # gradient norm below this → local minimum


class SDAgent(Agent):
    """A single steepest-descent agent."""

    def __init__(
        self,
        position: NDArray[np.float64],
        search_space: SearchSpace,
        bounds: tuple[int, int],
        step: float,
        alpha: float,
        rng: np.random.Generator,
    ) -> None:
        self._pos = position.copy()
        self._search_space = search_space
        self._lo = np.array([0.0, 0.0])
        self._hi = np.array(
            [float(bounds[0] - 1), float(bounds[1] - 1)]
        )
        self._step = step
        self._alpha = alpha
        self._rng = rng
        self._best_pos = position.copy()
        self._best_score: float = search_space(self._as_position())
        # Gradient stored from the most recent update(); zero before first call.
        self._last_grad: NDArray[np.float64] = np.zeros(2, dtype=np.float64)

    def _as_position(self) -> Position:
        return (float(self._pos[0]), float(self._pos[1]))

    @property
    def position(self) -> Position:
        return self._as_position()

    @property
    def score(self) -> float:
        return float(self._search_space(self._as_position()))

    def _gradient(self) -> NDArray[np.float64]:
        """Estimate the gradient at the current position via central differences.

        The search space clamps out-of-bounds queries to the grid boundary,
        so no explicit boundary guard is needed here.
        """
        x, y = float(self._pos[0]), float(self._pos[1])
        gx = (
            self._search_space((x + _DELTA, y))
            - self._search_space((x - _DELTA, y))
        ) / (2.0 * _DELTA)
        gy = (
            self._search_space((x, y + _DELTA))
            - self._search_space((x, y - _DELTA))
        ) / (2.0 * _DELTA)
        return np.array([gx, gy], dtype=np.float64)

    def update(self, shared_memory: dict[str, Any]) -> None:
        # Step 1 & 2 — estimate gradient and descent direction
        grad = self._gradient()
        grad_norm = float(np.linalg.norm(grad))

        # Step 3 — backtracking line search (skipped at local minimum)
        x_descent = self._pos.copy()
        descended = False
        if grad_norm >= _GRAD_TOL:
            direction = -grad / grad_norm
            current_score = self.score
            step_trial = self._step
            while step_trial >= _STEP_MIN:
                x_trial = self._pos + step_trial * direction
                np.clip(x_trial, self._lo, self._hi, out=x_trial)
                if (
                    self._search_space(
                        (float(x_trial[0]), float(x_trial[1]))
                    )
                    < current_score
                ):
                    x_descent = x_trial
                    descended = True
                    break
                step_trial *= _TAU

        # Only record the gradient when a descent step was actually taken.
        # If the line search found no improvement the agent is stuck; storing
        # zeros suppresses the attraction arrow for that iteration.
        self._last_grad = grad.copy() if descended else np.zeros(2, dtype=np.float64)

        # Step 4 — unconditional random perturbation
        self._pos = x_descent
        if self._alpha > 0.0:
            eps = self._rng.uniform(-0.5, 0.5, size=2)
            self._pos = self._pos + self._alpha * eps

        # Step 5 — boundary clamping
        np.clip(self._pos, self._lo, self._hi, out=self._pos)

        # Step 6 — update personal best
        new_score = self.score
        if new_score < self._best_score:
            self._best_score = new_score
            self._best_pos = self._pos.copy()


class SDSwarm(Swarm):
    """Steepest Descent swarm.

    Agents are fully independent — shared_memory is always empty.
    """

    def __init__(
        self,
        n_agents: int,
        search_space: SearchSpace,
        dimensions: tuple[int, int],
        seed: int,
        step: float | None = None,
        alpha: float = 0.0,
        noise: float = 0.0,
    ) -> None:
        w_dim, h_dim = dimensions
        resolved_step = (
            step if step is not None else 0.1 * min(w_dim, h_dim)
        )

        rng = np.random.default_rng(seed)
        positions = rng.uniform(
            [0.0, 0.0],
            [float(w_dim - 1), float(h_dim - 1)],
            size=(n_agents, 2),
        )
        agent_rngs = rng.spawn(n_agents)

        self._agents = [
            SDAgent(
                position=positions[i],
                search_space=(
                    _make_noisy(search_space, noise, agent_rngs[i])
                    if noise > 0 else search_space
                ),
                bounds=dimensions,
                step=resolved_step,
                alpha=alpha,
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
        """Return one gradient-direction arrow per agent.

        Arrow direction: negative gradient (steepest descent direction).
        Arrow weight: gradient magnitude normalised by the per-frame maximum,
        so the agent with the steepest local slope gets a full-length arrow
        and all others are shown relative to it.
        """
        grads = [a._gradient() for a in self._agents]
        magnitudes = [float(np.linalg.norm(g)) for g in grads]
        max_mag = max(magnitudes) if any(m > 0 for m in magnitudes) else 1.0

        result: list[list[AttractionVector]] = []
        for agent, grad, mag in zip(self._agents, grads, magnitudes):
            if mag < _GRAD_TOL:
                result.append([])
                continue
            direction = -grad / mag   # unit vector along steepest descent
            pos = agent._pos
            # Place target far in the descent direction; _draw_arrow
            # normalises the direction itself, so only direction matters.
            target: Position = (
                float(pos[0] + direction[0] * 1000.0),
                float(pos[1] + direction[1] * 1000.0),
            )
            result.append([AttractionVector(
                target=target,
                weight=mag / max_mag,
                kind="gradient",
            )])
        return result
