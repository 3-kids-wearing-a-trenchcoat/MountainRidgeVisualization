"""Search space factory registry and lookup."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from mountain_ridge.swarm.base import Position, SearchSpace


SpaceFactory = Callable[[int, tuple[int, int]], SearchSpace]

_REGISTRY: dict[str, SpaceFactory] = {}


def register(name: str) -> Callable[[SpaceFactory], SpaceFactory]:
    """Decorator to register a search space factory by name."""
    def decorator(fn: SpaceFactory) -> SpaceFactory:
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_space(
    name: str,
    seed: int,
    dimensions: tuple[int, int],
) -> SearchSpace:
    """Return an instantiated search space by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown search space: {name!r}. "
            f"Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name](seed, dimensions)


def available() -> list[str]:
    """Return the names of all registered search spaces."""
    return list(_REGISTRY)


def _bilinear_interpolate(
    grid: NDArray[np.float64],
    position: Position,
) -> float:
    """Bilinearly interpolate a position into a 2D grid."""
    h, w = grid.shape
    x = float(np.clip(position[0], 0, w - 1))
    y = float(np.clip(position[1], 0, h - 1))
    x0, y0 = int(x), int(y)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    return float(
        grid[y0, x0] * (1 - dx) * (1 - dy)
        + grid[y0, x1] * dx       * (1 - dy)
        + grid[y1, x0] * (1 - dx) * dy
        + grid[y1, x1] * dx       * dy
    )


@register("gaussian")
def _gaussian_mixture(
    seed: int,
    dimensions: tuple[int, int],
) -> SearchSpace:
    """Height map built from a sum of random Gaussians."""
    rng = np.random.default_rng(seed)
    w, h = dimensions
    n = max(5, (w * h) // 500)
    cx = rng.uniform(0, w, n)
    cy = rng.uniform(0, h, n)
    amp = rng.uniform(-1.0, 1.0, n)
    sig = rng.uniform(min(w, h) * 0.05, min(w, h) * 0.3, n)

    gx, gy = np.meshgrid(
        np.arange(w, dtype=float),
        np.arange(h, dtype=float),
    )
    grid = np.zeros((h, w), dtype=np.float64)
    for x, y, a, s in zip(cx, cy, amp, sig):
        grid += a * np.exp(
            -((gx - x) ** 2 + (gy - y) ** 2) / (2 * s ** 2)
        )

    def space(position: Position) -> float:
        return _bilinear_interpolate(grid, position)

    return space
