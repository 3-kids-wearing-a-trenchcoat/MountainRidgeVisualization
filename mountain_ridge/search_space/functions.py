"""Search space factory registry and lookup."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from mountain_ridge.swarm.base import Position, SearchSpace


SpaceFactory = Callable[
    [int, tuple[int, int]], tuple[SearchSpace, NDArray[np.float64]]
]

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
) -> tuple[SearchSpace, NDArray[np.float64]]:
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
) -> tuple[SearchSpace, NDArray[np.float64]]:
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

    return space, grid


@register("rastrigin")
def _rastrigin(
    seed: int,
    dimensions: tuple[int, int],
) -> tuple[SearchSpace, NDArray[np.float64]]:
    """Height map based on a Rastrigin-like function.

    A parabolic bowl perturbed by a cosine grid, producing a large
    number of evenly-spaced local minima with one global minimum near
    the centre.  The seed controls per-axis phase offsets and the
    oscillation frequency, making each seed a distinct landscape.
    """
    rng = np.random.default_rng(seed)
    w, h = dimensions

    phase_x = rng.uniform(0.0, 2 * np.pi)
    phase_y = rng.uniform(0.0, 2 * np.pi)
    freq = rng.uniform(0.8, 1.5)

    x = np.linspace(-5.0, 5.0, w)
    y = np.linspace(-5.0, 5.0, h)
    gx, gy = np.meshgrid(x, y)

    A = 10.0
    grid = (
        A * 2
        + (gx ** 2 - A * np.cos(2 * np.pi * freq * gx + phase_x))
        + (gy ** 2 - A * np.cos(2 * np.pi * freq * gy + phase_y))
    ).astype(np.float64)

    def space(position: Position) -> float:
        return _bilinear_interpolate(grid, position)

    return space, grid


@register("ackley")
def _ackley(
    seed: int,
    dimensions: tuple[int, int],
) -> tuple[SearchSpace, NDArray[np.float64]]:
    """Height map based on an Ackley-like function.

    A near-flat surface covered by a dense cosine carpet with a single
    sharp global minimum.  The seed controls the position of that
    minimum within the central half of the space.
    """
    rng = np.random.default_rng(seed)
    w, h = dimensions

    cx = rng.uniform(w * 0.25, w * 0.75)
    cy = rng.uniform(h * 0.25, h * 0.75)

    # Scale so the function spans ~[-4, 4] around the minimum
    scale = 8.0 / min(w, h)
    xs = (np.arange(w, dtype=np.float64) - cx) * scale
    ys = (np.arange(h, dtype=np.float64) - cy) * scale
    gx, gy = np.meshgrid(xs, ys)

    a, b, c = 20.0, 0.2, 2 * np.pi
    grid = (
        -a * np.exp(-b * np.sqrt((gx ** 2 + gy ** 2) / 2))
        - np.exp((np.cos(c * gx) + np.cos(c * gy)) / 2)
        + a + np.e
    ).astype(np.float64)

    def space(position: Position) -> float:
        return _bilinear_interpolate(grid, position)

    return space, grid


@register("multiwell")
def _multiwell(
    seed: int,
    dimensions: tuple[int, int],
) -> tuple[SearchSpace, NDArray[np.float64]]:
    """Height map with N identical Gaussian wells on a flat background.

    Wells are equally deep and arranged in a ring with uniform angular
    spacing.  The seed controls the number of wells (4–8), ring radius,
    and rotation angle.  Designed to expose PSO's tendency to collapse
    into the first promising well and abandon equally-good alternatives.
    """
    rng = np.random.default_rng(seed)
    w, h = dimensions

    n_wells = int(rng.integers(4, 9))
    cx, cy = w / 2.0, h / 2.0
    radius = min(w, h) * rng.uniform(0.25, 0.35)
    angle_offset = rng.uniform(0.0, 2 * np.pi)
    sigma = min(w, h) * 0.07

    angles = (
        np.linspace(0.0, 2 * np.pi, n_wells, endpoint=False)
        + angle_offset
    )
    well_x = cx + radius * np.cos(angles)
    well_y = cy + radius * np.sin(angles)

    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float64),
        np.arange(h, dtype=np.float64),
    )
    grid = np.ones((h, w), dtype=np.float64)
    for wx, wy in zip(well_x, well_y):
        grid -= np.exp(
            -((gx - wx) ** 2 + (gy - wy) ** 2) / (2 * sigma ** 2)
        )

    def space(position: Position) -> float:
        return _bilinear_interpolate(grid, position)

    return space, grid


@register("deceptive")
def _deceptive(
    seed: int,
    dimensions: tuple[int, int],
) -> tuple[SearchSpace, NDArray[np.float64]]:
    """Height map with a broad deceptive basin and hidden sharp pits.

    A wide, shallow basin near the centre attracts agents early due to
    its large footprint.  One or two narrow, deeper pits placed in the
    periphery are the true global minima but occupy little area and are
    easy to miss.  Designed to expose PSO's vulnerability to early
    convergence toward the most visible rather than the deepest feature.
    """
    rng = np.random.default_rng(seed)
    w, h = dimensions

    basin_x = w / 2.0 + rng.uniform(-w * 0.1, w * 0.1)
    basin_y = h / 2.0 + rng.uniform(-h * 0.1, h * 0.1)
    basin_sigma = min(w, h) * rng.uniform(0.20, 0.30)

    n_pits = int(rng.integers(1, 3))
    pit_dist = min(w, h) * rng.uniform(0.28, 0.38)
    pit_angles = rng.uniform(0.0, 2 * np.pi, n_pits)
    pit_x = w / 2.0 + pit_dist * np.cos(pit_angles)
    pit_y = h / 2.0 + pit_dist * np.sin(pit_angles)
    pit_sigma = min(w, h) * rng.uniform(0.03, 0.05)

    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float64),
        np.arange(h, dtype=np.float64),
    )
    grid = np.ones((h, w), dtype=np.float64)

    # Basin: large footprint, moderate depth (0.6) — mediocre minimum
    grid -= 0.6 * np.exp(
        -((gx - basin_x) ** 2 + (gy - basin_y) ** 2)
        / (2 * basin_sigma ** 2)
    )
    # Pits: narrow, deeper (1.0) — true global minima
    for px, py in zip(pit_x, pit_y):
        grid -= 1.0 * np.exp(
            -((gx - px) ** 2 + (gy - py) ** 2) / (2 * pit_sigma ** 2)
        )

    def space(position: Position) -> float:
        return _bilinear_interpolate(grid, position)

    return space, grid
