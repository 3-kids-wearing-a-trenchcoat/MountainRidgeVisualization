"""Build animated GIFs from swarm simulation runs."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from tqdm import tqdm

from mountain_ridge.swarm.base import Position, Swarm


# Terrain colormap: (normalised_value, (R, G, B))
_TERRAIN: list[tuple[float, tuple[int, int, int]]] = [
    (0.00, (0,   20,  80)),
    (0.25, (0,   80, 200)),
    (0.50, (40, 160,  40)),
    (0.75, (160, 100,  40)),
    (1.00, (255, 255, 255)),
]

_STOPS_T = np.array([s[0] for s in _TERRAIN], dtype=np.float64)
_STOPS_R = np.array([s[1][0] for s in _TERRAIN], dtype=np.float64)
_STOPS_G = np.array([s[1][1] for s in _TERRAIN], dtype=np.float64)
_STOPS_B = np.array([s[1][2] for s in _TERRAIN], dtype=np.float64)

_BEST_COLOUR: tuple[int, int, int] = (255, 220, 0)
_TRUE_MIN_COLOUR: tuple[int, int, int] = (255, 255, 255)

# Agent colour gradient: bright (low score) → dim (high score)
_AGENT_BRIGHT: tuple[int, int, int] = (255, 240, 80)
_AGENT_DIM: tuple[int, int, int] = (80, 10, 10)


def _find_global_minima(
    grid: NDArray[np.float64],
) -> list[tuple[int, int]]:
    """Return (x, y) pixel coords of all local minima at the global depth.

    A cell qualifies if it is lower than or equal to all 8 neighbours
    AND its value is within 2 % of the grid's value range above the
    true grid minimum.  The tolerance handles the discretisation gap
    between intended equal-depth minima (e.g. multiwell) whose centres
    do not fall exactly on grid points.
    """
    h, w = grid.shape
    padded = np.pad(grid, 1, constant_values=np.inf)

    is_local_min = np.ones((h, w), dtype=bool)
    for dr, dc in [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]:
        neighbour = padded[1 + dr: h + 1 + dr, 1 + dc: w + 1 + dc]
        is_local_min &= grid <= neighbour

    global_min = float(grid.min())
    grid_range = float(grid.max()) - global_min
    threshold = global_min + grid_range * 0.02

    selected = np.argwhere(is_local_min & (grid <= threshold))
    return [(int(c), int(r)) for r, c in selected]


def _draw_diamond(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    radius: int,
) -> None:
    """Draw a filled diamond (rotated square) centred at (cx, cy)."""
    pts = [
        (cx,          cy - radius),
        (cx + radius, cy),
        (cx,          cy + radius),
        (cx - radius, cy),
    ]
    draw.polygon(pts, fill=_TRUE_MIN_COLOUR, outline=(0, 0, 0))


def _score_to_colour(
    score: float,
    score_lo: float,
    score_hi: float,
) -> tuple[int, int, int]:
    """Interpolate between bright (low score) and dim (high score)."""
    span = score_hi - score_lo if score_hi != score_lo else 1.0
    t = float(np.clip((score - score_lo) / span, 0.0, 1.0))
    return (
        int(_AGENT_BRIGHT[0] + t * (_AGENT_DIM[0] - _AGENT_BRIGHT[0])),
        int(_AGENT_BRIGHT[1] + t * (_AGENT_DIM[1] - _AGENT_BRIGHT[1])),
        int(_AGENT_BRIGHT[2] + t * (_AGENT_DIM[2] - _AGENT_BRIGHT[2])),
    )


def _grid_to_image(grid: NDArray[np.float64]) -> Image.Image:
    """Convert a 2-D height grid to an RGB PIL Image."""
    g_min = float(grid.min())
    g_max = float(grid.max())
    span = g_max - g_min if g_max != g_min else 1.0
    norm = (grid - g_min) / span
    flat = norm.ravel()
    r = np.interp(flat, _STOPS_T, _STOPS_R).astype(np.uint8)
    g = np.interp(flat, _STOPS_T, _STOPS_G).astype(np.uint8)
    b = np.interp(flat, _STOPS_T, _STOPS_B).astype(np.uint8)
    h, w = grid.shape
    rgb = np.stack([r, g, b], axis=1).reshape(h, w, 3)
    return Image.fromarray(rgb, mode="RGB")


def _render_frame(
    bg: Image.Image,
    positions: list[Position],
    best: Position,
    agent_radius: int,
    scores: list[float],
    score_lo: float,
    score_hi: float,
) -> Image.Image:
    """Draw agents and global best onto a copy of *bg*."""
    frame = bg.copy()
    draw = ImageDraw.Draw(frame)
    for pos, score in zip(positions, scores):
        x, y = int(round(pos[0])), int(round(pos[1]))
        colour = _score_to_colour(score, score_lo, score_hi)
        r = agent_radius
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=colour,
            outline=(0, 0, 0),
            width=1,
        )
    bx, by = int(round(best[0])), int(round(best[1]))
    r = agent_radius + 2
    draw.ellipse(
        [bx - r, by - r, bx + r, by + r],
        fill=_BEST_COLOUR,
        outline=(0, 0, 0),
        width=1,
    )
    return frame


def build_gif(
    swarm: Swarm,
    grid: NDArray[np.float64],
    n_iterations: int,
    iterations_per_frame: int,
    fps: int,
    output_path: str | Path,
    dot_size: int | None = None,
    desc: str = "Simulating",
    progress_position: int = 0,
) -> None:
    """Run *swarm* for *n_iterations* and write an animated GIF.

    Parameters
    ----------
    swarm:
        An already-initialised ``Swarm``.  Its initial state becomes
        frame 0.
    grid:
        Precomputed height-map array with shape ``(height, width)``.
    n_iterations:
        Total simulation steps to run.
    iterations_per_frame:
        Capture one frame every this many iterations (plus frame 0).
    fps:
        Playback speed of the output GIF.
    output_path:
        Destination ``.gif`` file.  Created or overwritten.
    dot_size:
        Radius of agent dots in pixels.  ``None`` (default) scales
        automatically: ``max(2, round(min(height, width) / 35))``.
    """
    agent_radius: int = (
        dot_size
        if dot_size is not None
        else max(2, round(min(grid.shape) / 35))
    )

    score_lo = float(grid.min())
    score_hi = float(grid.max())

    bg = _grid_to_image(grid)
    marker_r = max(4, agent_radius + 2)
    bg_draw = ImageDraw.Draw(bg)
    for mx, my in _find_global_minima(grid):
        _draw_diamond(bg_draw, mx, my, marker_r)
    del bg_draw

    frames: list[Image.Image] = [
        _render_frame(
            bg,
            swarm.get_positions(),
            swarm.get_best_position(),
            agent_radius,
            swarm.get_scores(),
            score_lo,
            score_hi,
        )
    ]

    with tqdm(
        total=n_iterations,
        desc=desc,
        unit="iter",
        position=progress_position,
        leave=(progress_position == 0),
        dynamic_ncols=True,
    ) as bar:
        for i in range(1, n_iterations + 1):
            swarm.update()
            bar.update(1)
            if i % iterations_per_frame == 0:
                frames.append(
                    _render_frame(
                        bg,
                        swarm.get_positions(),
                        swarm.get_best_position(),
                        agent_radius,
                        swarm.get_scores(),
                        score_lo,
                        score_hi,
                    )
                )

    duration_ms = max(1, round(1000 / fps))
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
