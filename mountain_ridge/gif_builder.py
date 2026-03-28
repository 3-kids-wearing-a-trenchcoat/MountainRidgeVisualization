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

_AGENT_COLOUR: tuple[int, int, int] = (220, 50, 50)
_BEST_COLOUR: tuple[int, int, int] = (255, 220, 0)


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
) -> Image.Image:
    """Draw agents and global best onto a copy of *bg*."""
    frame = bg.copy()
    draw = ImageDraw.Draw(frame)
    for pos in positions:
        x, y = int(round(pos[0])), int(round(pos[1]))
        r = agent_radius
        draw.ellipse([x - r, y - r, x + r, y + r], fill=_AGENT_COLOUR)
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

    bg = _grid_to_image(grid)
    frames: list[Image.Image] = [
        _render_frame(
            bg,
            swarm.get_positions(),
            swarm.get_best_position(),
            agent_radius,
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
