"""Build animated GIFs from swarm simulation runs."""

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from mountain_ridge.swarm.base import AttractionVector, Position, Swarm


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

_BEST_COLOUR: tuple[int, int, int] = (255, 255, 0)
_TRUE_MIN_COLOUR: tuple[int, int, int] = (255, 255, 255)

# Agent colour gradient: bright/best (low score) → dim/worst (high score)
_AGENT_BRIGHT: tuple[int, int, int] = (255, 255, 0)   # yellow
_AGENT_DIM: tuple[int, int, int] = (255, 0, 0)         # red

# Fixed agent colour used when score-based colouring is disabled (e.g. PSO)
_AGENT_COLOUR: tuple[int, int, int] = (255, 0, 0)      # red

# Attraction arrow colours by kind
_ATTRACTION_COLOURS: dict[str, tuple[int, int, int]] = {
    "pbest":    (0,   200, 255),   # cyan
    "gbest":    (220,   0, 220),   # magenta
    "firefly":  (255, 140,   0),   # orange
    "gradient": (0,   220,  80),   # green
    "inertia":  (255, 220,   0),   # yellow
}
_ARROW_MAX_LEN_FACTOR: int = 6    # max arrow length = factor × agent_radius
_ARROW_HEAD_FACTOR: float = 0.3   # arrowhead size = factor × arrow length

# Info bar (detailed output)
_INFO_BAR_WIDTH: int = 75
_BAR_BG: tuple[int, int, int] = (20, 20, 20)
_BAR_LABEL: tuple[int, int, int] = (150, 150, 150)
_BAR_VALUE: tuple[int, int, int] = (255, 255, 255)


def _make_info_bar(
    height: int,
    iteration: int,
    best_score: float,
    best_iter: int,
    global_min: float,
) -> Image.Image:
    """Render a vertical info bar showing simulation statistics."""
    bar = Image.new("RGB", (_INFO_BAR_WIDTH, height), _BAR_BG)
    draw = ImageDraw.Draw(bar)
    font = ImageFont.load_default(size=13)
    pad = 5
    line_h = 18
    gap = 8
    y = pad
    for label, value in [
        ("Iteration", str(iteration)),
        ("Best height", f"{best_score:.6f}"),
        ("Best since", f"iter {best_iter}"),
        ("Global min", f"{global_min:.6f}"),
    ]:
        draw.text((pad, y), label, fill=_BAR_LABEL, font=font)
        y += line_h
        draw.text((pad, y), value, fill=_BAR_VALUE, font=font)
        y += line_h + gap
    return bar


def _find_global_minima(
    grid: NDArray[np.float64],
) -> list[tuple[int, int]]:
    """Return (x, y) pixel coords of every cell at the global minimum.

    A cell qualifies if and only if its value is exactly equal to the
    lowest value anywhere in the grid (``grid.min()``).
    """
    global_min = float(grid.min())
    selected = np.argwhere(grid == global_min)
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


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    ax: float,
    ay: float,
    tx: float,
    ty: float,
    weight: float,
    colour: tuple[int, int, int],
    max_len: int,
) -> None:
    """Draw a weighted arrow from (ax, ay) toward (tx, ty).

    The arrow length equals ``weight * max_len`` pixels; the tip does
    not necessarily reach (tx, ty).
    """
    dx, dy = tx - ax, ty - ay
    dist = math.hypot(dx, dy)
    if dist == 0:
        return
    ux, uy = dx / dist, dy / dist
    length = weight * max_len
    tip_x = ax + ux * length
    tip_y = ay + uy * length

    draw.line(
        [(round(ax), round(ay)), (round(tip_x), round(tip_y))],
        fill=colour,
        width=1,
    )

    # Small filled triangle arrowhead at the tip
    head = max(2, round(length * _ARROW_HEAD_FACTOR))
    px, py = -uy * head, ux * head
    pts = [
        (round(tip_x), round(tip_y)),
        (round(tip_x - ux * head + px), round(tip_y - uy * head + py)),
        (round(tip_x - ux * head - px), round(tip_y - uy * head - py)),
    ]
    draw.polygon(pts, fill=colour)


def _render_frame(
    bg: Image.Image,
    positions: list[Position],
    best: Position,
    agent_radius: int,
    scores: list[float] | None = None,
    score_lo: float = 0.0,
    score_hi: float = 1.0,
    attractions: list[list[AttractionVector]] | None = None,
    show_best: bool = True,
) -> Image.Image:
    """Draw agents and global best onto a copy of *bg*.

    When *scores* is provided, each agent is coloured by its score
    (bright = low score, dim = high score).  When *scores* is ``None``
    all agents are drawn in the dim colour.

    When *attractions* is provided, arrows are drawn from each agent
    toward its attraction points before agents are rendered (so agents
    appear on top).
    """
    frame = bg.copy()
    draw = ImageDraw.Draw(frame)

    if attractions is not None:
        max_len = _ARROW_MAX_LEN_FACTOR * agent_radius
        for i, pos in enumerate(positions):
            ax, ay = float(pos[0]), float(pos[1])
            for vec in attractions[i]:
                colour = _ATTRACTION_COLOURS.get(vec.kind, (255, 255, 255))
                _draw_arrow(
                    draw, ax, ay,
                    float(vec.target[0]), float(vec.target[1]),
                    vec.weight, colour, max_len,
                )

    for i, pos in enumerate(positions):
        x, y = int(round(pos[0])), int(round(pos[1]))
        colour = (
            _score_to_colour(scores[i], score_lo, score_hi)
            if scores is not None
            else _AGENT_COLOUR
        )
        r = agent_radius
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=colour,
            outline=(0, 0, 0),
            width=1,
        )
    if show_best:
        bx, by = int(round(best[0])), int(round(best[1]))
        r = agent_radius + 2
        draw.ellipse(
            [bx - r, by - r, bx + r, by + r],
            fill=_BEST_COLOUR,
            outline=(0, 0, 0),
            width=1,
        )
    return frame


def _simulate_frames(
    swarm: Swarm,
    grid: NDArray[np.float64],
    bg: Image.Image,
    n_iterations: int,
    iterations_per_frame: int,
    colour_by_score: bool,
    detailed: bool,
    score_lo: float,
    score_hi: float,
    agent_radius: int,
    global_min: float,
    desc: str,
    progress_position: int | None,
    show_attractions: bool = False,
    show_best: bool = True,
) -> list[Image.Image]:
    """Run *swarm* and return every captured frame as a PIL Image."""
    def _lookup_best_score() -> float:
        bx, by = swarm.get_best_position()
        return float(grid[int(round(by)), int(round(bx))])

    best_ever: float = _lookup_best_score()
    best_iter: int = 0

    def _frame(iteration: int) -> Image.Image:
        scores = swarm.get_scores() if colour_by_score else None
        att = swarm.get_attractions() if show_attractions else None
        f = _render_frame(
            bg,
            swarm.get_positions(),
            swarm.get_best_position(),
            agent_radius,
            scores,
            score_lo,
            score_hi,
            att,
            show_best,
        )
        if detailed:
            info = _make_info_bar(
                f.height, iteration, best_ever, best_iter, global_min
            )
            wide = Image.new(
                "RGB", (f.width + _INFO_BAR_WIDTH, f.height)
            )
            wide.paste(f, (0, 0))
            wide.paste(info, (f.width, 0))
            f = wide
        return f

    frames: list[Image.Image] = [_frame(0)]

    with tqdm(
        total=n_iterations,
        desc=desc,
        unit="iter",
        position=progress_position,
        leave=(progress_position == 0),
        disable=progress_position is None,
        dynamic_ncols=True,
    ) as bar:
        for i in range(1, n_iterations + 1):
            swarm.update()
            bar.update(1)
            if detailed:
                current = _lookup_best_score()
                if current < best_ever:
                    best_ever = current
                    best_iter = i
            if i % iterations_per_frame == 0:
                frames.append(_frame(i))

    return frames



def _make_bg(
    grid: NDArray[np.float64],
    agent_radius: int,
) -> Image.Image:
    """Build the static terrain background with global-minima markers."""
    bg = _grid_to_image(grid)
    marker_r = max(4, agent_radius + 2)
    bg_draw = ImageDraw.Draw(bg)
    for mx, my in _find_global_minima(grid):
        _draw_diamond(bg_draw, mx, my, marker_r)
    del bg_draw
    return bg


def build_gif(
    swarm: Swarm,
    grid: NDArray[np.float64],
    n_iterations: int,
    iterations_per_frame: int,
    fps: int,
    output_path: str | Path,
    dot_size: int | None = None,
    colour_by_score: bool = False,
    detailed: bool = False,
    use_gifsicle: bool = True,
    show_attractions: bool = False,
    show_best: bool = True,
    desc: str = "Simulating",
    progress_position: int | None = 0,
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
    bg = _make_bg(grid, agent_radius)

    frames = _simulate_frames(
        swarm=swarm,
        grid=grid,
        bg=bg,
        n_iterations=n_iterations,
        iterations_per_frame=iterations_per_frame,
        colour_by_score=colour_by_score,
        detailed=detailed,
        score_lo=score_lo,
        score_hi=score_hi,
        agent_radius=agent_radius,
        global_min=score_lo,
        desc=desc,
        progress_position=progress_position,
        show_attractions=show_attractions,
        show_best=show_best,
    )

    # Quantize every frame to one shared palette.  When attraction arrows
    # are enabled, frame 0 (iteration 0, no update yet) contains no arrows
    # and therefore no arrow colours.  Use frame 1 in that case so the
    # palette includes cyan/magenta/orange arrow colours.
    # Dithering is disabled: the noise it introduces hurts LZW compression.
    palette_src = frames[1] if show_attractions and len(frames) > 1 else frames[0]
    palette_img = palette_src.quantize(colors=256)
    quantized: list[Image.Image] = [
        f.quantize(palette=palette_img, dither=Image.Dither.NONE)
        for f in frames
    ]

    duration_ms = max(1, round(1000 / fps))
    quantized[0].save(
        str(output_path),
        save_all=True,
        append_images=quantized[1:],
        loop=0,
        duration=duration_ms,
        optimize=True,
    )

    if use_gifsicle and shutil.which("gifsicle") is not None:
        subprocess.run(
            ["gifsicle", "--batch", "-O3", str(output_path)],
            check=True,
        )


def build_frames(
    swarm: Swarm,
    grid: NDArray[np.float64],
    n_iterations: int,
    iterations_per_frame: int,
    output_dir: Path,
    dot_size: int | None = None,
    colour_by_score: bool = False,
    detailed: bool = False,
    use_png: bool = False,
    show_attractions: bool = False,
    show_best: bool = True,
    desc: str = "Simulating",
    progress_position: int | None = 0,
) -> int:
    """Run *swarm* and write each captured frame as an individual image.

    Frames are saved as JPEG (quality 85) by default, which is consistently
    4× smaller than PNG for this type of content across all built-in search
    spaces.  Pass ``use_png=True`` to write lossless PNG files instead.

    Parameters
    ----------
    output_dir:
        Directory that will receive ``frame_0000.jpeg`` (or ``.png``)
        files.  Must already exist.
    use_png:
        Write PNG instead of JPEG.

    Returns
    -------
    int
        Number of frames written.
    """
    agent_radius: int = (
        dot_size
        if dot_size is not None
        else max(2, round(min(grid.shape) / 35))
    )

    score_lo = float(grid.min())
    score_hi = float(grid.max())
    bg = _make_bg(grid, agent_radius)

    frames = _simulate_frames(
        swarm=swarm,
        grid=grid,
        bg=bg,
        n_iterations=n_iterations,
        iterations_per_frame=iterations_per_frame,
        colour_by_score=colour_by_score,
        detailed=detailed,
        score_lo=score_lo,
        score_hi=score_hi,
        agent_radius=agent_radius,
        global_min=score_lo,
        desc=desc,
        progress_position=progress_position,
        show_attractions=show_attractions,
        show_best=show_best,
    )

    if use_png:
        ext, save_kwargs = "png", {}
    else:
        ext, save_kwargs = "jpeg", {"quality": 85}
    for idx, frame in enumerate(frames):
        frame.save(
            output_dir / f"frame_{idx:04d}.{ext}",
            format=ext.upper(),
            **save_kwargs,
        )
    return len(frames)
