"""JobConfig dataclass and TOML config-file loader."""

import argparse
import itertools
import os
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JobConfig:
    """Parameters for a single GIF-generation run."""

    dimensions: tuple[int, int]
    seed: int
    algorithm: str
    space: str
    n_agents: int
    iterations: int
    iterations_per_frame: int
    fps: int
    dot_size: int | None
    inertia: float | None
    variant: str | None
    gamma: float | None
    beta0: float | None
    alpha: float | None
    levy_exp: float | None
    sd_step: float | None
    sd_alpha: float | None
    sa_t0: float | None
    sa_cooling_rate: float | None
    sa_step: float | None
    detailed: bool
    show_attractions: bool
    frames: bool
    frames_png: bool
    use_gifsicle: bool
    output_prefix: str
    output_dir: Path


def _parse_dims(value: str) -> tuple[int, int]:
    """Parse a *WxH* string into an integer (width, height) pair."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Dimensions must be in WxH format, got {value!r}"
        )
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Dimension values must be integers, got {value!r}"
        )
    if w < 2 or h < 2:
        raise argparse.ArgumentTypeError(
            f"Dimensions must be at least 2x2, got {value!r}"
        )
    return w, h


def _build_jobs(
    dimensions: list[tuple[int, int]],
    seeds: list[int | None],
    algorithms: list[str],
    spaces: list[str],
    n_agents_l: list[int],
    iterations_l: list[int],
    ipf_l: list[int],
    fps_l: list[int],
    dot_sizes: list[int | None],
    inertias: list[float | None],
    variants: list[str | None],
    gammas: list[float | None],
    beta0s: list[float | None],
    fa_alphas: list[float | None],
    levy_exps: list[float | None],
    sd_steps: list[float | None],
    sd_alphas: list[float | None],
    sa_t0s: list[float | None],
    sa_cooling_rates: list[float | None],
    sa_steps: list[float | None],
    detailed: bool,
    show_attractions: bool,
    frames: bool,
    frames_png: bool,
    use_gifsicle: bool,
    output_prefix: str,
    output_dir: Path,
) -> list[JobConfig]:
    """Expand parameter lists into one JobConfig per combination.

    Common parameters are crossed with all algorithms; each algorithm
    is then crossed only with *its own* specific parameters, so PSO
    inertia values never bleed into FA jobs and vice-versa.
    """
    jobs: list[JobConfig] = []
    for common in itertools.product(
        dimensions, seeds, spaces, n_agents_l,
        iterations_l, ipf_l, fps_l, dot_sizes,
    ):
        dims, seed, space, n_agents, iters, ipf, fps, dot = common
        for algo in algorithms:
            if algo == "pso":
                specific_iter = itertools.product(inertias)
            elif algo == "fa":
                specific_iter = itertools.product(
                    variants, gammas, beta0s, fa_alphas, levy_exps
                )
            elif algo == "sd":
                specific_iter = itertools.product(sd_steps, sd_alphas)
            elif algo == "sa":
                specific_iter = itertools.product(
                    sa_t0s, sa_cooling_rates, sa_steps
                )
            else:
                specific_iter = iter([()])

            for specific in specific_iter:
                if algo == "pso":
                    (w,) = specific
                    variant = gamma = beta0 = alpha = levy_exp = None
                    sd_step = sd_alpha = None
                    sa_t0 = sa_cooling_rate = sa_step = None
                elif algo == "fa":
                    w = None
                    (variant, gamma, beta0, alpha, levy_exp) = specific
                    sd_step = sd_alpha = None
                    sa_t0 = sa_cooling_rate = sa_step = None
                elif algo == "sd":
                    w = None
                    variant = gamma = beta0 = alpha = levy_exp = None
                    (sd_step, sd_alpha) = specific
                    sa_t0 = sa_cooling_rate = sa_step = None
                elif algo == "sa":
                    w = None
                    variant = gamma = beta0 = alpha = levy_exp = None
                    sd_step = sd_alpha = None
                    (sa_t0, sa_cooling_rate, sa_step) = specific
                else:
                    w = variant = gamma = beta0 = alpha = levy_exp = None
                    sd_step = sd_alpha = None
                    sa_t0 = sa_cooling_rate = sa_step = None

                resolved_seed: int = (
                    random.randint(0, 2**31 - 1)
                    if seed is None else seed
                )
                jobs.append(JobConfig(
                    dimensions=dims,
                    seed=resolved_seed,
                    algorithm=algo,
                    space=space,
                    n_agents=n_agents,
                    iterations=iters,
                    iterations_per_frame=ipf,
                    fps=fps,
                    dot_size=dot,
                    inertia=w,
                    variant=variant,
                    gamma=gamma,
                    beta0=beta0,
                    alpha=alpha,
                    levy_exp=levy_exp,
                    sd_step=sd_step,
                    sd_alpha=sd_alpha,
                    sa_t0=sa_t0,
                    sa_cooling_rate=sa_cooling_rate,
                    sa_step=sa_step,
                    detailed=detailed,
                    show_attractions=show_attractions,
                    frames=frames,
                    frames_png=frames_png,
                    use_gifsicle=use_gifsicle,
                    output_prefix=output_prefix,
                    output_dir=output_dir,
                ))
    return jobs


def load_config(path: Path) -> tuple[list[JobConfig], int]:
    """Load job configurations from a TOML config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as fh:
        data: dict[str, object] = tomllib.load(fh)

    def _norm(
        val: object, default: list[object]
    ) -> list[object]:
        """Normalise a scalar/list/None/empty-list to a non-empty list."""
        if val is None or val == []:
            return default
        if isinstance(val, list):
            return val
        return [val]

    def _get(
        d: dict[str, object],
        key: str,
        default: list[object],
    ) -> list[object]:
        return _norm(d.get(key), default)

    # ── Parallel workers ─────────────────────────────────────────────────
    workers: int = int(data.get("workers", os.cpu_count() or 1))

    # ── Output / display flags (single values, not batched) ──────────────
    output_prefix: str = str(data.get("output_prefix", "out"))
    output_dir = Path(str(data.get("output_dir", ".")))
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed: bool = bool(data.get("detailed", False))
    show_attractions: bool = bool(
        data.get("show_attractions", False)
    )
    frames: bool = bool(data.get("frames", False))
    frames_png: bool = bool(data.get("png", False))
    use_gifsicle: bool = not bool(data.get("no_gifsicle", False))

    if frames_png and not frames:
        raise ValueError("'png = true' requires 'frames = true'")

    # ── Batch parameters ─────────────────────────────────────────────────
    dims_raw = _get(data, "dimensions", ["100x100"])
    dimensions: list[tuple[int, int]] = []
    for raw in dims_raw:
        if not isinstance(raw, str):
            raise ValueError(
                f"dimensions values must be 'WxH' strings, "
                f"got {raw!r}"
            )
        try:
            dimensions.append(_parse_dims(raw))
        except argparse.ArgumentTypeError as exc:
            raise ValueError(str(exc)) from exc

    seeds: list[int | None] = list(  # type: ignore[assignment]
        _get(data, "seed", [None])
    )
    algorithms: list[str] = list(  # type: ignore[assignment]
        _get(data, "algorithm", ["pso"])
    )
    spaces: list[str] = list(  # type: ignore[assignment]
        _get(data, "space", ["ridge"])
    )
    n_agents_l: list[int] = list(  # type: ignore[assignment]
        _get(data, "n_agents", [20])
    )
    iterations_l: list[int] = list(  # type: ignore[assignment]
        _get(data, "iterations", [100])
    )
    ipf_l: list[int] = list(  # type: ignore[assignment]
        _get(data, "iterations_per_frame", [5])
    )
    fps_l: list[int] = list(  # type: ignore[assignment]
        _get(data, "fps", [10])
    )
    dot_sizes: list[int | None] = list(  # type: ignore[assignment]
        _get(data, "dot_size", [None])
    )

    # ── Algorithm-specific parameters ────────────────────────────────────
    pso_s: dict[str, object] = (  # type: ignore[assignment]
        data.get("pso") or {}
    )
    fa_s: dict[str, object] = (  # type: ignore[assignment]
        data.get("fa") or {}
    )
    sd_s: dict[str, object] = (  # type: ignore[assignment]
        data.get("sd") or {}
    )
    sa_s: dict[str, object] = (  # type: ignore[assignment]
        data.get("sa") or {}
    )

    inertias: list[float | None] = list(  # type: ignore[assignment]
        _get(pso_s, "inertia", [None])
    )
    variants: list[str | None] = list(  # type: ignore[assignment]
        _get(fa_s, "variant", [None])
    )
    gammas: list[float | None] = list(  # type: ignore[assignment]
        _get(fa_s, "gamma", [None])
    )
    beta0s: list[float | None] = list(  # type: ignore[assignment]
        _get(fa_s, "beta0", [None])
    )
    fa_alphas: list[float | None] = list(  # type: ignore[assignment]
        _get(fa_s, "alpha", [None])
    )
    levy_exps: list[float | None] = list(  # type: ignore[assignment]
        _get(fa_s, "levy_exp", [None])
    )
    sd_steps: list[float | None] = list(  # type: ignore[assignment]
        _get(sd_s, "step", [None])
    )
    sd_alphas: list[float | None] = list(  # type: ignore[assignment]
        _get(sd_s, "alpha", [None])
    )
    sa_t0s: list[float | None] = list(  # type: ignore[assignment]
        _get(sa_s, "t0", [None])
    )
    sa_cooling_rates: list[float | None] = list(  # type: ignore[assignment]
        _get(sa_s, "cooling_rate", [None])
    )
    sa_steps: list[float | None] = list(  # type: ignore[assignment]
        _get(sa_s, "step", [None])
    )

    jobs = _build_jobs(
        dimensions=dimensions,
        seeds=seeds,
        algorithms=algorithms,
        spaces=spaces,
        n_agents_l=n_agents_l,
        iterations_l=iterations_l,
        ipf_l=ipf_l,
        fps_l=fps_l,
        dot_sizes=dot_sizes,
        inertias=inertias,
        variants=variants,
        gammas=gammas,
        beta0s=beta0s,
        fa_alphas=fa_alphas,
        levy_exps=levy_exps,
        sd_steps=sd_steps,
        sd_alphas=sd_alphas,
        sa_t0s=sa_t0s,
        sa_cooling_rates=sa_cooling_rates,
        sa_steps=sa_steps,
        detailed=detailed,
        show_attractions=show_attractions,
        frames=frames,
        frames_png=frames_png,
        use_gifsicle=use_gifsicle,
        output_prefix=output_prefix,
        output_dir=output_dir,
    )
    return jobs, workers
