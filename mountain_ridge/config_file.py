"""JobConfig dataclass and TOML config-file loader."""

import argparse
import itertools
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


def load_config(path: Path) -> list[JobConfig]:
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

    # ── Cross-algorithm validation ────────────────────────────────────────
    if any(v is not None for v in inertias):
        bad = [a for a in algorithms if a != "pso"]
        if bad:
            raise ValueError(
                f"[pso].inertia only supported with algorithm "
                f"'pso' (got: {bad})"
            )

    fa_given = [
        name for name, lst in [
            ("variant",  variants),
            ("gamma",    gammas),
            ("beta0",    beta0s),
            ("alpha",    fa_alphas),
            ("levy_exp", levy_exps),
        ] if any(v is not None for v in lst)
    ]
    if fa_given:
        bad = [a for a in algorithms if a != "fa"]
        if bad:
            keys = ", ".join(f"[fa].{k}" for k in fa_given)
            raise ValueError(
                f"{keys} only supported with algorithm "
                f"'fa' (got: {bad})"
            )

    sd_given = [
        name for name, lst in [
            ("step",  sd_steps),
            ("alpha", sd_alphas),
        ] if any(v is not None for v in lst)
    ]
    if sd_given:
        bad = [a for a in algorithms if a != "sd"]
        if bad:
            keys = ", ".join(f"[sd].{k}" for k in sd_given)
            raise ValueError(
                f"{keys} only supported with algorithm "
                f"'sd' (got: {bad})"
            )

    sa_given = [
        name for name, lst in [
            ("t0",           sa_t0s),
            ("cooling_rate", sa_cooling_rates),
            ("step",         sa_steps),
        ] if any(v is not None for v in lst)
    ]
    if sa_given:
        bad = [a for a in algorithms if a != "sa"]
        if bad:
            keys = ", ".join(f"[sa].{k}" for k in sa_given)
            raise ValueError(
                f"{keys} only supported with algorithm "
                f"'sa' (got: {bad})"
            )

    # ── Expand Cartesian product → JobConfigs ─────────────────────────────
    jobs: list[JobConfig] = []
    for combo in itertools.product(
        dimensions,
        seeds,
        algorithms,
        spaces,
        n_agents_l,
        iterations_l,
        ipf_l,
        fps_l,
        dot_sizes,
        inertias,
        variants,
        gammas,
        beta0s,
        fa_alphas,
        levy_exps,
        sd_steps,
        sd_alphas,
        sa_t0s,
        sa_cooling_rates,
        sa_steps,
    ):
        (
            dims, seed, algo, space, n_agents,
            iters, ipf, fps, dot, w,
            variant, gamma, beta0, alpha, levy_exp,
            sd_step, sd_alpha,
            sa_t0, sa_cooling_rate, sa_step,
        ) = combo
        resolved_seed: int = (
            random.randint(0, 2**31 - 1) if seed is None else seed
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
