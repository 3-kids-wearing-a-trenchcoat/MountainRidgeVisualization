"""CLI argument parsing and batch job generation."""

import argparse
import os
import sys
from pathlib import Path

from mountain_ridge.config_file import (
    JobConfig, _build_jobs, _parse_dims, load_config,
)


def parse_jobs() -> tuple[list[JobConfig], int]:
    """Parse command-line arguments and return one config per GIF job."""
    parser = argparse.ArgumentParser(
        prog="mountain_ridge",
        description=(
            "Visualize swarm intelligence algorithms "
            "on a 2-D height-map search space."
        ),
    )
    parser.add_argument(
        "--dimensions", "-d",
        nargs="+",
        type=_parse_dims,
        default=[(100, 100)],
        metavar="WxH",
        help="Search space dimensions (default: 100x100)",
    )
    parser.add_argument(
        "--seed", "-s",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed; omit for a random seed per job",
    )
    parser.add_argument(
        "--algorithm", "-a",
        nargs="+",
        default=["pso"],
        metavar="ALGO",
        help="Optimization algorithm (default: pso)",
    )
    parser.add_argument(
        "--space",
        nargs="+",
        default=["ridge"],
        metavar="SPACE",
        help="Search space function (default: ridge)",
    )
    parser.add_argument(
        "--n_agents", "-n",
        nargs="+",
        type=int,
        default=[20],
        metavar="N",
        dest="n_agents",
        help="Number of agents (default: 20)",
    )
    parser.add_argument(
        "--iterations", "-i",
        nargs="+",
        type=int,
        default=[100],
        metavar="N",
        help="Number of iterations (default: 100)",
    )
    parser.add_argument(
        "--iterations_per_frame",
        nargs="+",
        type=int,
        default=[5],
        metavar="N",
        dest="iterations_per_frame",
        help="Iterations between captured frames (default: 5)",
    )
    parser.add_argument(
        "--fps",
        nargs="+",
        type=int,
        default=[10],
        metavar="FPS",
        help="Output GIF frames per second (default: 10)",
    )
    parser.add_argument(
        "--dot_size",
        nargs="+",
        type=int,
        default=None,
        metavar="PX",
        dest="dot_size",
        help=(
            "Agent dot radius in pixels. "
            "Omit to scale automatically with the search space dimensions"
        ),
    )
    parser.add_argument(
        "--inertia",
        nargs="+",
        type=float,
        default=None,
        metavar="W",
        help=(
            "PSO inertia weight w in [0, 1] (default: 0.0). "
            "Error if used with a non-PSO algorithm."
        ),
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        default=None,
        choices=["brownian", "levy"],
        metavar="VARIANT",
        help=(
            "FA random walk distribution: brownian or levy "
            "(default: brownian). Error if used with a non-FA algorithm."
        ),
    )
    parser.add_argument(
        "--gamma",
        nargs="+",
        type=float,
        default=None,
        metavar="G",
        help=(
            "FA light absorption coefficient (default: 1.0). "
            "Error if used with a non-FA algorithm."
        ),
    )
    parser.add_argument(
        "--beta0",
        nargs="+",
        type=float,
        default=None,
        metavar="B",
        help=(
            "FA maximum attractiveness at distance zero (default: 1.0). "
            "Error if used with a non-FA algorithm."
        ),
    )
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=None,
        metavar="A",
        help=(
            "FA random walk step size (default: 0.25). "
            "Error if used with a non-FA algorithm."
        ),
    )
    parser.add_argument(
        "--levy_exp",
        nargs="+",
        type=float,
        default=None,
        metavar="L",
        dest="levy_exp",
        help=(
            "FA Lévy exponent in [1, 2] (default: 1.5). "
            "Only used when --variant levy. "
            "Error if used with a non-FA algorithm."
        ),
    )
    parser.add_argument(
        "--sd_step",
        nargs="+",
        type=float,
        default=None,
        metavar="S",
        dest="sd_step",
        help=(
            "SD initial step length for backtracking line search "
            "(default: 0.1 · min(W, H)). "
            "Error if used with a non-SD algorithm."
        ),
    )
    parser.add_argument(
        "--sd_alpha",
        nargs="+",
        type=float,
        default=None,
        metavar="A",
        dest="sd_alpha",
        help=(
            "SD random perturbation amplitude added every iteration "
            "(default: 0.0 — pure steepest descent). "
            "Error if used with a non-SD algorithm."
        ),
    )
    parser.add_argument(
        "--sa_t0",
        nargs="+",
        type=float,
        default=None,
        metavar="T",
        dest="sa_t0",
        help=(
            "SA initial temperature (default: 0.3 · (f_max − f_min) of the "
            "search space). Error if used with a non-SA algorithm."
        ),
    )
    parser.add_argument(
        "--sa_cooling_rate",
        nargs="+",
        type=float,
        default=None,
        metavar="A",
        dest="sa_cooling_rate",
        help=(
            "SA geometric cooling factor per iteration (default: 0.95). "
            "Error if used with a non-SA algorithm."
        ),
    )
    parser.add_argument(
        "--sa_step",
        nargs="+",
        type=float,
        default=None,
        metavar="S",
        dest="sa_step",
        help=(
            "SA neighbour proposal step size (default: 0.1 · min(W, H)). "
            "Error if used with a non-SA algorithm."
        ),
    )
    parser.add_argument(
        "--no_gifsicle",
        action="store_true",
        default=False,
        dest="no_gifsicle",
        help=(
            "Disable gifsicle post-processing even if gifsicle is installed."
        ),
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        default=False,
        help=(
            "Append an info bar to the right of each frame showing "
            "the iteration number, current best height, and global minimum."
        ),
    )
    parser.add_argument(
        "--show_attractions",
        action="store_true",
        default=False,
        dest="show_attractions",
        help=(
            "Draw arrows from each agent toward its attraction points "
            "(pbest/gbest for PSO; brighter fireflies for FA). "
            "Arrow length encodes influence strength; colour encodes kind."
        ),
    )
    parser.add_argument(
        "--frames",
        action="store_true",
        default=False,
        help=(
            "Write each frame as a separate image instead of an animated "
            "GIF. Default format is JPEG (quality 85); use --png to "
            "produce lossless PNG files instead."
        ),
    )
    parser.add_argument(
        "--png",
        action="store_true",
        default=False,
        help=(
            "When used with --frames, write frames as PNG instead of JPEG."
        ),
    )
    parser.add_argument(
        "--output_prefix", "-o",
        default="out",
        metavar="PREFIX",
        dest="output_prefix",
        help="Output filename prefix (default: out)",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        metavar="DIR",
        dest="output_dir",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of parallel worker processes for batch runs "
            "(default: number of CPU cores). Use 1 to disable parallelism."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "Load parameters from a TOML file. "
            "Mutually exclusive with all other flags."
        ),
    )

    args = parser.parse_args()

    if args.config is not None:
        other: list[str] = []
        skip_next = False
        for token in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if token == "--config":
                skip_next = True
                continue
            if token.startswith("--config="):
                continue
            if token.startswith("-"):
                other.append(token)
        if other:
            parser.error(
                "--config cannot be combined with other flags: "
                + " ".join(other)
            )
        return load_config(args.config)  # returns (jobs, workers)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.png and not args.frames:
        parser.error("--png requires --frames")

    seeds: list[int | None] = (
        [None] if args.seed is None else args.seed
    )
    dot_sizes: list[int | None] = (
        [None] if args.dot_size is None else args.dot_size
    )
    inertias: list[float | None] = (
        [None] if args.inertia is None else args.inertia
    )
    variants: list[str | None] = (
        [None] if args.variant is None else args.variant
    )
    gammas: list[float | None] = (
        [None] if args.gamma is None else args.gamma
    )
    beta0s: list[float | None] = (
        [None] if args.beta0 is None else args.beta0
    )
    alphas: list[float | None] = (
        [None] if args.alpha is None else args.alpha
    )
    levy_exps: list[float | None] = (
        [None] if args.levy_exp is None else args.levy_exp
    )
    sd_steps: list[float | None] = (
        [None] if args.sd_step is None else args.sd_step
    )
    sd_alphas: list[float | None] = (
        [None] if args.sd_alpha is None else args.sd_alpha
    )
    sa_t0s: list[float | None] = (
        [None] if args.sa_t0 is None else args.sa_t0
    )
    sa_cooling_rates: list[float | None] = (
        [None] if args.sa_cooling_rate is None else args.sa_cooling_rate
    )
    sa_steps: list[float | None] = (
        [None] if args.sa_step is None else args.sa_step
    )

    workers: int = (
        args.workers if args.workers is not None
        else os.cpu_count() or 1
    )
    jobs = _build_jobs(
        dimensions=args.dimensions,
        seeds=seeds,
        algorithms=args.algorithm,
        spaces=args.space,
        n_agents_l=args.n_agents,
        iterations_l=args.iterations,
        ipf_l=args.iterations_per_frame,
        fps_l=args.fps,
        dot_sizes=dot_sizes,
        inertias=inertias,
        variants=variants,
        gammas=gammas,
        beta0s=beta0s,
        fa_alphas=alphas,
        levy_exps=levy_exps,
        sd_steps=sd_steps,
        sd_alphas=sd_alphas,
        sa_t0s=sa_t0s,
        sa_cooling_rates=sa_cooling_rates,
        sa_steps=sa_steps,
        detailed=args.detailed,
        show_attractions=args.show_attractions,
        frames=args.frames,
        frames_png=args.png,
        use_gifsicle=not args.no_gifsicle,
        output_prefix=args.output_prefix,
        output_dir=output_dir,
    )
    return jobs, workers
