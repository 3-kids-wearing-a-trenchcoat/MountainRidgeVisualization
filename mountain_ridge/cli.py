"""CLI argument parsing and batch job generation."""

import argparse
import itertools
import random
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
    detailed: bool
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


def parse_jobs() -> list[JobConfig]:
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
        default=["gaussian"],
        metavar="SPACE",
        help="Search space function (default: gaussian)",
    )
    parser.add_argument(
        "--n-agents", "-n",
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
        "--iterations-per-frame",
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
        "--dot-size",
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
        "--levy-exp",
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
        "--detailed",
        action="store_true",
        default=False,
        help=(
            "Append an info bar to the right of each frame showing "
            "the iteration number, current best height, and global minimum."
        ),
    )
    parser.add_argument(
        "--output-prefix", "-o",
        default="out",
        metavar="PREFIX",
        dest="output_prefix",
        help="Output filename prefix (default: out)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        metavar="DIR",
        dest="output_dir",
        help="Output directory (default: current directory)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.inertia is not None:
        bad = [a for a in args.algorithm if a != "pso"]
        if bad:
            parser.error(
                f"--inertia is only supported with --algorithm pso "
                f"(got: {bad})"
            )

    fa_flags_given = [
        name for name, val in [
            ("--variant", args.variant),
            ("--gamma",   args.gamma),
            ("--beta0",   args.beta0),
            ("--alpha",   args.alpha),
            ("--levy-exp", args.levy_exp),
        ] if val is not None
    ]
    if fa_flags_given:
        bad = [a for a in args.algorithm if a != "fa"]
        if bad:
            parser.error(
                f"{', '.join(fa_flags_given)} only supported with "
                f"--algorithm fa (got: {bad})"
            )

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

    jobs: list[JobConfig] = []
    for combo in itertools.product(
        args.dimensions,
        seeds,
        args.algorithm,
        args.space,
        args.n_agents,
        args.iterations,
        args.iterations_per_frame,
        args.fps,
        dot_sizes,
        inertias,
        variants,
        gammas,
        beta0s,
        alphas,
        levy_exps,
    ):
        (
            dims, seed, algo, space, n_agents,
            iters, ipf, fps, dot, w,
            variant, gamma, beta0, alpha, levy_exp,
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
            detailed=args.detailed,
            output_prefix=args.output_prefix,
            output_dir=output_dir,
        ))

    return jobs
