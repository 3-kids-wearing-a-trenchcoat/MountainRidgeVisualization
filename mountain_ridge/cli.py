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

    seeds: list[int | None] = (
        [None] if args.seed is None else args.seed
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
    ):
        dims, seed, algo, space, n_agents, iters, ipf, fps = combo
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
            output_prefix=args.output_prefix,
            output_dir=output_dir,
        ))

    return jobs
