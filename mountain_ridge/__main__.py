"""Entry point: parse CLI jobs and run each one."""

from pathlib import Path

from mountain_ridge.cli import JobConfig, parse_jobs
from mountain_ridge.gif_builder import build_gif
from mountain_ridge.search_space.functions import get_space
from mountain_ridge.swarm.base import Swarm
from mountain_ridge.swarm.algorithms.pso import PSOSwarm


_ALGORITHMS: dict[str, type[Swarm]] = {
    "pso": PSOSwarm,
}


def _next_output_path(output_dir: Path, prefix: str) -> Path:
    """Return the next available *prefix_NN.gif* path in *output_dir*."""
    n = 0
    while True:
        path = output_dir / f"{prefix}_{n:02d}.gif"
        if not path.exists():
            return path
        n += 1


def _run_job(job: JobConfig, output_path: Path) -> None:
    """Instantiate space and swarm, run simulation, write GIF."""
    space_fn, grid = get_space(job.space, job.seed, job.dimensions)

    swarm_cls = _ALGORITHMS.get(job.algorithm)
    if swarm_cls is None:
        raise ValueError(
            f"Unknown algorithm: {job.algorithm!r}. "
            f"Available: {list(_ALGORITHMS)}"
        )

    swarm = swarm_cls(  # type: ignore[call-arg]
        n_agents=job.n_agents,
        search_space=space_fn,
        dimensions=job.dimensions,
        seed=job.seed,
    )

    build_gif(
        swarm=swarm,
        grid=grid,
        n_iterations=job.iterations,
        iterations_per_frame=job.iterations_per_frame,
        fps=job.fps,
        output_path=output_path,
    )


def main() -> None:
    """Parse jobs and run each one."""
    jobs = parse_jobs()
    for job in jobs:
        output_path = _next_output_path(job.output_dir, job.output_prefix)
        print(f"Generating {output_path} ...")
        _run_job(job, output_path)
        print(f"  -> saved {output_path}")


if __name__ == "__main__":
    main()
