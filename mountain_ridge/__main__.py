"""Entry point: parse CLI jobs and run each one."""

import shutil
from pathlib import Path

from tqdm import tqdm

from mountain_ridge.cli import JobConfig, parse_jobs
from mountain_ridge.gif_builder import build_gif
from mountain_ridge.search_space.functions import get_space
from mountain_ridge.swarm.base import Swarm
from mountain_ridge.swarm.algorithms.fa import FASwarm
from mountain_ridge.swarm.algorithms.pso import PSOSwarm


_ALGORITHMS: dict[str, type[Swarm]] = {
    "fa": FASwarm,
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


def _run_job(
    job: JobConfig,
    output_path: Path,
    use_gifsicle: bool,
    progress_position: int = 0,
) -> None:
    """Instantiate space and swarm, run simulation, write GIF."""
    space_fn, grid = get_space(job.space, job.seed, job.dimensions)

    swarm_cls = _ALGORITHMS.get(job.algorithm)
    if swarm_cls is None:
        raise ValueError(
            f"Unknown algorithm: {job.algorithm!r}. "
            f"Available: {list(_ALGORITHMS)}"
        )

    kwargs: dict[str, object] = dict(
        n_agents=job.n_agents,
        search_space=space_fn,
        dimensions=job.dimensions,
        seed=job.seed,
    )
    if job.algorithm == "pso" and job.inertia is not None:
        kwargs["w"] = job.inertia
    if job.algorithm == "fa":
        if job.beta0 is not None:
            kwargs["beta0"] = job.beta0
        if job.gamma is not None:
            kwargs["gamma"] = job.gamma
        if job.alpha is not None:
            kwargs["alpha"] = job.alpha
        if job.variant is not None:
            kwargs["variant"] = job.variant
        if job.levy_exp is not None:
            kwargs["levy_exp"] = job.levy_exp
    swarm = swarm_cls(**kwargs)  # type: ignore[call-arg]

    build_gif(
        swarm=swarm,
        grid=grid,
        n_iterations=job.iterations,
        iterations_per_frame=job.iterations_per_frame,
        fps=job.fps,
        output_path=output_path,
        dot_size=job.dot_size,
        colour_by_score=(job.algorithm == "fa"),
        detailed=job.detailed,
        use_gifsicle=use_gifsicle,
        desc=output_path.name,
        progress_position=progress_position,
    )


def main() -> None:
    """Parse jobs and run each one."""
    jobs = parse_jobs()
    batch = len(jobs) > 1

    use_gifsicle = jobs[0].use_gifsicle
    if use_gifsicle and shutil.which("gifsicle") is None:
        tqdm.write(
            "Warning: gifsicle not found on PATH — "
            "output GIFs will not be compressed. "
            "Install gifsicle or pass --no-gifsicle to suppress this warning."
        )
        use_gifsicle = False

    outer: tqdm[JobConfig] = tqdm(
        jobs,
        desc="Batch",
        unit="gif",
        position=0,
        leave=True,
        dynamic_ncols=True,
        disable=not batch,
    )
    with outer:
        for job in outer:
            output_path = _next_output_path(
                job.output_dir, job.output_prefix
            )
            tqdm.write(f"Generating {output_path} ...")
            _run_job(
                job,
                output_path,
                use_gifsicle=use_gifsicle,
                progress_position=1 if batch else 0,
            )
            tqdm.write(f"  -> saved {output_path}")


if __name__ == "__main__":
    main()
