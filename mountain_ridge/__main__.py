"""Entry point: parse CLI jobs and run each one."""

import multiprocessing
import os
import queue as _queue
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, wait as _futures_wait
from pathlib import Path

from tqdm import tqdm

from mountain_ridge.cli import parse_jobs
from mountain_ridge.config_file import JobConfig
from mountain_ridge.gif_builder import build_frames, build_gif
from mountain_ridge.search_space.functions import get_space
from mountain_ridge.swarm.base import Swarm
from mountain_ridge.swarm.algorithms.fa import FASwarm
from mountain_ridge.swarm.algorithms.pso import PSOSwarm
from mountain_ridge.swarm.algorithms.sa import SASwarm
from mountain_ridge.swarm.algorithms.sd import SDSwarm


_ALGORITHMS: dict[str, type[Swarm]] = {
    "fa": FASwarm,
    "pso": PSOSwarm,
    "sa": SASwarm,
    "sd": SDSwarm,
}

# Shared progress queue for parallel runs.  Set in main() before the pool
# is created so forked workers inherit it without pickling.
_progress_queue: multiprocessing.Queue | None = None  # type: ignore[type-arg]


def _next_output_path(output_dir: Path, prefix: str) -> Path:
    """Return the next available *prefix_NN.gif* path in *output_dir*."""
    n = 0
    while True:
        path = output_dir / f"{prefix}_{n:02d}.gif"
        if not path.exists():
            return path
        n += 1


def _next_frames_dir(output_dir: Path, prefix: str) -> Path:
    """Return the next available *prefix_NN/* subdirectory in *output_dir*."""
    n = 0
    while True:
        path = output_dir / f"{prefix}_{n:02d}"
        if not path.exists():
            return path
        n += 1


def _run_job(
    job: JobConfig,
    output_path: Path,
    use_gifsicle: bool,
    progress_position: int | None = 0,
    job_id: int = 0,
) -> int | None:
    """Instantiate space and swarm, run simulation, write output.

    Returns the number of frames written when in frames mode, else None.
    """
    on_progress: Callable[[int, int], None] | None = None
    if _progress_queue is not None:
        _pid = os.getpid()
        _desc = output_path.name
        _every = max(1, job.iterations_per_frame)

        def on_progress(i: int, total: int) -> None:
            if i == 0 or i % _every == 0 or i == total:
                _progress_queue.put((_pid, job_id, i, total, _desc))

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
        noise=job.noise,
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
    if job.algorithm == "sd":
        if job.sd_step is not None:
            kwargs["step"] = job.sd_step
        if job.sd_alpha is not None:
            kwargs["alpha"] = job.sd_alpha
    if job.algorithm == "sa":
        kwargs["t0"] = (
            job.sa_t0 if job.sa_t0 is not None
            else 0.3 * float(grid.max() - grid.min())
        )
        if job.sa_cooling_rate is not None:
            kwargs["cooling_rate"] = job.sa_cooling_rate
        if job.sa_step is not None:
            kwargs["step_size"] = job.sa_step
    swarm = swarm_cls(**kwargs)  # type: ignore[call-arg]

    show_best = job.algorithm not in ("sa", "sd")

    if job.frames:
        output_path.mkdir(parents=True, exist_ok=True)
        return build_frames(
            swarm=swarm,
            grid=grid,
            n_iterations=job.iterations,
            iterations_per_frame=job.iterations_per_frame,
            output_dir=output_path,
            dot_size=job.dot_size,
            colour_by_score=(job.algorithm in ("fa", "sa", "sd")),
            detailed=job.detailed,
            use_png=job.frames_png,
            show_attractions=job.show_attractions,
            show_best=show_best,
            desc=output_path.name,
            progress_position=progress_position,
            on_progress=on_progress,
        )

    build_gif(
        swarm=swarm,
        grid=grid,
        n_iterations=job.iterations,
        iterations_per_frame=job.iterations_per_frame,
        fps=job.fps,
        output_path=output_path,
        dot_size=job.dot_size,
        colour_by_score=(job.algorithm in ("fa", "sa", "sd")),
        detailed=job.detailed,
        use_gifsicle=use_gifsicle,
        show_attractions=job.show_attractions,
        show_best=show_best,
        desc=output_path.name,
        progress_position=progress_position,
        on_progress=on_progress,
    )
    return None


def _assign_output_paths(jobs: list[JobConfig]) -> list[Path]:
    """Pre-assign one output path per job in the main process.

    Creating placeholder files/directories here prevents filename races
    when jobs run in parallel.
    """
    paths: list[Path] = []
    for job in jobs:
        if job.frames:
            path = _next_frames_dir(job.output_dir, job.output_prefix)
            path.mkdir(parents=True, exist_ok=True)
        else:
            path = _next_output_path(job.output_dir, job.output_prefix)
            path.touch()
        paths.append(path)
    return paths


def main() -> None:
    """Parse jobs and run each one."""
    jobs, workers = parse_jobs()
    batch = len(jobs) > 1

    use_gifsicle = jobs[0].use_gifsicle
    if use_gifsicle and shutil.which("gifsicle") is None:
        tqdm.write(
            "Warning: gifsicle not found on PATH — "
            "output GIFs will not be compressed. "
            "Install gifsicle or pass --no-gifsicle to suppress this warning."
        )
        use_gifsicle = False

    output_paths = _assign_output_paths(jobs)

    if workers == 1 or not batch:
        # ── Serial path (original behaviour) ─────────────────────────────
        outer: tqdm[tuple[JobConfig, Path]] = tqdm(
            list(zip(jobs, output_paths)),
            desc="Batch",
            unit="gif",
            position=0,
            leave=True,
            dynamic_ncols=True,
            disable=not batch,
        )
        with outer:
            for job, output_path in outer:
                if job.frames:
                    tqdm.write(
                        f"Generating frames in {output_path}/"
                        f" (seed={job.seed}) ..."
                    )
                else:
                    tqdm.write(
                        f"Generating {output_path} (seed={job.seed}) ..."
                    )
                n_frames = _run_job(
                    job,
                    output_path,
                    use_gifsicle=use_gifsicle,
                    progress_position=1 if batch else 0,
                )
                if job.frames:
                    tqdm.write(
                        f"  -> saved {n_frames} frames to {output_path}/"
                    )
                else:
                    tqdm.write(f"  -> saved {output_path}")
    else:
        # ── Parallel path ─────────────────────────────────────────────────
        tqdm.write(
            f"Running {len(jobs)} jobs across {workers} worker(s) ..."
        )

        # Per-worker progress bars live at positions 1..N; the summary
        # bar is at position 0.  Bars are created lazily on first contact
        # from each worker PID and reused if that worker runs a second job.
        global _progress_queue
        q: multiprocessing.Queue = multiprocessing.Queue()  # type: ignore[type-arg]
        _progress_queue = q  # inherited by forked workers; not pickled
        pid_to_bar: dict[int, tqdm] = {}
        pid_to_job_id: dict[int, int] = {}
        job_id_to_pid: dict[int, int] = {}
        next_pos: list[int] = [1]  # mutable so the closure can increment it

        def _get_bar(
            pid: int, job_id: int, total: int, desc: str
        ) -> tqdm:
            if pid not in pid_to_bar:
                bar: tqdm = tqdm(
                    total=total,
                    desc=desc,
                    unit="iter",
                    position=next_pos[0],
                    leave=True,
                    dynamic_ncols=True,
                )
                next_pos[0] += 1
                pid_to_bar[pid] = bar
            elif pid_to_job_id.get(pid) != job_id:
                # Same worker process, new job — reset the existing bar.
                pid_to_bar[pid].reset(total=total)
                pid_to_bar[pid].set_description(desc)
            pid_to_job_id[pid] = job_id
            job_id_to_pid[job_id] = pid
            return pid_to_bar[pid]

        def _drain_queue() -> None:
            while True:
                try:
                    pid, jid, i, total, desc = q.get_nowait()
                except _queue.Empty:
                    break
                bar = _get_bar(pid, jid, total, desc)
                bar.n = i
                bar.refresh()

        summary: tqdm[None] = tqdm(
            total=len(jobs),
            desc="Batch",
            unit="gif",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )
        _ctx = multiprocessing.get_context("fork")
        with summary, ProcessPoolExecutor(
            max_workers=workers, mp_context=_ctx
        ) as pool:
            future_to_info: dict = {
                pool.submit(
                    _run_job, job, path, use_gifsicle, None, jid
                ): (job, path, jid)
                for jid, (job, path) in enumerate(zip(jobs, output_paths))
            }
            pending = set(future_to_info)
            while True:
                _drain_queue()
                if not pending:
                    break
                done, pending = _futures_wait(pending, timeout=0.1)
                _drain_queue()
                for future in done:
                    job, output_path, jid = future_to_info[future]
                    n_frames = future.result()
                    # Force the bar for this job to 100 %.
                    pid = job_id_to_pid.get(jid)
                    if pid is not None and pid in pid_to_bar:
                        b = pid_to_bar[pid]
                        if b.total and b.n != b.total:
                            b.n = b.total
                            b.refresh()
                    if job.frames:
                        tqdm.write(
                            f"  -> saved {n_frames} frames"
                            f" to {output_path}/"
                        )
                    else:
                        tqdm.write(f"  -> saved {output_path}")
                    summary.update(1)
        # Close all per-worker bars.
        for b in pid_to_bar.values():
            b.close()


if __name__ == "__main__":
    main()
