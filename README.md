# MountainRidgeVisualization

A CLI tool that visualizes swarm intelligence optimization algorithms as
animated GIFs. Agents move across a procedurally generated height map,
searching for the lowest point (valley) according to the chosen algorithm.

## Output format

Each GIF shows the swarm searching the height map over time. One frame is
captured at the start and then every `--iterations-per-frame` steps.

### Height map

The background is a colour-mapped rendering of the search space. Colour encodes
height, running from low to high:

| Colour | Height |
|--------|--------|
| Dark navy | Lowest (valleys) |
| Blue | Low |
| Green | Mid |
| Brown | High |
| White | Highest (peaks) |

### Markers

| Marker | Meaning |
|--------|---------|
| White diamond | True global minimum — every grid cell whose height is exactly equal to the lowest value in the space. Static; does not move. Multiple diamonds appear when the space has more than one equally-deep minimum. |
| Yellow circle (outlined) | Best position found by the swarm so far. Updates as the swarm improves. |
| Coloured circles (outlined) | Current positions of all agents. Colour depends on the algorithm — see below. |

#### Agent colours by algorithm

**PSO** — all agents are drawn in solid red. The best agent also receives the yellow best-position circle on top.

**FA (Firefly Algorithm)** — each agent's colour reflects its brightness (the inverse of its height). The colour interpolates on a gradient between:
- **Yellow** `(255, 255, 0)` — brightest firefly (lowest height, best position)
- **Red** `(255, 0, 0)` — dimmest firefly (highest height, worst position)

The goal of each algorithm is to drive the yellow circle onto the white diamond.

## Requirements

- Python 3.14+
- Dependencies listed in `REQUIREMENTS.txt`

```
pip install -r REQUIREMENTS.txt
```

## Usage

```
python -m mountain_ridge [OPTIONS]
```

Each flag has a sensible default, so the simplest invocation produces one GIF
with all defaults:

```
python -m mountain_ridge
```

Output files are named `<prefix>_NN.gif` (e.g. `out_00.gif`) and the counter
increments automatically to avoid overwriting existing files.

### Flags

| Flag                     | Short | Type     | Default    | Description                                                                                                                                                                                                     |
|--------------------------|-------|----------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--dimensions`           | `-d`  | `WxH`    | `100x100`  | Width and height of the search space grid                                                                                                                                                                       |
| `--seed`                 | `-s`  | `int`    | *(random)* | Random seed for the search space and agents. Omit for a fresh random seed per job                                                                                                                               |
| `--algorithm`            | `-a`  | `string` | `pso`      | Optimization algorithm to use. Available: `pso`, `fa`  (Particle Swarm Optimization and Firefly Algorithm respectively)                                                                                         |
| `--space`                |       | `string` | `gaussian` | Search space function. Available: `gaussian`, `rastrigin`, `ackley`, `multiwell`, `deceptive`                                                                                                                   |
| `--n-agents`             | `-n`  | `int`    | `20`       | Number of agents in the swarm                                                                                                                                                                                   |
| `--iterations`           | `-i`  | `int`    | `100`      | Total number of simulation steps                                                                                                                                                                                |
| `--iterations-per-frame` |       | `int`    | `5`        | Simulation steps between captured frames                                                                                                                                                                        |
| `--fps`                  |       | `int`    | `10`       | Playback speed of the output GIF (frames per second)                                                                                                                                                            |
| `--inertia`              |       | `float`  | `0.0`      | **PSO only.** Inertia weight `w` — scales how much of the previous velocity carries over each iteration. Typical range `[0, 1]`. Error if used with a non-PSO algorithm.                                        |
| `--variant`              |       | `string` | `brownian` | **FA only.** Random walk distribution: `brownian` (bounded uniform step, `U(−0.5, 0.5)`) or `levy` (heavy-tailed Lévy flight, same typical scale as `brownian` but with occasional large jumps; better at escaping local minima). Error if used with a non-FA algorithm. |
| `--gamma`                |       | `float`  | *(auto)*   | **FA only.** Light absorption coefficient `γ` — controls how quickly attractiveness decays with distance. Higher values shorten attraction range and preserve diversity. Default scales with the grid: `1 / min(W, H)²`. Error if used with a non-FA algorithm. |
| `--beta0`                |       | `float`  | `1.0`      | **FA only.** Maximum attractiveness `β₀` at distance zero. Error if used with a non-FA algorithm.                                                                                                               |
| `--alpha`                |       | `float`  | *(auto)*   | **FA only.** Random walk step size `α`. Default scales with the grid: `0.05 · min(W, H)`. Error if used with a non-FA algorithm.                                                                                |
| `--levy-exp`             |       | `float`  | `1.5`      | **FA only.** Lévy exponent `λ` — controls tail weight of the Lévy distribution. Only used when `--variant levy`. Typical range `[1.0, 2.0]`. Error if used with a non-FA algorithm.                             |
| `--dot-size`             |       | `int`    | *(auto)*   | Agent dot radius in pixels. Omit to scale automatically: `max(2, round(min(W, H) / 35))`                                                                                                                        |
| `--detailed`             |       | flag     | off        | Append a statistics bar (150 px wide) to the right of every frame. See [Detailed output](#detailed-output).                                                                                                     |
| `--output-prefix`        | `-o`  | `string` | `out`      | Prefix for output filenames (`<prefix>_NN.gif`)                                                                                                                                                                 |
| `--output-dir`           |       | `path`   | `.`        | Directory to write output GIFs into (created if absent)                                                                                                                                                         |

### Detailed output

When `--detailed` is passed, a 150-pixel-wide statistics bar is appended to the
right edge of every frame. The bar shows:

| Field | Description |
|-------|-------------|
| **Iteration** | Current iteration number (0 = initial state) |
| **Best height** | Lowest height reached by any agent at any point so far (monotonically non-increasing) |
| **Best since** | Iteration at which that best height was first achieved |
| **Global min** | True minimum height of the search space |

The output GIF dimensions become `(W + 150) × H`.

### Batch mode

Any flag marked with a type above (except `--output-prefix` and `--output-dir`)
accepts multiple space-separated values. When one or more flags receive multiple
values, the program produces a GIF for **every combination** (cartesian product)
of the provided values.

**Example — three seeds:**
```
python -m mountain_ridge --seed 1 2 3
```
Produces `out_00.gif`, `out_01.gif`, `out_02.gif`, one per seed.

**Example — all combinations of seeds and agent counts:**
```
python -m mountain_ridge --seed 1 2 --n-agents 10 20
```
Produces 4 GIFs: `(seed=1, n=10)`, `(seed=1, n=20)`, `(seed=2, n=10)`,
`(seed=2, n=20)`.

In batch mode a nested progress bar is shown: the outer bar tracks overall job
progress and the inner bar tracks iterations within the current job.

### Examples

```bash
# Single GIF with a fixed seed and larger space
python -m mountain_ridge --seed 42 --dimensions 200x200 --iterations 200

# Compare PSO and FA on the same seed and space
python -m mountain_ridge --algorithm pso fa --seed 99 --space multiwell

# Batch: sweep over several seeds, save to a dedicated folder
python -m mountain_ridge --seed 10 20 30 40 --output-dir results --output-prefix sweep
```
