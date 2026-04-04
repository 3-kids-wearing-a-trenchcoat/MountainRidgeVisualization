# MountainRidgeVisualization

A CLI tool that visualizes swarm intelligence optimization algorithms as
animated GIFs or image series. Agents move across a procedurally generated
height map, searching for the lowest point (valley) according to the chosen
algorithm.

## Output format

By default, the program writes an animated GIF. Pass `--frames` to instead
write each captured frame as a separate image file (see
[Frames mode](#frames-mode)).

One frame is captured at the start and then every `--iterations_per_frame`
steps.

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
| Arrows (optional) | When `--show_attractions` is enabled, arrows are drawn from each agent toward its attraction points. See [Attraction arrows](#attraction-arrows). |

#### Agent colours by algorithm

**PSO** — all agents are drawn in solid red. The best agent also receives the yellow best-position circle on top.

**FA (Firefly Algorithm)** — each agent's colour reflects its brightness (the inverse of its height). The colour interpolates on a gradient between:
- **Yellow** `(255, 255, 0)` — brightest firefly (lowest height, best position)
- **Red** `(255, 0, 0)` — dimmest firefly (highest height, worst position)

**SD (Steepest Descent)** — same score-based colour gradient as FA. Because agents are fully independent, colour shows at a glance which agents have converged to good local minima (yellow) versus which are still high up or stuck on a plateau (red).

**SA (Simulated Annealing)** — same score-based colour gradient as FA and SD. Each agent cools independently, so colour reflects how well each parallel run has settled: yellow agents have found low-scoring positions, red agents are still exploring or stuck at high elevation.

The goal of each algorithm is to drive the yellow circle onto the white diamond.

### Attraction arrows

Pass `--show_attractions` to draw arrows from each agent toward the positions
that are currently influencing its movement. Arrows are rendered beneath the
agent dots so agents always appear on top.

**Arrow length** is proportional to the strength of that influence. All
arrows within the same algorithm share a single fixed reference scale, so
the same force always produces the same arrow size regardless of frame or
what other agents are doing:

- **PSO** — all three force types are velocity contributions and are
  normalised by `v_max` (fixed at init). A full-length arrow means that
  force alone would push the particle to the velocity cap. Cognitive and
  social arrows saturate at full length when the attractor is farther than
  `v_max / c` away, which correctly signals that the position-based pull is
  at its effective maximum.
- **FA** — length encodes `β / β₀`, the fraction of maximum attractiveness.
  `β₀` is fixed at init, so the scale never shifts. Arrows shorten as the
  attractor grows more distant (controlled by `--gamma`). Attractors whose
  influence falls below 1% (`β / β₀ < 0.01`) are omitted to reduce clutter.
- **SD** — length encodes the local gradient magnitude normalised by the
  steepest gradient found across all agents in the current frame. The agent
  on the steepest slope always gets a full-length arrow; all others are shown
  relative to it. Agents at (or near) a local minimum show no arrow.

**Arrow colour** encodes the kind of influence:

| Colour | Meaning |
|--------|---------|
| Cyan `(0, 200, 255)` | PSO — cognitive pull toward the agent's personal best (pbest) |
| Magenta `(220, 0, 220)` | PSO — social pull toward the global best (gbest) |
| Yellow `(255, 220, 0)` | PSO — inertia (current velocity scaled by `w`; the momentum that will carry into the next step) |
| Orange `(255, 140, 0)` | FA — attraction toward a brighter (lower-scoring) firefly |
| Green `(0, 220, 80)` | SD — direction of steepest descent (negative gradient) |

All arrows show the forces that will act on each agent in the **next**
iteration, not the forces that produced the current position.

Frame 0 (the initial state before any update) shows arrows for SD and FA
(computed from the initial positions/gradient); PSO shows no inertia arrow on
Frame 0 because the initial velocity is zero.

## Requirements

- Python 3.14+
- Dependencies listed in `REQUIREMENTS.txt`

```
pip install -r REQUIREMENTS.txt
```

### Optional: gifsicle

If [`gifsicle`](https://www.lcdf.org/gifsicle/) is installed and available on
your `PATH`, it is invoked automatically after each GIF is written to apply
delta-frame optimisation (`-O3`). Because the terrain background is static
across all frames, this typically reduces file size by 50–80% with no change
to visual quality or frame count.

If `gifsicle` is not found, a warning is printed once and the program
continues normally, producing a valid (unoptimised) GIF. Pass `--no_gifsicle`
to disable gifsicle entirely and suppress the warning.

## Usage

```
python -m mountain_ridge [OPTIONS]
```

Each flag has a sensible default, so the simplest invocation produces one GIF
with all defaults:

```
python -m mountain_ridge
```

In GIF mode, output files are named `<prefix>_NN.gif` (e.g. `out_00.gif`).
In frames mode, a subdirectory `<prefix>_NN/` is created and frames are
written into it as `frame_0000.jpeg`, `frame_0001.jpeg`, etc.
The counter `NN` increments automatically to avoid overwriting existing files.

### Flags

| Flag                     | Short | Type     | Default    | Description                                                                                                                                                                                                     |
|--------------------------|-------|----------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--dimensions`           | `-d`  | `WxH`    | `100x100`  | Width and height of the search space grid                                                                                                                                                                       |
| `--seed`                 | `-s`  | `int`    | *(random)* | Random seed for the search space and agents. Omit for a fresh random seed per job                                                                                                                               |
| `--algorithm`            | `-a`  | `string` | `pso`      | Optimization algorithm to use. Available: `pso`, `fa`, `sd`, `sa` (Particle Swarm Optimization, Firefly Algorithm, Steepest Descent, and Simulated Annealing respectively)                                     |
| `--space`                |       | `string` | `ridge`    | Search space function. Available: `ridge`, `gaussian`, `rastrigin`, `ackley`, `multiwell`, `deceptive`                                                                                                          |
| `--n_agents`             | `-n`  | `int`    | `20`       | Number of agents in the swarm                                                                                                                                                                                   |
| `--iterations`           | `-i`  | `int`    | `100`      | Total number of simulation steps                                                                                                                                                                                |
| `--iterations_per_frame` |       | `int`    | `5`        | Simulation steps between captured frames                                                                                                                                                                        |
| `--fps`                  |       | `int`    | `10`       | Playback speed of the output GIF (frames per second)                                                                                                                                                            |
| `--inertia`              |       | `float`  | `0.0`      | **PSO only.** Inertia weight `w` — scales how much of the previous velocity carries over each iteration. Typical range `[0, 1]`. Only applied to PSO jobs; ignored for other algorithms.                                        |
| `--variant`              |       | `string` | `brownian` | **FA only.** Random walk distribution: `brownian` (bounded uniform step, `U(−0.5, 0.5)`) or `levy` (heavy-tailed Lévy flight, same typical scale as `brownian` but with occasional large jumps; better at escaping local minima). Only applied to FA jobs; ignored for other algorithms. |
| `--gamma`                |       | `float`  | *(auto)*   | **FA only.** Light absorption coefficient `γ` — controls how quickly attractiveness decays with distance. Higher values shorten attraction range and preserve diversity. Default scales with the grid: `1 / min(W, H)²`. Only applied to FA jobs; ignored for other algorithms. |
| `--beta0`                |       | `float`  | `1.0`      | **FA only.** Maximum attractiveness `β₀` at distance zero. Only applied to FA jobs; ignored for other algorithms.                                                                                                               |
| `--alpha`                |       | `float`  | *(auto)*   | **FA only.** Random walk step size `α`. Default scales with the grid: `0.05 · min(W, H)`. Only applied to FA jobs; ignored for other algorithms.                                                                                |
| `--levy_exp`             |       | `float`  | `1.5`      | **FA only.** Lévy exponent `λ` — controls tail weight of the Lévy distribution. Only used when `--variant levy`. Typical range `[1.0, 2.0]`. Only applied to FA jobs; ignored for other algorithms.                             |
| `--sd_step`              |       | `float`  | *(auto)*   | **SD only.** Initial step length for backtracking line search. Default scales with the grid: `0.1 · min(W, H)`. Only applied to SD jobs; ignored for other algorithms.                                                          |
| `--sd_alpha`             |       | `float`  | `0.0`      | **SD only.** Random perturbation amplitude `α` added unconditionally every iteration. `0.0` gives strictly deterministic steepest descent. Only applied to SD jobs; ignored for other algorithms.                               |
| `--sa_t0`                |       | `float`  | *(auto)*   | **SA only.** Initial temperature `T₀`. Default: `0.3 · (f_max − f_min)` of the search space. Only applied to SA jobs; ignored for other algorithms.                                                                            |
| `--sa_cooling_rate`      |       | `float`  | `0.95`     | **SA only.** Geometric cooling factor `α` applied each iteration: `T ← α · T`. Only applied to SA jobs; ignored for other algorithms.                                                                                          |
| `--sa_step`              |       | `float`  | *(auto)*   | **SA only.** Half-width of the uniform neighbour proposal step. Default scales with the grid: `0.1 · min(W, H)`. Only applied to SA jobs; ignored for other algorithms.                                                         |
| `--dot_size`             |       | `int`    | *(auto)*   | Agent dot radius in pixels. Omit to scale automatically: `max(2, round(min(W, H) / 35))`                                                                                                                        |
| `--show_attractions`     |       | flag     | off        | Draw arrows from each agent toward its attraction points. Arrow length encodes influence strength; colour encodes kind. See [Attraction arrows](#attraction-arrows).                                              |
| `--detailed`             |       | flag     | off        | Append a statistics bar (150 px wide) to the right of every frame. See [Detailed output](#detailed-output).                                                                                                     |
| `--frames`               |       | flag     | off        | Write each frame as a separate image file instead of an animated GIF. See [Frames mode](#frames-mode).                                                                                                          |
| `--png`                  |       | flag     | off        | **Requires `--frames`.** Write frames as lossless PNG instead of JPEG. Error if used without `--frames`.                                                                                                        |
| `--no_gifsicle`          |       | flag     | off        | Disable `gifsicle` post-processing and suppress the "not found" warning.                                                                                                                                        |
| `--output_prefix`        | `-o`  | `string` | `out`      | Prefix for output filenames (`<prefix>_NN.gif` or `<prefix>_NN/`)                                                                                                                                              |
| `--output_dir`           |       | `path`   | `.`        | Directory to write output into (created if absent)                                                                                                                                                              |
| `--config`               |       | `path`   | *(none)*   | Load all parameters from a TOML file. Mutually exclusive with all other flags. See [Config file](#config-file).                                                                                                 |

### Available search spaces

| Name | Description |
|------|-------------|
| `ridge` *(default)* | Realistic mountain terrain generated by spectral synthesis. White noise is filtered in the frequency domain by a `1/f^β` power-law envelope, producing fractal terrain with ridges, valleys, saddle points, and plateaus at multiple scales. A random directional bias creates a dominant ridge orientation. |
| `gaussian` | A sum of random Gaussians with random amplitudes and widths. Produces a varied, organic landscape with many local minima. |
| `rastrigin` | A parabolic bowl perturbed by a cosine grid. Produces a large number of evenly-spaced local minima with one global minimum near the centre. |
| `ackley` | A near-flat surface covered by a dense cosine carpet with a single sharp global minimum. Difficult because the flat carpet gives little gradient information. |
| `multiwell` | Several identical Gaussian wells arranged in a ring, all equally deep. Designed to expose an algorithm's tendency to commit to one well and ignore equally-good alternatives. |
| `deceptive` | A wide, shallow basin near the centre (easy to find but not the true minimum) with one or two narrow, deeper pits in the periphery (the true global minimum). Designed to trap algorithms that converge early. |

### Steepest Descent (SD)

SD is a classical single-agent local search heuristic included as a baseline
for comparison with swarm algorithms. Each agent independently follows the
direction of steepest descent — the negative gradient of the height map —
until it reaches a local minimum, where it stalls.

**Gradient estimation** — because the height map is accessible only through
point queries, the gradient is estimated by central finite differences with a
fixed perturbation of 1 grid unit.

**Step size** — each iteration uses backtracking line search: the agent
attempts a step of size `--sd_step` in the descent direction and halves it
repeatedly (by factor 0.5) until either a strict height decrease is achieved
or the step falls below a minimum threshold (`1e-4`), at which point the
move is cancelled and the agent stays put (local minimum).

**Random perturbation** — `--sd_alpha` adds an unconditional random
displacement `α · U(−0.5, 0.5)²` every iteration, regardless of whether the
agent is at a local minimum. When `α = 0` (the default) the algorithm is
strictly deterministic. Increasing `α` gives agents a chance to escape
shallow minima, trading accuracy for exploration.

**No communication** — agents share no memory. The yellow best-position
marker tracks the lowest score found by any individual agent's personal best.

### Simulated Annealing (SA)

SA is a probabilistic local search heuristic included as a baseline for
comparison with swarm algorithms. Multiple agents are run simultaneously as
fully independent parallel searches from random starting positions — identical
in structure to SD, but with a probabilistic acceptance rule instead of a
strict descent rule.

Each iteration an agent proposes a random **neighbour** position by adding a
uniform displacement `α · U(−0.5, 0.5)²` (scaled by `--sa_step`) to its
current position. The move is always accepted if it improves the objective.
Worsening moves are accepted with probability `exp(−Δf / T)`, where `T` is the
current temperature and `Δf` is the increase in height. After each iteration
the temperature cools geometrically: `T ← cooling_rate · T`, flooring at
`1e-4`.

At high temperature the agent accepts almost any move, enabling broad
exploration. As temperature falls, the agent becomes increasingly selective and
converges toward the best basin it has found.

**Initial temperature** — The default `T₀ = 0.3 · (f_max − f_min)` is derived
from the full height range of the search space, so that at the start a move
worsening the objective by 30% of the total range is still accepted with
probability `e⁻¹ ≈ 37%`.

**No communication** — agents share no memory. The yellow best-position marker
tracks the lowest score found by any individual agent's personal best.

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

### Frames mode

Pass `--frames` to write each captured frame as an individual image file
instead of assembling them into a GIF.

**Output layout:**

```
<output-dir>/
  <prefix>_00/
    frame_0000.jpeg
    frame_0001.jpeg
    ...
```

**Format:** JPEG at quality 85 by default. Across all built-in search spaces
and dimensions, JPEG is consistently ~4× smaller than PNG for this type of
content (smooth terrain gradients with small solid-colour agent markers).
Pass `--png` to write lossless PNG files instead. `--png` requires `--frames`
and will produce an error if used alone.

`--fps` and `--no_gifsicle` have no effect in frames mode.

**Example:**
```bash
# Write frames as JPEG (default)
python -m mountain_ridge --frames --seed 42 --iterations 100

# Write frames as PNG
python -m mountain_ridge --frames --png --seed 42 --iterations 100
```

### Batch mode

Any flag marked with a type above (except `--output_prefix` and `--output_dir`)
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
python -m mountain_ridge --seed 1 2 --n_agents 10 20
```
Produces 4 GIFs: `(seed=1, n=10)`, `(seed=1, n=20)`, `(seed=2, n=10)`,
`(seed=2, n=20)`.

**Algorithm-specific parameters** (`--inertia`, `--variant`, `--gamma`, etc.)
are an exception to the global Cartesian product: they expand only within
their own algorithm's jobs and are ignored for every other algorithm. When
batching multiple algorithms, each algorithm is independently crossed with
its own specific parameters, and the results are concatenated. For example,
`--algorithm pso fa --inertia 0.3 0.6 --variant brownian levy --n_agents 10 20`
produces 4 PSO jobs (2 inertia × 2 n_agents) plus 4 FA jobs
(2 variants × 2 n_agents) = 8 jobs total.

In batch mode a nested progress bar is shown: the outer bar tracks overall job
progress and the inner bar tracks iterations within the current job.

### Config file

Pass `--config path/to/file.toml` to load all parameters from a TOML file
instead of the command line. `--config` is mutually exclusive with all other
flags.

The file supports every parameter listed in [Flags](#flags), using the same
names without the leading `--` (e.g. `--n_agents` → `n_agents`).
Algorithm-specific parameters belong in a sub-table named after the algorithm
(`[pso]`, `[fa]`, `[sd]`, `[sa]`).

**Batch mode** works the same way as on the CLI: any parameter can be a list.
A single value produces one job; a list produces one job per value; multiple
lists produce the Cartesian product. An empty list `[]` or a missing key both
mean "use the default". `seed = []` (or omitting `seed`) assigns one random
seed per job.

The following keys are **not** lists and apply to all jobs uniformly:
`output_prefix`, `output_dir`, `detailed`, `show_attractions`, `frames`,
`png`, `no_gifsicle`.

**Example:**

```toml
output_prefix = "sweep"
output_dir    = "results/"
detailed      = true

dimensions = ["200x200"]
algorithm  = ["pso", "fa"]   # two algorithms → one job each (2 total)
seed       = [1, 2, 3]       # × 3 seeds → 6 jobs total
n_agents   = [20]

[fa]
variant = ["levy"]
```

### Examples

```bash
# Single GIF with a fixed seed and larger space
python -m mountain_ridge --seed 42 --dimensions 200x200 --iterations 200

# Compare PSO and FA on the same seed and space
python -m mountain_ridge --algorithm pso fa --seed 99 --space multiwell

# Batch: sweep over several seeds, save to a dedicated folder
python -m mountain_ridge --seed 10 20 30 40 --output_dir results --output_prefix sweep

# Show attraction arrows (PSO: cyan=pbest, magenta=gbest)
python -m mountain_ridge --show_attractions --seed 42 --dimensions 300x300

# Show attraction arrows for FA
python -m mountain_ridge --algorithm fa --show_attractions --seed 42 --dimensions 300x300

# Steepest descent (pure, no randomness)
python -m mountain_ridge --algorithm sd --seed 42 --dimensions 200x200

# Steepest descent with random perturbation to escape shallow minima
python -m mountain_ridge --algorithm sd --sd_alpha 2.0 --seed 42 --dimensions 200x200

# SD with gradient arrows — shows each agent's descent direction and slope intensity
python -m mountain_ridge --algorithm sd --show_attractions --seed 42 --dimensions 300x300

# Side-by-side comparison: SD vs PSO on the same seed
python -m mountain_ridge --algorithm sd pso --seed 42 --dimensions 200x200

# Simulated annealing (default parameters)
python -m mountain_ridge --algorithm sa --seed 42 --dimensions 200x200

# SA with a slower cooling schedule to explore longer
python -m mountain_ridge --algorithm sa --sa_cooling_rate 0.99 --seed 42 --dimensions 200x200

# Side-by-side comparison: SA vs SD on the same seed
python -m mountain_ridge --algorithm sa sd --seed 42 --dimensions 200x200
```
