# NEREUS Solvers and Structure test suites

These are the regression tests for NEREUS. They are development tools and are not part of the program, and the program does not need them to build or run.

**Written with Claude Opus 5.5 (Anthropic).** The suites, fixtures and runner were written and maintained with Claude
Opus 5.5 then reviewed and run by human programmers in full. 

## Running them

```bash
tests/run_all.sh                    # build, then every suite and the headless check (about 15 minutes)
tests/run_all.sh flip units         # only the named suites
tests/run_all.sh flip "3D"          # one suite, only the cases whose label contains "3D"
NVIDIA_GPU=0 tests/run_all.sh       # force a CPU-only build (any Makefile variable passes through)
KEEP_WORK=1 tests/run_all.sh io     # keep tests/work/run, the files the cases wrote
```

The runner prints `ALL SUITES PASSED` and exits 0 on success, and otherwise lists the failed suites and exits 1.

## Self-contained

The tests need only the source tree: `src/`, the libraries in `include/`, and the project `Makefile`.

- `run_all.sh` builds NEREUS from `src/` into `tests/work/build` (objects) and `tests/work/bin` (the headless program
  and the suite binaries). It uses the project Makefile with `BUILD_DIR` and `BIN_DIR` set, so the flags match the
  program's. The project's own `build/` and `bin/` are not touched.
- Every case runs in `tests/work/run`. That directory holds a copy of `configs/solver_pressure_config.txt` and a link
  back to this folder, so a fixture is read as `tests/scenes/...`. Everything a case writes (configs, scenes,
  checkpoints, exports) lands there and is deleted at the end.
- `tests/work/` is git-ignored. The objects stay between runs, so a second run only recompiles what changed.
- The one read outside `tests/` and `src/`: the setup-screen suite loads every preset in the project's `configs/`
  and checks that each one validates.

The suites link CUDA and AMGX only when the build has the GPU pressure solver. The Makefile decides that
(`NVIDIA_GPU=auto`), and the runner reads the decision off the build's mode stamp.

## Layout

| file | what it is |
|---|---|
| `run_all.sh` | the runner: build, then the suites, then the headless check |
| `run_headless_check.sh` | drives the built `nereus_headless` program (called by `run_all.sh` with the `io` suite) |
| `suite_<name>.cpp` | one suite = one binary. Each case is a former stand-alone test inside its own namespace. |
| `SuiteMain.h` | the shared driver: runs each case in a child process of its own, so no global state carries from one case to the next |
| `TestMethods.h` | registers every shipped method, as `main()` does, plus `LoadReservoirScene` (a dam-break liquid region) |
| `Functions.h` | the old hard-coded cases (cavity, obstacle, dam break) as boundary and mask functions, kept as fixtures |
| `exact_solutions.h` | closed-form solutions to score against: Taylor-Green 2D, Ethier-Steinman 3D |
| `configs/headless_base.cfg` | the config the headless check runs |
| `configs/solver_pressure_config.txt` | the pressure-solver settings for cases that read none from a config (a fixed copy, so the numbers do not move with the user's settings) |
| `scenes/*.scene` | scene files some cases load (cavity, obstacle, dam break, Taylor-Green, Ethier-Steinman) |

## What each suite checks

### `adi_obstacle`: the regression checksums

| case | checks |
|---|---|
| 2D obstacle checksums | 50 ADI steps of a channel with an obstacle. It prints the divergence, the peak velocity and the sums of `|u|`, `|v|`, `p` and the interior `p`. |
| 3D obstacle checksums | the same in 3D, with `|w|` as well |

These are the guard on the shared pressure solver: any change to it must leave the numbers unchanged to the last
digit. The baseline values are recorded in the development notes. The 2D baseline was taken on the GPU (AMGX)
pressure path, so a CPU-only build matches the 3D column exactly and the 2D one only to the solver tolerance.

### `taylorgreen`: second order in space and time

| case | checks |
|---|---|
| Taylor-Green, uniform | runs the 2D Taylor-Green vortex at two resolutions, halving `dt` with `h`, and measures the error against the exact solution and the observed order |
| Taylor-Green, stretched | the same on a stretched grid. With equal spacing the stretched code path reproduces the uniform answer; with real stretching the order is kept |

### `exact3d`: the 3D exact solution

| case | checks |
|---|---|
| Ethier-Steinman | the scene matches the closed form, the `initial` field is seeded exactly, and the run stays bounded and divergence-free with a small error |
| Ethier-Steinman, stretched | the same on stretched grids, through both pressure paths. The two paths must agree. |

### `pressure`: prescribed-pressure faces and warm start

| case | checks |
|---|---|
| Dirichlet pressure, modes 0-3, 5-7, 9 | a pressure outlet with a known value. A linear field is the exact solution of the discrete system, so the solve must reproduce it to machine precision. The modes cover 2D/3D, CPU/AMGX, the FLIP and ADI scalings, and the incremental scheme with a time-dependent `p_b`. |
| warm start | starting each solve from the previous pressure gives the same answer, and counts the iterations it saves |

### `flip`: the FLIP method

| case | checks |
|---|---|
| FLIP 2D / 3D dam break | the column collapses, stays finite and divergence-free, and the answer does not depend on the density (`rho = 1` against `1000`) |
| FLIP 2D / 3D tank at rest | a still tank stays still (hydrostatic balance), and air faces carry no velocity |
| FLIP 2D / 3D obstacle contact | no particle ends inside a solid cell; particles slide past the obstacle |
| FLIP solid faces | zero normal velocity on every face of an interior solid |
| FLIP density correction | rest density is exact on the lattice, a still tank does not move, and the volume drift falls |
| viscous solve 2D / 3D | the implicit viscous solves (ADI split and single matrix) agree, and leave pinned faces alone |

### `sph`: the WCSPH method

| case | checks |
|---|---|
| WCSPH | a 2D tank keeps its hydrostatic bed pressure; 2D and 3D dam breaks run, keep their density in range, and keep every particle in the box; the telemetry grid reads the level and pressure |

### `transport`: scalar transport

| case | checks |
|---|---|
| scalar transport | the scalar registry (copy, checkpoint, rebuild), the 2D and 3D solvers (upwind bounded, central second order, totals conserved), inflow values, diffusion, reaction and source against closed forms, and the 3D solver reproducing the 2D one slice by slice |

### `io`: files in and out

| case | checks |
|---|---|
| checkpoint | 30 steps, save, rebuild from the config, load, 30 more steps must equal 60 straight steps to the last digit (ADI, FLIP 3D, SPH 3D). Mismatched files are refused. |
| export | writes `.vtr` and `.npy` files and reads them back: every value on its own point, coordinates on the cell centres, the `.npy` header and order |
| restart | ADI, FLIP, SPH and EXPLICIT each run in a fresh process, then twice in a row in one process with `DisposeSimulation` between. All runs must give the same result to the last bit. |
| headless check | `run_headless_check.sh` drives the program: exit codes, a run ending where its file says, export on and off, frame intervals, two runs agreeing byte for byte, auto-save, interrupt, and a field that goes non-finite |

### `setup_ui`: the setup screen

| case | checks |
|---|---|
| setup screen | the method table, writing a config and reading it back through the real config reader, AMGX settings that the solver accepts (GPU builds only; a CPU-only build skips them), scene, scalar and probe round trips, the one body force, and every shipped preset validating as saved |

### `units`: small pieces on their own

| case | checks |
|---|---|
| stopping rule | a run stops on its final time exactly; the last step is shortened to land on it |
| adaptive timestep | `TIME_STEP` is the ceiling; frames at a fixed simulated-time rate |
| grid spacing | the stretched-grid tables and refinement modes |
| boundary conditions | every boundary kind on every face, 2D and 3D, uniform and stretched |
| flow integrals | kinetic energy and enstrophy against closed forms |
| cell Reynolds | the largest cell Reynolds number and where it is |
| probes | each probe quantity, the sensor radius, solids left out |
| vorticity confinement | its force and where it acts |
| scenes | the scene parser and evaluator against the hard-coded cases in `Functions.h` |
| STL meshes | signed distance of placed meshes, nested components, 2D sections |
| NACA airfoils | the airfoil shape, distance and voxelised area |
