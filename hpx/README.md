# HPX Barnes-Hut N-body Simulation

A distributed/parallel Barnes-Hut gravitational N-body solver built on
[HPX](https://hpx.stellar-group.org/), part of the [AMT-BH-benchmark](../README.md) suite.
It runs across HPX localities (MPI, TCP, or LCI) and threads per locality, and supports
both real solar-system datasets and synthetic galaxy models (see `data/`).

## How to build

Build HPX from source and this project against it with:

```bash
./build.sh
```

This clones and builds HPX (once, into `hpx`/`build/hpx`) and then configures and builds
this project against it into `build_hpx/`. On a host named `kamand*` it builds into
`build_hpx_kamand/` instead (and HPX itself into `build/hpx_kamand`), so a kamand build
never collides with an existing buran one. Useful flags:
*   `-c`: clean this project's own build directory first
*   `-C`: additionally reconfigure/rebuild HPX itself from scratch (slow)
*   `-d`: build this project in Debug instead of Release (HPX itself is always Release)
*   `-j N`: parallel build jobs (default: all cores)

This builds the main executable `build_hpx/simulate` (or `build_hpx_kamand/simulate` on kamand).

## Quick example

```bash
./run.sh
```

Runs a small subset of the solar-system scenario for a few steps as a smoke test. See
`run.sh` for the exact command; adjust `--bodies`/`--dt`/`--tend` for a bigger run.

## How to run

The CSV files are located in the `data/state_vectors_csvs` directory. All commands below
are run from this directory (`hpx/`).

The simulation supports several force calculation strategies via the `--fc` flag:
*   `tree`: Baseline method where all ranks exchange their full local trees.
*   `let`: Default method using Locally Essential Trees (LET) where remote interactions are handled by merging trees.
*   `let_direct`: An alternative LET method where remote forces are calculated with a direct summation.

Threads per rank are controlled with HPX's own `--hpx:threads=N` flag (not `OMP_NUM_THREADS`).

Gravitational softening is off by default (`--softening 0`), which is fine for well-separated
real-body data (e.g. solar-system scenarios) but not for dense datasets with close encounters
(e.g. galaxy models) - without softening those can hit near-singular forces that blow up energy
conservation regardless of timestep. Set `--softening <epsilon>` (same distance unit as the
input positions) to add a Plummer softening length for such data.

#### Scenario 1:

```bash
srun -N 4 ./build_hpx/simulate \
    --file data/state_vectors_csvs/scenario1_19054.csv \
    --dt 1h --tend 12y --vs 2d --outdir sim_s1 \
    --theta 1.05 --fc let --hpx:threads=48
```

#### Scenario 2:

```bash
srun -N 4 ./build_hpx/simulate \
    --file data/state_vectors_csvs/scenario2_300149.csv \
    --dt 1h --tend 1y --vs 7d --outdir sim_s2 \
    --theta 1.05 --fc let --hpx:threads=48
```

To run with fewer bodies, add the `--bodies` option. For a full list of options, use `--help`.

You can also use `sbatch scenario_1.sh` or `sbatch scenario_2.sh` to submit a job to the queue.

### Diagnostics

*   `--energy` tracks total (kinetic + potential) energy every step (or every `--et`
    interval), logs it to `<outdir>/energy.csv`, and reports the relative drift, for
    checking energy conservation. Add `--energy-exact` to use an O(N²) direct-sum
    potential energy instead of the Barnes-Hut approximation (only practical for small N).
*   `--timers loadbal,tree,interactions,output` (or `all`/`none`) prints a per-phase
    timing breakdown; add `--timers-per-step` to print it for every step instead of
    just the first plus the run aggregate.
*   `-r`/`--reference` forces `--fc tree` and saves final positions to
    `reference/final_ref.csv` instead of comparing against one, for generating a
    baseline to diff other runs/branches against.

## Approach

This simulation uses a parallel Barnes-Hut algorithm with HPX for N-body problems, running
across ranks (HPX localities) and threads on each rank.

The main loop has several phases:

1.  **Load Balancing:** To balance the work, particles are moved between ranks every so often. The histogram, splitter, and migration steps are all skipped when running on a single locality (nothing to rebalance against).
    *   It maps particles to 1D space with Morton Z-order curves.
    *   It builds a global histogram of where particles are (via HPX's `all_reduce` collective).
    *   It calculates new boundaries to divide the particle count evenly.
    *   It moves particles to their new ranks with HPX's `all_to_all` collective.

2.  **Local Octree Construction:** After balancing, each rank builds an octree with just its own particles, by sorting them by Morton key and recursing top-down over the sorted ranges, aggregating each cell's mass on the way back up. Subdivision stops as soon as a cell holds at most `--leaf-capacity` bodies (1 by default - the classic one-body-per-leaf rule), which keeps the tree small. For each force-calculation call, the tree is flattened once into a contiguous, directly-indexed array, so the per-body walks that follow are plain array indexing instead of hashmap lookups.

3.  **Force Calculation:** The simulation has a few ways to calculate forces.
    *   **Full Tree Exchange (`--fc tree`):** A basic method where every rank sends its full local tree to everyone else. Each rank then builds the same global tree and traverses it. This is simple but sends a lot of data.
    *   **Locally Essential Tree (LET) (`--fc let` or `--fc let_direct`):** This approach sends less data.
        *   Each rank finds the nodes (pseudo-leaves) from its tree that other ranks need to know about based on the Barnes-Hut opening angle (`θ`).
        *   It sends only these "essential" nodes.
        *   The receiving rank can either **merge** these nodes into its local tree (`let`) or calculate their forces with a **direct sum** (`let_direct`).

4.  **Integration:** It uses a Kick-Drift-Kick (Leapfrog) integrator to update particle positions and velocities. HPX's parallel algorithms and tasks are used to parallelize the force calculations and updates across each rank's threads.

5.  **Visualization:** The program saves output that can be opened in ParaView. Each rank writes its own particle data to `.vtp` files. The root rank also creates `.pvd` timeline files to animate the particle snapshots, the load-balancing histograms, and the exchanged LET data for debugging.


## Generating the scenario files

The orbital element CSVs are in `data/orbital_elements_raw_csvs`.

*   For scenario 2, large asteroid data (300,000 bodies) comes from the JPL database.
*   Planets, moons, and small asteroid data are provided.

To generate the state vector CSVs for each scenario, you can use the provided command-line conversion tool.

To build the tool:

```bash
g++ -std=c++17 -o orbital_converter src/orbital_converter.cpp src/kepler_to_cartesian.cpp -I./include
```

To run the tool:

```bash
./orbital_converter <planets_and_moons.csv> <asteroids.csv> <output_dir>
```

This will output the combined state vector CSVs required for the simulation.

