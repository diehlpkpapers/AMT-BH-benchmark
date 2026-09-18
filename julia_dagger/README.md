# Barnes-Hut on Dagger.jl (Team Fulda)

A Barnes-Hut gravitational N-body solver in Julia, parallelised with
[Dagger.jl](https://github.com/JuliaParallel/Dagger.jl).

Requires Julia 1.12. For the distributed-memory backend, an MPI installation.

## Build

```shell
./build.sh              # shared memory only
MPI=1 ./build.sh        # also configure the MPI backend against the system MPI
```

No CMake: `Project.toml` and `Manifest.toml` pin the dependencies, and
`build.sh` only instantiates and precompiles the environment.

## Run

```shell
./run.sh [input.csv] [extra --key=value args...]
```

Defaults to `data/collision_1000.csv`. Options via env:

`THREADS` (4), `PROCS` (1), `RANKS` (2), `BACKEND` (distributed),
`STEPS` (10), `WARMUP` (1), `THETA` (0.5), `LEAF` (16), `DT`, `EPS`, `G`,
`TILES`, `QUIET`.

Anything left unset keeps the program's own default; `bin/simulate.jl --help`
lists every option.

The binary prints a per-phase timing table, the relative energy drift and the
interaction counts.

## Datasets

Input is CSV with columns `id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z`. The
reference layout with additional `name,class` columns is also accepted.

| file | bodies | what |
| --- | --- | --- |
| `data/collision_1000.csv` | 1 000 | two colliding Plummer spheres, from this package's own `--generate`; the default for `run.sh` |
| `data/collision_4M.json` | 4 000 000 | generator config for the benchmark dataset (see below) |

The measurement dataset is produced with the project's
[data generator](https://github.com/vancraar/DataGenerator) from
`data/collision_4M.json`: three galaxies with seed 20260918 — a `main` disk of
2 080 000 bodies, an `incoming` disk of 1 280 000 on a crash orbit at 45
degrees, and a 640 000-body `satellite` on a stable orbit. The CSV itself is
598 MB and is not in the repository; regenerate it with

```shell
python generate_data.py data/collision_4M.json
```

### Units

The generator works in parsecs, solar masses and years, where
G = 4.4997236660676e-15. This code takes `--G`, so the generator's output can
be used as it is:

```shell
G=4.4997236660676e-15 ./run.sh collision_4M.csv
```

Implementations whose gravitational constant is compiled in need the masses
scaled by `G_data / G_theirs` instead, after which every implementation's own
constant produces the same accelerations and no flag is needed. The runs
behind the reported numbers use such a rescaled file.

## Architecture: shared and distributed memory

A **partition** is one Dagger process and owns one contiguous block of the
particle array, one octree and one locally essential tree. Inside a process the
work is cut into **tiles**, index ranges into that block that the runtime
spreads over the process's threads. Partitions therefore follow memory domains
while cores are taken up by threads, which keeps the tree exchange from growing
with the machine.

**Shared memory** — one partition, threads inside it. No tree exchange at all,
since the single octree already spans the domain:

```shell
THREADS=64 ./run.sh data.csv
```

**Distributed memory, one node** — one partition per NUMA domain, threads
inside each. Uses Julia's `Distributed` (`addprocs`):

```shell
PROCS=2 THREADS=32 ./run.sh data.csv
```

**Distributed memory, multiple nodes** — the same code on Dagger's MPI backend,
one rank per node or per NUMA domain. MPI is used as Dagger's backend only and
is never called directly:

```shell
BACKEND=mpi RANKS=8 THREADS=16 ./run.sh data.csv
```

Inside a Slurm allocation `run.sh` launches with `srun`; outside it uses the
`mpiexec` that MPI.jl is configured with. `slurm/strong_scaling.sh` sweeps the
core count on one node.

## Test

```shell
julia --project=. -e 'using Pkg; Pkg.test()'
```
