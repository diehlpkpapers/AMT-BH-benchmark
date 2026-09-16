# pepc-ttg

A Barnes-Hut gravitational N-body solver expressed as a task graph over
[TTG](https://github.com/TESSEorg/ttg) (Template Task Graph), on top of the
PaRSEC runtime. It exists primarily to measure TTG/PaRSEC per-task overhead
on a well-understood, physically checkable workload, and secondarily to
compare against [PEPC](https://github.com/SC-HPC-JSC/pepc)'s `pepc-gravity`
frontend as a reference implementation.

## Algorithm

Each timestep:

1. **Tree build** (sequential, host-side): particles are recursively
   partitioned into an octree until each leaf holds at most
   `--max-leaf` particles (`src/tree.h`, `src/tree.cc`). Nodes are addressed
   by `Key` (level + per-axis translation, `src/key.h`) rather than an
   allocated id, so parent/child relationships are pure arithmetic on the
   key.
2. **Upward pass** (TTG task graph, `src/upward_ttg.h`): one task per leaf
   computes that leaf's raw moment about its parent's box center; an
   aggregator per interior node fires once all of its children's moments
   have arrived, finalizing that node's multipole and forwarding a
   re-expressed moment to its own parent. The root's aggregator firing is
   the graph's own signal that every multipole in the tree is final —
   no host-side fence is needed between the upward and downward passes.
3. **Downward pass** (TTG task graph, `src/traversal_ttg.h`): a MAC
   (multipole acceptance criterion, `theta`) driven descent, batched per
   *target leaf* rather than per particle — every particle in a leaf
   shares the same traversal decisions, which cuts task count by roughly
   `max-leaf`. A source node is either accepted (its multipole is applied
   to every target particle) or expanded into its children; results are
   aggregated back up the source's side of the tree to a per-leaf sink.
4. **Integration**: leapfrog kick/drift is fused directly into the
   downward pass's per-leaf sink callback, so particles in a leaf that
   finishes early start integrating while other leaves are still deep in
   traversal (`src/integrate_ttg.h`, `src/integrate.h`).

The upward and downward passes are wired together into a single graph
(`src/bh_pass.h`) with one fence per step, rather than fencing between
passes.

See `Whitepaper_Barnes_Hut.pdf` for the underlying method and
`src/multipole.h` for the moment/quadrupole math.

## Building

Requires CMake ≥ 3.19 and a C++20 compiler. TTG (and its PaRSEC/MADNESS
dependencies) are pulled in via `FetchContent`; see
`cmake/modules/FindOrFetchTTG.cmake`.

```sh
cmake -B build
cmake --build build
ctest --test-dir build
```

Defaults to a `Release` build, since this project's purpose (measuring
per-task overhead) would otherwise be distorted by unoptimized code —
override with `-DCMAKE_BUILD_TYPE=Debug` if needed.

## Running

```sh
build/src/pepc-ttg-parsec --input data/two_body.csv --steps 100 --dt 1e-3
```

Without `--input`, a random uniform-in-cube particle distribution is
generated instead. Key flags (see `src/main.cc` for the full list):

| Flag | Meaning |
|---|---|
| `--input <csv>` | particle file: `id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z` |
| `--steps`, `--dt` | number of leapfrog steps and timestep |
| `--theta` | MAC opening angle (smaller = more accurate, more work) |
| `--eps` | gravitational softening length |
| `--max-leaf` | max particles per leaf before splitting |
| `--no-quadrupole` | monopole-only force evaluation |
| `--energy-every N` | report total energy every N steps (sanity check) |
| `--time` | print tree-build vs. traversal timing per step |
| `--vtk-prefix <p>` | write per-step VTK snapshots |
| `--n`, `--seed`, `--spread`, `--vel-spread` | random particle generation (used when `--input` is omitted) |

## Layout

- `src/` — the library and `pepc-ttg` executable: tree build (`tree.h/.cc`),
  TTG passes (`upward_ttg.h`, `traversal_ttg.h`, `bh_pass.h`), physics
  (`multipole.h`, `brute_force.h`, `integrate.h`/`integrate_ttg.h`), and
  I/O (`io.h/.cc`, CSV and VTK).
- `tests/` — correctness tests (Barnes-Hut vs. brute force, energy
  conservation), a pure-math force-kernel check, and a standalone
  tree-build performance probe.
- `tools/` — scripts for the PEPC comparison workflow:
  `gen_shared_ic.py` generates matching initial conditions in both
  pepc-ttg's CSV format and PEPC's binary format; `compare_pepc.cc` runs
  pepc-ttg's own brute-force kernel on those ICs; `parse_pepc_vtu.py`
  extracts PEPC's VTU output for the same ICs so the two can be diffed.
- `data/` — sample initial-condition CSVs.
