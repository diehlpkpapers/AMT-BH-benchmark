# ItoyoriFBC Barnes-Hut (Team Kassel)

Distributed Implementation for Single Node Milestone.

Requires OpenMPI 5.x.

## Build (Benchmark+Runtime System)
./build.sh

## Run
./run.sh [input.csv]

Set env variable RANKS to number of cores of the machine, e.g.,
RANKS=32 ./run.sh

Defaults to data/plummer_1000.csv. Options via env:
RANKS (4), STEPS (10), THETA (0.5), EPS2 (1e-6), DT (0.01), LEAF (8), CHUNK (1024), OUTPUT_EVERY (0)

Input: CSV with columns id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z.
Larger datasets: https://github.com/vancraar/DataGenerator

The binary prints a per-step energy table and a "timings (ms)" line.
