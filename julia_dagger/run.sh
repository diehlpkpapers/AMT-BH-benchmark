#!/bin/bash
# usage: ./run.sh [input.csv] [extra simulate.jl args...]
# env: THREADS PROCS RANKS BACKEND STEPS WARMUP THETA LEAF DT EPS G TILES QUIET
set -e
cd "$(dirname "$0")"

INPUT=data/collision_1000.csv
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then INPUT=$1; shift; fi

JULIA=${JULIA:-julia}
THREADS=${THREADS:-4}
PROCS=${PROCS:-1}
BACKEND=${BACKEND:-distributed}

ARGS=(--input="$INPUT" --steps="${STEPS:-10}" --warmup_steps="${WARMUP:-1}"
      --theta="${THETA:-0.5}" --leaf_capacity="${LEAF:-16}" --quiet="${QUIET:-false}")
[ -n "${DT:-}" ]    && ARGS+=(--dt="$DT")
[ -n "${EPS:-}" ]   && ARGS+=(--softening="$EPS")
[ -n "${G:-}" ]     && ARGS+=(--G="$G")
[ -n "${TILES:-}" ] && ARGS+=(--tiles="$TILES")

if [ "$BACKEND" = "mpi" ]; then
  # One rank per memory domain, THREADS threads inside each.  Inside a Slurm
  # allocation the site launcher is the right one; otherwise fall back to the
  # mpiexec that MPI.jl was configured with.
  RANKS=${RANKS:-2}
  ARGS+=(--backend=mpi --threads="$THREADS")
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    srun -n "$RANKS" "$JULIA" --project=. --threads="$THREADS" bin/simulate.jl "${ARGS[@]}" "$@"
  else
    "$JULIA" --project=. scripts/run_mpi.jl "$RANKS" "$THREADS" bin/simulate.jl "${ARGS[@]}" "$@"
  fi
else
  # PROCS processes on this node, THREADS threads each; process 1 owns a
  # partition too, so only PROCS-1 workers are added.
  ARGS+=(--workers=$((PROCS - 1)) --use_master=true --threads="$THREADS")
  "$JULIA" --project=. --threads="$THREADS" bin/simulate.jl "${ARGS[@]}" "$@"
fi
