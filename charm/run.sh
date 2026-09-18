#!/bin/bash
#
# Run the Barnes-Hut benchmark.
#
#   ./run.sh [input.csv] [extra barnes options...]
#
# With no input file, a 10k-body Plummer input is generated once into
# particles.csv and reused. That input is a convenience for local testing; the
# datasets the comparison actually runs on come from the project's data
# generator, and are passed as the first argument.
#
# Environment:
#   PES         PEs (worker threads) per process.    (default 4)
#   PROCS       Processes.                           (default 1)
#   STEPS       Steps to run, -killat.               (default 10)
#   NBODY       Bodies to generate when no input is given.  (default 10000)
#   PPC         Target particles per TreePiece, -ppc.
#   LEAF        Max particles per leaf, -b.
#   THETA       Opening angle, -theta.
#   DT          Timestep, -dtime.
#   EPS         Softening length, -eps.
#   G           Gravitational constant, -G.
#   TIMERS      Phases to time, -timers.             (default all)
#   OUTPUT      ParaView snapshot prefix, -output.   (default: no output)
#   OUTPUT_FREQ Write a snapshot every Nth step.
#   LAUNCHER    Launcher command, used verbatim, e.g.
#               "srun -n 8 --cpu-bind=none --unbuffered".
#   CHARM_PATH  Charm++ build directory. Defaults to whatever build.sh used.
#
# Anything unset is left to the program's own default; see README.md for the
# table of options and their values.
#
# Examples:
#   ./run.sh                                  # 10k bodies, 10 steps, 4 PEs
#   PES=8 STEPS=100 ./run.sh                  # longer, wider
#   ./run.sh data.csv -ppc=2000               # the project's dataset
#   PROCS=8 PES=15 LAUNCHER="srun -n 8 --cpu-bind=none --unbuffered" \
#       ./run.sh data.csv -ppc=2000 -G=4.498626405128888e-15 -dtime=5000 -eps=0.1
#
# The last one is a full Delta CPU node. Runs at that scale need the fabric
# environment and the physical units set too -- see "Running at scale" in
# README.md, which the defaults here deliberately do not try to guess at.

set -euo pipefail
cd "$(dirname "$0")"

die() { echo "run.sh: $*" >&2; exit 1; }

[ -x ./barnes ] || die "no ./barnes -- run ./build.sh first"

# build.sh records the Charm++ it built against; lcrun lives inside that tree.
if [ -f .build.env ]; then
    # shellcheck disable=SC1091
    . ./.build.env
fi

PES=${PES:-4}
PROCS=${PROCS:-1}
STEPS=${STEPS:-10}
NBODY=${NBODY:-10000}

# A leading argument that is not an option is the input file.
INPUT=
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then INPUT=$1; shift; fi

if [ -z "$INPUT" ]; then
    INPUT=particles.csv
    if [ ! -f "$INPUT" ]; then
        echo "==> generating $NBODY bodies into $INPUT"
        ./plummer "$NBODY" "$INPUT" csv
    fi
fi
[ -f "$INPUT" ] || die "input file $INPUT does not exist"

ARGS=(-in="$INPUT" -killat="$STEPS" -timers="${TIMERS:-all}")
if [ -n "${PPC:-}" ];   then ARGS+=(-ppc="$PPC");     fi
if [ -n "${LEAF:-}" ];  then ARGS+=(-b="$LEAF");      fi
if [ -n "${THETA:-}" ]; then ARGS+=(-theta="$THETA"); fi
if [ -n "${DT:-}" ];    then ARGS+=(-dtime="$DT");    fi
if [ -n "${EPS:-}" ];   then ARGS+=(-eps="$EPS");     fi
if [ -n "${G:-}" ];     then ARGS+=(-G="$G");         fi
if [ -n "${OUTPUT:-}" ]; then
    ARGS+=(-output="$OUTPUT")
    if [ -n "${OUTPUT_FREQ:-}" ]; then ARGS+=(-outputfreq="$OUTPUT_FREQ"); fi
fi

# Under Reconverse the process count comes from the launcher and +ppn is the
# PE count within each process, so PROCS x PES PEs in total.
LAUNCH=()
if [ -n "${LAUNCHER+x}" ]; then
    read -r -a LAUNCH <<< "$LAUNCHER"
elif [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
    # --cpu-bind=none so Slurm's binding does not fight Charm++'s own, and
    # --unbuffered because libfabric's fatal messages are otherwise lost to
    # buffering and a dying job looks like a hang.
    LAUNCH=(srun -n "$PROCS" --cpu-bind=none --unbuffered --kill-on-bad-exit=1)
elif [ "$PROCS" -eq 1 ]; then
    LAUNCH=()
elif [ -n "${CHARM_PATH:-}" ] && [ -x "$CHARM_PATH/_deps/lci-src/lcrun" ]; then
    # Reconverse bootstraps through LCI's PMI file backend, which lcrun sets
    # up and a plain mpirun does not.
    LAUNCH=("$CHARM_PATH/_deps/lci-src/lcrun" -n "$PROCS")
else
    die "PROCS=$PROCS needs a launcher; set LAUNCHER, or run under Slurm"
fi

# The ${a[@]+...} spelling keeps `set -u` from tripping over an empty array,
# which is a real case here (single-process runs take no launcher) and which
# bash 3.2, still the default shell on macOS, does not special-case.
echo "==> ${LAUNCH[@]+${LAUNCH[*]}} ./barnes +ppn $PES ${ARGS[*]} $*"
exec ${LAUNCH[@]+"${LAUNCH[@]}"} ./barnes +ppn "$PES" "${ARGS[@]}" "$@"
