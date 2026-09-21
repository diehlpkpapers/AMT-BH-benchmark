#!/usr/bin/env bash
#SBATCH --job-name=bh-dagger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x-%j.out
#
# Strong scaling on one node: total cores on the x axis.
#
#   INPUT=/path/data.csv sbatch slurm/strong_scaling.sh
#   CORES="1 2 4 8" DOMAINS=1 INPUT=/path/data.csv sbatch slurm/strong_scaling.sh
#
# DOMAINS is the number of processes, one per memory domain; the cores of each
# are filled with threads.  DOMAINS=1 is the shared-memory layout.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

INPUT=${INPUT:?set INPUT to a particle CSV}
CORES=${CORES:-"1 2 4 8 16 32 48 64 96 128 192"}
DOMAINS=${DOMAINS:-1}
STEPS=${STEPS:-3}
WARMUP=${WARMUP:-1}
THETA=${THETA:-0.5}
LEAF=${LEAF:-16}
REPEATS=${REPEATS:-3}

./build.sh

printf "%-6s %-6s %-8s %-10s %s\n" cores procs threads s/step drift
for c in $CORES; do
    p=$(( c < DOMAINS ? c : DOMAINS ))
    t=$(( c / p ))
    best=""
    for _ in $(seq "$REPEATS"); do
        log=$(PROCS=$p THREADS=$t STEPS=$STEPS WARMUP=$WARMUP THETA=$THETA LEAF=$LEAF \
              ./run.sh "$INPUT" 2>&1) || { echo "$c cores FAILED"; continue 2; }
        s=$(awk '/mean step/{print $(NF-1)}' <<<"$log")
        [ -z "$best" ] && best=$s
        best=$(awk -v a="$best" -v b="$s" 'BEGIN{print (b<a)?b:a}')
        drift=$(awk '/relative drift/{print $NF}' <<<"$log" | tr -d ')')
    done
    printf "%-6s %-6s %-8s %-10s %s\n" "$c" "$p" "$t" "$best" "${drift:-}"
done
