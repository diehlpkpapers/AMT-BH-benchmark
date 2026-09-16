#!/bin/bash
# usage: ./run.sh [input.csv] [extra barneshut args...]
# env: RANKS STEPS THETA EPS2 DT LEAF CHUNK OUTPUT_EVERY
set -e
cd "$(dirname "$0")"

INPUT=data/plummer_1000.csv
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then INPUT=$1; shift; fi

RANKS=${RANKS:-4}
BIN=${BIN:-build/barneshut.out}
export ITYR_CACHE_FLUSH="${ITYR_CACHE_FLUSH:-0}"   # set 1 on fabrics that need the RMA write-back

ARGS=(--input "$INPUT" --steps "${STEPS:-10}" --theta "${THETA:-0.5}"
      --eps2 "${EPS2:-1e-6}" --dt "${DT:-0.01}" --leaf "${LEAF:-8}" --chunk "${CHUNK:-1024}")
if [ "${OUTPUT_EVERY:-0}" -gt 0 ]; then
  ARGS+=(--output-every "$OUTPUT_EVERY" --output-prefix snapshot)
else
  ARGS+=(--no-output)
fi

mpirun -n "$RANKS" setarch "$(uname -m)" --addr-no-randomize \
  "$BIN" "${ARGS[@]}" "$@"
