#!/bin/bash
#
# Quick smoke-test run: a small subset of the solar-system scenario for a
# few steps. Assumes ./build.sh has already produced build_hpx/simulate
# (or build_hpx_kamand/simulate on a kamand host).

set -e

bin="build_hpx/simulate"
[ -x "$bin" ] || bin="build_hpx_kamand/simulate"

"$bin" \
  --file data/state_vectors_csvs/scenario1_19054.csv \
  --bodies 1000 \
  --dt 1h --tend 1d \
  --theta 0.5 --fc let \
  --hpx:threads=4
