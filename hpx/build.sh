#!/bin/bash
#
# Clones, configures and builds HPX from source (upstream master), then
# configures and builds this repo's HPX port (develop branch) against it.
# Supports buran and kamand; kamand builds into separate
# build/hpx_kamand + build_hpx_kamand directories (below) so it never
# touches an existing buran build (build/hpx + build_hpx) - the HPX git
# source clone (./hpx) is shared as-is between hosts since it's
# host-agnostic (only its build+install directories are per-host).
#
# Stage 1 (HPX): official TheHPXProject/hpx.git, branch master. Built with
# TCP/MPI/LCI parcelports; HPX's own test suite is left off
# (HPX_WITH_TESTS=OFF) since nothing here needs it, only this repo's
# simulate target.
#
# Stage 2 (this repo): points CMAKE_PREFIX_PATH at the HPX we just built.
#
# HPX must be compiled with the exact same compiler used here (mismatched
# libstdc++ versions fail at link time); the per-host module loads below
# pin that compiler for both stages.
#
# Usage: ./compile_hpx.sh [-c] [-C] [-d] [-j N]
#   -c   clean this project's build directory before configuring
#   -C   also clean HPX's build+install and reconfigure/rebuild it from
#        scratch (slow - only the source clone under ./hpx is kept)
#   -d   configure a Debug build (default: Release) for this project
#        (the HPX dependency itself is always built Release)
#   -j N number of parallel build jobs (default: nproc)

set -e
set -x

build_type="Release"
clean_project=0
clean_hpx=0
jobs=$(nproc)

while getopts "cCdj:" opt; do
  case "$opt" in
    c) clean_project=1 ;;
    C) clean_project=1; clean_hpx=1 ;;
    d) build_type="Debug" ;;
    j) jobs="$OPTARG" ;;
    *) echo "Usage: $0 [-c] [-C] [-d] [-j N]" >&2; exit 1 ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Match the compiler HPX is built with, and pick this host's build
# directory suffix (kamand gets its own so it never collides with buran) ---
case "$(hostname)" in
    buran*)
        module load gcc/14.2.0
        module load openmpi/5.0.5
        host_suffix=""
        ;;
    kamand*)
        module load gcc/14.2.0
        module load openmpi/5.0.5
        host_suffix="_kamand"
        ;;
    *)
        echo "ERROR: unknown host $(hostname), cannot load modules" >&2
        exit 1
        ;;
esac
export CC=gcc
export CXX=g++

project_build_dir="${script_dir}/build_hpx${host_suffix}"

# =============================================================================
# Stage 1: clone + build HPX from source
# =============================================================================
HPX_VERSION=master
DOWNLOAD_URL="https://github.com/TheHPXProject/hpx.git"
TCP=ON
MPI=ON
LCI=ON

# Optional build flags (default to empty if unset).
CXXFLAGS="${CXXFLAGS:-}"
LDCXXFLAGS="${LDCXXFLAGS:-}"

hpx_src="${script_dir}/hpx"
hpx_build="${script_dir}/build/hpx${host_suffix}"
hpx_install="${HPX_INSTALL:-${hpx_build}/install}"

if [ "$clean_hpx" -eq 1 ]; then
  rm -rf "${hpx_build}"
fi

# Clone HPX if not present (never removed by -C: only the build+install are).
if [ ! -d "${hpx_src}" ]; then
  git clone "${DOWNLOAD_URL}" "${hpx_src}"
  git -C "${hpx_src}" checkout "${HPX_VERSION}"
fi

mkdir -p "${hpx_build}"

# Configure with cmake if not already configured.
if [ ! -d "${hpx_build}/CMakeFiles" ]; then
  cmake \
      -S "${hpx_src}" \
      -B "${hpx_build}" \
      -DCMAKE_INSTALL_PREFIX="${hpx_install}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
      -DCMAKE_EXE_LINKER_FLAGS="${LDCXXFLAGS}" \
      -DCMAKE_SHARED_LINKER_FLAGS="${LDCXXFLAGS}" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DHPX_WITH_MALLOC=system \
      -DHPX_WITH_NETWORKING=ON \
      -DHPX_WITH_DISTRIBUTED_RUNTIME=ON \
      -DHPX_WITH_MORE_THAN_64_THREADS=ON \
      -DHPX_WITH_MAX_CPU_COUNT=256 \
      -DHPX_WITH_LOGGING=OFF \
      -DHPX_WITH_EXAMPLES=OFF \
      -DHPX_WITH_TESTS=OFF \
      -DHPX_WITH_APEX=OFF \
      -DHPX_WITH_PKGCONFIG=OFF \
      -DHPX_WITH_PARCELPORT_TCP=${TCP} \
      -DHPX_WITH_PARCELPORT_MPI=${MPI} \
      -DHPX_WITH_PARCELPORT_LCI=${LCI} \
      -DHPX_WITH_FETCH_LCI=${LCI} \
      -DHPX_WITH_FETCH_BOOST=ON \
      -DHPX_WITH_FETCH_HWLOC=OFF \
      -DHPX_WITH_FETCH_ASIO=ON
fi

cmake --build "${hpx_build}" --target install -j "${jobs}"

# =============================================================================
# Stage 2: configure + build this repo's HPX port against it
# =============================================================================
if [ "$clean_project" -eq 1 ]; then
  rm -rf "${project_build_dir}"
fi

mkdir -p "${project_build_dir}"
cmake -S "${script_dir}" -B "${project_build_dir}" \
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_PREFIX_PATH="${hpx_install}"
cmake --build "${project_build_dir}" -j "${jobs}"

echo "HPX built and installed at: ${hpx_install}"
echo "Build finished: ${project_build_dir}/simulate"
echo "Run with srun (simplest, uses HPX's default TCP bootstrap, no extra flags needed):"
echo "  srun -n <N> ${project_build_dir}/simulate [simulate args...]"
echo "  (inside an existing allocation on one node, add --overlap --overcommit)"
echo ""
echo "Or with mpirun, single node oversubscribed:"
echo "  mpirun --oversubscribe --mca btl tcp,self --mca pml ob1 -np <N> ${project_build_dir}/simulate \\"
echo "    --hpx:ignore-batch-env --hpx:ini=hpx.parcel.tcp.enable=0 --hpx:ini=hpx.parcel.bootstrap=mpi \\"
echo "    [simulate args...]"
echo "  (--hpx:ignore-batch-env / the tcp overrides work around every locality binding the"
echo "   same default HPX TCP port when co-located; usually unneeded across separate nodes)"
