#!/bin/bash
#
# Build Charm++ and the Barnes-Hut benchmark.
#
#   ./build.sh [extra make arguments...]
#
# This clones Charm++ into ./_deps/charm, builds the reconverse-<os>-<arch>
# target for this machine, and then builds `barnes` and the two input
# generators against it. Charm++ is the only dependency: Reconverse and LCI
# are fetched by Charm++'s own CMake, and the geometry headers this code needs
# are self-contained in this directory.
#
# The Reconverse targets are not on Charm++'s main branch yet, so the default
# branch below is the one that carries them. It is also the runtime the
# measurements in README.md were taken on.
#
# Environment:
#   CHARM_PATH    Use this already-built Charm++ target directory instead of
#                 building one.
#   CHARM_SRC     Where to clone Charm++.        (default ./_deps/charm)
#   CHARM_REPO    Charm++ git remote.
#   CHARM_BRANCH  Branch or tag to clone.        (default reconverse-specific-build)
#   CHARM_TARGET  Charm++ build triplet.         (default reconverse-<os>-<arch>)
#   CHARM_OPTS    Options passed to Charm++'s ./build.
#                                                (default "--with-production --disable-fortran")
#   JOBS          Parallel build jobs.           (default: core count)
#   TRACE         none|summary|projections.      (default none)
#
# The Charm++ tree is built once and reused. To rebuild it from scratch --
# after changing CHARM_TARGET or CHARM_OPTS, say -- remove the build directory
# this prints, or remove ./_deps entirely.

set -euo pipefail
cd "$(dirname "$0")"

CHARM_SRC=${CHARM_SRC:-$PWD/_deps/charm}
CHARM_REPO=${CHARM_REPO:-https://github.com/charmplusplus/charm.git}
CHARM_BRANCH=${CHARM_BRANCH:-reconverse-specific-build}
# No `smp` here: the Reconverse network type is inherently SMP and Charm++
# forces CMK_SMP itself. --disable-fortran because this benchmark is pure C++
# and never links the Fortran bindings, while Charm++ builds them by default
# -- which turns a missing or half-installed gfortran into a failed build for
# no gain.
CHARM_OPTS=${CHARM_OPTS:---with-production --disable-fortran}
TRACE=${TRACE:-none}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}

die() { echo "build.sh: $*" >&2; exit 1; }

# The Reconverse targets Charm++ carries; see src/arch in the Charm++ tree.
default_target() {
    local os arch
    case "$(uname -s)" in
        Darwin) os=darwin ;;
        Linux)  os=linux ;;
        *) die "no Reconverse target for $(uname -s); set CHARM_TARGET" ;;
    esac
    case "$(uname -m)" in
        arm64|aarch64) arch=arm8 ;;
        x86_64|amd64)  arch=x86_64 ;;
        *) die "no Reconverse target for $(uname -m); set CHARM_TARGET" ;;
    esac
    case "$os-$arch" in
        darwin-arm8|linux-arm8|linux-x86_64) ;;
        *) die "Charm++ has no reconverse-$os-$arch target; set CHARM_TARGET" ;;
    esac
    echo "reconverse-$os-$arch"
}

CHARM_TARGET=${CHARM_TARGET:-$(default_target)}

if [ -n "${CHARM_PATH:-}" ]; then
    [ -x "$CHARM_PATH/bin/charmc" ] ||
        die "no charmc under CHARM_PATH=$CHARM_PATH -- point it at a built target directory"
    CHARM_PATH=$(cd "$CHARM_PATH" && pwd)
    echo "==> using existing Charm++ at $CHARM_PATH"
else
    if [ ! -d "$CHARM_SRC" ]; then
        echo "==> cloning Charm++ ($CHARM_BRANCH) into $CHARM_SRC"
        git clone --depth 1 --branch "$CHARM_BRANCH" "$CHARM_REPO" "$CHARM_SRC"
    else
        echo "==> reusing Charm++ source at $CHARM_SRC"
    fi

    # --destination fixes the build directory name. Without it Charm++ derives
    # it from the options, which would leave us guessing where charmc landed.
    dest=bh-$CHARM_TARGET
    CHARM_PATH=$CHARM_SRC/$dest

    if [ -x "$CHARM_PATH/bin/charmc" ]; then
        echo "==> Charm++ already built at $CHARM_PATH"
    else
        echo "==> building Charm++ $CHARM_TARGET $CHARM_OPTS"
        (
            cd "$CHARM_SRC"
            # CHARM_OPTS is deliberately unquoted: it is a list of words.
            ./build charm++ "$CHARM_TARGET" $CHARM_OPTS \
                    --destination="$dest" -j"$JOBS"
        )
        [ -x "$CHARM_PATH/bin/charmc" ] ||
            die "Charm++ build finished but produced no $CHARM_PATH/bin/charmc"
    fi
    CHARM_PATH=$(cd "$CHARM_PATH" && pwd)
fi

echo "==> building barnes (TRACE=$TRACE)"
make CHARM_PATH="$CHARM_PATH" TRACE="$TRACE" -j"$JOBS" "$@"

# run.sh reads this so a plain `./run.sh` needs no environment of its own.
{
    echo "# Written by build.sh. Delete it to forget this build."
    printf 'CHARM_PATH=%q\n' "$CHARM_PATH"
    printf 'CHARM_TARGET=%q\n' "$CHARM_TARGET"
} > .build.env

echo
echo "Charm++:  $CHARM_PATH"
echo "binaries: barnes, plummer, gen"
echo "next:     ./run.sh"
