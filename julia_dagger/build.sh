#!/bin/bash
# Resolves the Julia environment and precompiles the package.
# env: JULIA (julia binary), MPI=1 (also build against the system MPI)
set -e
cd "$(dirname "$0")"

JULIA=${JULIA:-julia}
command -v "$JULIA" >/dev/null || { echo "julia not found; set JULIA=/path/to/julia" >&2; exit 1; }

"$JULIA" --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# The MPI backend is an optional dependency: only added when asked for, so a
# shared-memory build needs no MPI installation at all.
if [ "${MPI:-0}" = "1" ]; then
  # MPIPreferences is what points MPI.jl at the system library, so it has to be
  # in the environment as well.  The library is looked up in the MPI wrapper's
  # own library directory too, which is not on the default search path for
  # e.g. a Homebrew Open MPI; MPI_LIBDIR overrides it.
  MPI_LIBDIR=${MPI_LIBDIR:-$(mpicc --showme:libdirs 2>/dev/null | awk '{print $1}' || true)}
  if [ -z "$MPI_LIBDIR" ] && command -v mpiexec >/dev/null; then
    MPI_LIBDIR=$(cd "$(dirname "$(command -v mpiexec)")/../lib" 2>/dev/null && pwd || true)
  fi
  MPI_LIBDIR="$MPI_LIBDIR" "$JULIA" --project=. -e '
      using Pkg; Pkg.add([PackageSpec(name="MPI", version="0.20.27"),
                          PackageSpec(name="MPIPreferences")])
      using MPIPreferences
      dir = get(ENV, "MPI_LIBDIR", "")
      MPIPreferences.use_system_binary(; extra_paths = isempty(dir) ? String[] : [dir])'
fi

"$JULIA" --project=. -e 'using NBodyDagger' && echo "build ok"
