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
  "$JULIA" --project=. -e '
      using Pkg; Pkg.add(name="MPI", version="0.20.27")
      using MPIPreferences; MPIPreferences.use_system_binary()'
fi

"$JULIA" --project=. -e 'using NBodyDagger' && echo "build ok"
