# Launch a Julia script under the `mpiexec` provided by MPI.jl.
#
#   julia --project=<pkg> scripts/run_mpi.jl <nranks> <nthreads> <script> [args...]
#
# On a cluster with a system MPI, call the site launcher directly instead:
#   srun -n <nranks> julia --project=<pkg> --threads=<nthreads> <script> [args...]
using MPI

length(ARGS) >= 3 || error("usage: run_mpi.jl <nranks> <nthreads> <script> [args...]")
const NRANKS = parse(Int, ARGS[1])
const NTHREADS = parse(Int, ARGS[2])
const SCRIPT = ARGS[3]
const REST = ARGS[4:end]

const PROJECT = dirname(dirname(abspath(@__FILE__)))
cmd = `$(MPI.mpiexec()) -n $NRANKS $(Base.julia_cmd()[1]) --project=$PROJECT --threads=$NTHREADS $SCRIPT $REST`
run(cmd)
