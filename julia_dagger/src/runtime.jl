"""
Execution backends.
"""

const MPI_PKGID = Base.PkgId(Base.UUID("da04e1cc-30fd-572f-bb4f-1f8673147195"), "MPI")

"""
    load_mpi!()

MPI.jl bei Bedarf laden.  Der Import aktiviert Daggers MPI-Erweiterung, die das
`:mpi`-Backend braucht; das `:distributed`-Backend fasst MPI nie an.  Das
Nachladen haelt eine kaputte MPI.jl-Installation davon ab, auch Laeufe zu
blockieren, die sie gar nicht benutzen.
"""
load_mpi!() = Base.require(MPI_PKGID)

"""
    setup_backend!(cfg)

Activate the requested Dagger backend.  For `:distributed` this adds
single-threaded worker processes and loads this package on them; for `:mpi` it
switches Dagger's acceleration and, on request, enables Dagger's SPMD
uniformity checks.
"""
function setup_backend!(cfg::Config)
    if cfg.backend === :mpi
        load_mpi!()
        # `Base.require` defines Dagger's MPI extension at runtime, so
        # `accelerate!(::Val{:mpi})` lives in a newer world age than this function
        # was compiled in.  `invokelatest` steps into the current world.
        Base.invokelatest(Dagger.accelerate!, :mpi)
        Base.invokelatest(Dagger.check_uniformity!, cfg.check_uniformity)
    else
        if cfg.workers > 0 && nprocs() < cfg.workers + 1
            project = dirname(dirname(pathof(@__MODULE__)))
            # One process per memory domain, threads inside it.  `--threads=1` would
            # hand each process a single processor and collapse the hybrid layout.
            nthr = cfg.threads > 0 ? cfg.threads : Sys.CPU_THREADS ÷ max(cfg.workers, 1)
            addprocs(cfg.workers + 1 - nprocs();
                     exeflags=["--project=$project", "--threads=$(max(1, nthr))"])
        end
        if nprocs() > 1
            Distributed.remotecall_eval(Main, procs(), :(using NBodyDagger))
        end
        Dagger.accelerate!(:distributed)
        Dagger.check_uniformity!(false)
    end
    return cfg
end

"""
    local_id(cfg) -> Int

Identifier of the process this call runs on: the Distributed worker id, or
under the MPI backend the rank Dagger reports for the local process.  Used only
to pick the single process that prints and writes the report files.
"""
function local_id(cfg::Config)
    if cfg.backend === :mpi
        # Dagger's own accessor for "the processor of the local process"; asking
        # Dagger keeps the rank out of the application's logic.
        p = Dagger.default_processor(Dagger.current_acceleration(), local_id)
        hasproperty(p, :rank) && return Int(getproperty(p, :rank))
    end
    return myid()
end

"True on the one process that prints and writes the report files."
is_reporter(cfg::Config) = local_id(cfg) == (cfg.backend === :mpi ? 0 : 1)

"""
    transport_info(cfg) -> String

How Dagger moved data in this run, recorded with every result.  For the MPI
backend the MPI library name and version are read out of MPI.jl purely as
run metadata.
"""
function transport_info(cfg::Config)
    cfg.backend === :mpi || return "Distributed.jl (addprocs), $(nprocs()) processes"
    mpi = get(Base.loaded_modules, MPI_PKGID, nothing)
    mpi === nothing && return "Dagger MPI accelerator"
    return "Dagger MPI accelerator, $(mpi.MPI_LIBRARY) $(mpi.MPI_LIBRARY_VERSION)"
end

