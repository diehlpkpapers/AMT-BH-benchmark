"""
Hybrid layout: partitions follow memory domains, tiles follow threads.
"""

"""
    Layout

How the run is spread over the machine.  `procs` are one representative Dagger
processor per process (they carry the `DArray` blocks), `tile_procs[i]` are all
processors of partition `i`'s process (the threads its tiles can run on).
"""
struct Layout
    procs::Vector{Dagger.Processor}              # one per partition
    tile_procs::Vector{Vector{Dagger.Processor}} # all threads of each partition
    npart::Int                                   # partitions == processes
    nthreads::Int                                # threads per partition (max)
end

"Deterministic ordering key for a Dagger CPU processor, on either backend."
function proc_sort_key(p::Dagger.Processor)
    if hasproperty(p, :rank)                      # MPI backend processor
        inner = getproperty(p, :innerProc)
        return (Int(getproperty(p, :rank)), Int(inner.owner), Int(inner.tid))
    end
    return (0, Int(p.owner), Int(p.tid))          # ThreadProc
end

"Logical owner (rank or Distributed worker id) of a processor."
function proc_owner(p::Dagger.Processor)
    hasproperty(p, :rank) && return Int(getproperty(p, :rank))
    return Int(p.owner)
end

"""
    build_layout(cfg) -> Layout

Group the processors the runtime enumerates by their owning process.  Worker ids
and ranks are never written down, always asked for: the same call returns `ThreadProc`s under the distributed backend and
rank-stamped processors under MPI, which is what makes one code path work on
both and keeps placement deterministic.
"""
function build_layout(cfg::Config)
    accel = Dagger.current_acceleration()
    all = collect(Dagger.Processor, Dagger.compatible_processors(accel))
    sort!(all; by=proc_sort_key)

    # The driver owns a partition unless asked not to; with one process per NUMA
    # domain there is no spare process to keep free, so `use_master` defaults on.
    if cfg.backend === :distributed && !cfg.use_master && nprocs() > 1
        keep = filter(p -> proc_owner(p) != 1, all)
        isempty(keep) || (all = keep)
    end

    groups = Dict{Int,Vector{Dagger.Processor}}()
    for p in all
        push!(get!(groups, proc_owner(p), Dagger.Processor[]), p)
    end
    owners = sort!(collect(keys(groups)))

    # `--partitions` may ask for fewer partitions than there are processes (for
    # a scaling sweep on a fixed allocation); it never asks for more, because a
    # partition is a process by definition here.
    if cfg.partitions > 0
        cfg.partitions <= length(owners) ||
            error("--partitions=$(cfg.partitions) exceeds the $(length(owners)) available processes")
        owners = owners[1:cfg.partitions]
    end

    tile_procs = [groups[o] for o in owners]
    procs = [first(g) for g in tile_procs]
    nthreads = maximum(length, tile_procs)
    return Layout(procs, tile_procs, length(procs), nthreads)
end

"""
    partition_scope(lay, i) -> Dagger.AbstractScope

Where partition `i`'s tasks may run.  Under the distributed backend this is the
whole *process*, so Dagger is free to place and balance tiles across its
threads -- that freedom is the point of the hybrid layout.  Under MPI the scope
is the exact processor, because SPMD execution requires every rank to build an
"""
partition_scope(lay::Layout, i::Int, backend::Symbol) =
    backend === :mpi ? Dagger.ExactScope(lay.procs[i]) :
                       Dagger.ProcessScope(proc_owner(lay.procs[i]))

"""
    tiles(n, k) -> Vector{UnitRange{Int}}

Split `1:n` into at most `k` contiguous ranges of nearly equal length.  Empty
ranges are dropped, so a partition with fewer particles than threads simply gets
fewer tiles instead of empty tasks.
"""
function tiles(n::Int, k::Int)
    n <= 0 && return UnitRange{Int}[]
    k = max(1, min(k, n))
    out = Vector{UnitRange{Int}}(undef, k)
    lo = 1
    for t in 1:k
        hi = lo + cld(n - lo + 1, k - t + 1) - 1
        out[t] = lo:hi
        lo = hi + 1
    end
    return out
end

"""
    tile_count(cfg, lay, n) -> Int

How many tiles a partition of `n` particles is cut into.  `--tiles_per_thread`
tiles per thread, four by default.
"""
function tile_count(cfg::Config, lay::Layout, n::Int)
    cfg.tiles == 0 || return max(1, min(cfg.tiles, n))
    t = max(1, lay.nthreads)
    # `--tiles_per_thread` tiles per thread, bounded by `--max_tiles`.
    #
    # Below the thread count tiles buy parallelism; above it they cost, because
    # the driver issues every task from one thread at a per-task cost that grows
    # with the number of processors in the process.  The phase cost is roughly
    # `k*c + W/k`, smallest at `k = sqrt(W/c)`, which at high core counts is
    # fewer tiles than threads -- so the cap is allowed to bind below the thread
    # count.  `interact.spawn` against `interact.wait` in the metrics shows which
    # side of the optimum a run is on.
    return max(1, min(t * max(1, cfg.tiles_per_thread), cfg.max_tiles, n))
end
