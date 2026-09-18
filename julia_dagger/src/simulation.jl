"""
The Dagger layer: distributed state, the timestep pipeline, and the run driver.
"""

# Field layout inside one partition block.
const F_M = 1
const F_X = 2
const F_Y = 3
const F_Z = 4
const F_VX = 5
const F_VY = 6
const F_VZ = 7
const F_AX = 8
const F_AY = 9
const F_AZ = 10
const F_POT = 11
const NFIELD = 11

# Metadata layout (UInt64) inside one partition block.
const M_KEY = 1
const M_ID = 2
const NMETA = 2

# Per-partition reduction slots.  Only values produced *inside* Datadeps tasks
# live here; the bounding boxes and particle counts come back from pure tasks
# and are kept on the driver.
const RED_KE = 1
const RED_PE = 2
const RED_DIRECT = 3
const RED_APPROX = 4
const RED_LETNODES = 5
const RED_NIMPORT = 6
const NRED = 6

"View of field `f` of a partition block holding `n` valid particles."
@inline fview(buf::AbstractVector{Float64}, bs::Int, f::Int, n::Int) =
    view(buf, ((f - 1) * bs + 1):((f - 1) * bs + n))

@inline mview(meta::AbstractVector{UInt64}, bs::Int, f::Int, n::Int) =
    view(meta, ((f - 1) * bs + 1):((f - 1) * bs + n))

"All field views of one partition block, as a named tuple of contiguous slices."
@inline function pfields(buf::AbstractVector{Float64}, bs::Int, n::Int)
    return (m=fview(buf, bs, F_M, n), x=fview(buf, bs, F_X, n),
            y=fview(buf, bs, F_Y, n), z=fview(buf, bs, F_Z, n),
            vx=fview(buf, bs, F_VX, n), vy=fview(buf, bs, F_VY, n),
            vz=fview(buf, bs, F_VZ, n), ax=fview(buf, bs, F_AX, n),
            ay=fview(buf, bs, F_AY, n), az=fview(buf, bs, F_AZ, n),
            pot=fview(buf, bs, F_POT, n))
end

# ---------------------------------------------------------------------------
# task bodies (top level, so that they can run on any process)
# ---------------------------------------------------------------------------

"""
    task_post_drift_box(buf, bs, n, dt) -> (Box, count, mass)

Bounding box the partition *will* have after this step's kick and drift,
computed exactly as `x + (v + a dt/2) dt`.  Computing it before the update lets
the kick, the drift, the Morton keys and the local sort share a single Datadeps
region: the keys need the global domain, and the domain needs the post-drift
"""
function task_post_drift_box(buf, bs::Int, n::Int, dt::Float64)::PartitionExtent
    n == 0 && return (EMPTY_BOX, 0, 0.0)
    p = pfields(buf, bs, n)
    h = 0.5 * dt
    xlo = ylo = zlo = Inf
    xhi = yhi = zhi = -Inf
    mass = 0.0
    @inbounds for i in 1:n
        xi = p.x[i] + (p.vx[i] + p.ax[i] * h) * dt
        yi = p.y[i] + (p.vy[i] + p.ay[i] * h) * dt
        zi = p.z[i] + (p.vz[i] + p.az[i] * h) * dt
        xlo = min(xlo, xi); xhi = max(xhi, xi)
        ylo = min(ylo, yi); yhi = max(yhi, yi)
        zlo = min(zlo, zi); zhi = max(zhi, zi)
        mass += p.m[i]
    end
    return (Box(xlo, ylo, zlo, xhi, yhi, zhi), n, mass)
end

"""
Per-partition value gathered by a pure task: bounding box, particle count and
mass.  A concrete named type (rather than a tuple of `Any`) is what lets Dagger
know the task's result type without asking the other ranks -- see the
`return_type` note below.
"""
const PartitionExtent = Tuple{Box,Int,Float64}

gather_extents(v...)::Vector{PartitionExtent} = collect(v)
"""
    collect_summaries(summaries) -> Vector{PartitionSummary}

Bring the per-partition cell summaries to the driver, which needs all of them
to build the coarse tree.
"""
function collect_summaries(summaries::Vector, backend::Symbol)
    # Under MPI a `fetch` on a DTask is collective, so P of them is P collectives
    # per step.  One gather task plus one fetch replicates the metadata in a
    # single collective and keeps the task graph SPMD-uniform.  Under Distributed
    # the direct fetch is faster, so the backends take different paths.
    backend === :mpi || return PartitionSummary[fetch(t) for t in summaries]
    return fetch(Dagger.@spawn task_gather_summaries(summaries...))
end

task_gather_summaries(xs::PartitionSummary...) = PartitionSummary[xs...]
task_gather_extents(xs::PartitionExtent...) = PartitionExtent[xs...]

"Half kick, drift, Morton keys and the local Morton sort, in place."
function task_move!(buf, meta, bs::Int, n::Int, dom::Domain, dt::Float64,
                    kick_h::Float64, lo::Int, hi::Int)::Nothing
    n == 0 && return nothing
    p = pfields(buf, bs, n)
    key = mview(meta, bs, M_KEY, n)
    r = lo:hi
    # `dt == 0` is the warm-up step: re-key without moving anything.
    if dt != 0
        kick_by!(p.vx, p.vy, p.vz, p.ax, p.ay, p.az, kick_h, r)
        drift!(p.x, p.y, p.z, p.vx, p.vy, p.vz, dt, r)
    end
    compute_keys!(key, dom, p.x, p.y, p.z, r)
    return nothing
end

"""
    task_sort!(buf, meta, bs, n)

Re-sort a partition into Morton order after `task_move!` has re-keyed it.
"""
function task_sort!(buf, meta, bs::Int, n::Int)::Nothing
    n == 0 && return nothing
    p = pfields(buf, bs, n)
    key = mview(meta, bs, M_KEY, n)
    id = mview(meta, bs, M_ID, n)
    sort_partition!(key, id, p.m, p.x, p.y, p.z, p.vx, p.vy, p.vz)
    return nothing
end

function task_build_tree(buf, meta, bs::Int, n::Int, dom::Domain,
                         leaf_capacity::Int, max_depth::Int)::Tree
    p = pfields(buf, bs, n)
    key = mview(meta, bs, M_KEY, n)
    return build_local_tree(dom, key, p.x, p.y, p.z, p.m, leaf_capacity, max_depth)
end


function task_summarize(tree, owner::Int, box::Box, n::Int, max_branches::Int, buf,
                        bs::Int)::PartitionSummary
    p = pfields(buf, bs, n)
    return summarize_partition(tree, owner, box, n, p.x, p.y, p.z; max_branches)
end

function task_export(tree, request_keys, cells, theta::Float64, buf, meta, bs::Int,
                     n::Int, owner::Int)::Bundle
    p = pfields(buf, bs, n)
    id = mview(meta, bs, M_ID, n)
    return export_bundle(tree, request_keys, cells, theta, p.x, p.y, p.z, p.m, id, owner)
end

task_merge_bundles(bundles...)::Bundle = merge_bundles(Bundle[bundles...])

task_assemble_let(tree, ct, cells, branch_keys, theta, self::Int, bundles...)::Tree =
    assemble_let(ct, tree, Bundle[bundles...], cells, branch_keys, theta, self)

# `reset` is true for the first batch of a partition: the batches of one
# partition are serialised on their own chunks anyway, so zeroing the counters
# here saves a separate Datadeps task.  `_after` is ignored; it exists only to
# order one batch after the previous batch of the same partition.
function task_forces!(buf, red, meta, bs::Int, n::Int, tree, lo::Int, hi::Int,
                      params::PhysicsParams, rowoff::Int)::Nothing
    # Every tile owns its own group of `NRED` rows and writes its own particle
    # range, so the tiles are disjoint and need no ordering between them.  Only
    # tile 1 records the tree sizes, which are per partition.
    red[rowoff + RED_DIRECT, 1] = 0.0
    red[rowoff + RED_APPROX, 1] = 0.0
    if rowoff == 0
        red[RED_LETNODES, 1] = nnodes(tree)
        red[RED_NIMPORT, 1] = nimported(tree)
    end
    p = pfields(buf, bs, n)
    id = mview(meta, bs, M_ID, n)
    key = mview(meta, bs, M_KEY, n)
    counts = InteractionCounts()
    traverse_batch!(p.ax, p.ay, p.az, p.pot, tree, lo:hi, p.x, p.y, p.z, p.m, id,
                    key, params, counts, Int32[])
    red[rowoff + RED_DIRECT, 1] += counts.direct
    red[rowoff + RED_APPROX, 1] += counts.approx
    return nothing
end

function task_kick!(buf, bs::Int, n::Int, dt::Float64)::Nothing
    n == 0 && return nothing
    p = pfields(buf, bs, n)
    kick!(p.vx, p.vy, p.vz, p.ax, p.ay, p.az, dt)
    return nothing
end

function task_energy!(red, buf, bs::Int, n::Int)::Nothing
    p = pfields(buf, bs, n)
    red[RED_KE, 1] = kinetic_energy(p.m, p.vx, p.vy, p.vz)
    s = 0.0
    @inbounds for i in 1:n
        s += p.pot[i]
    end
    red[RED_PE, 1] = s
    return nothing
end

"Where a task actually ran.  Reported to show that work leaves process 1."
task_where(_buf)::Tuple{Int,String} = (myid(), string(Dagger.task_processor()))

task_check_tree(tree, expect_mass::Float64) = check_tree(tree; expect_mass)

function task_csv_piece(path, buf, meta, bs::Int, n::Int)::String
    p = pfields(buf, bs, n)
    id = mview(meta, bs, M_ID, n)
    return write_csv_piece(path, id, p.m, p.x, p.y, p.z, p.vx, p.vy, p.vz)
end

function task_vtp_piece(vs_dir, part::Int, index::Int, buf, meta, bs::Int, n::Int)::String
    p = pfields(buf, bs, n)
    id = mview(meta, bs, M_ID, n)
    return write_vtp_piece(vs_dir, part, index, id, p.m, p.x, p.y, p.z,
                           p.vx, p.vy, p.vz, p.ax, p.ay, p.az)
end

function task_vtk_piece(dir, base, part::Int, nparts::Int, buf, meta, bs::Int,
                        n::Int)::String
    p = pfields(buf, bs, n)
    id = mview(meta, bs, M_ID, n)
    return write_vtk_piece(dir, base, part, nparts, id, p.m, p.x, p.y, p.z,
                           p.vx, p.vy, p.vz, p.ax, p.ay, p.az)
end

# ---------------------------------------------------------------------------
# distributed state
# ---------------------------------------------------------------------------

mutable struct Sim
    cfg::Config
    params::PhysicsParams
    tm::Timers
    n::Int                      # particles
    p::Int                      # partitions
    bs::Int                     # particle slots per partition
    lay::Layout                 # processes x threads (see layout.jl)
    procs::Vector{Dagger.Processor}
    assignment::Vector{Dagger.Processor}
    buf::Dagger.DArray{Float64,1}
    meta::Dagger.DArray{UInt64,1}
    # Scratch with the same blocking as `buf`/`meta`, so a partition's scratch
    # block lives on the process that owns its particles.  The sample sort
    # scatters into it and copies back; a sort cannot be done in place.
    sbuf::Dagger.DArray{Float64,1}
    smeta::Dagger.DArray{UInt64,1}
    red::Dagger.DArray{Float64,2}
    reduced::Matrix{Float64}
    dom::Domain
    boxes::Vector{Box}
    npart::Vector{Int}
    pmass::Vector{Float64}
    cells::Vector{Vector{Box}}
    branch_keys::Vector{Vector{UInt64}}
    ct::CoarseTree
    trees::Vector{Any}
    lets::Vector{Any}
    executed_on::Vector{Tuple{Int,String}}
    step_times::Vector{Float64}
    tree_report::Vector{String}
    invariant_worst::Float64
    import_edges::Int
    snapshot_times::Vector{Float64}
    v_synced::Bool
    ntile::Int                  # tiles per partition, adapted at run time
    last_spawn::Float64         # driver time of the last interaction phase
    last_wait::Float64          # worker time of the last interaction phase
end

partition_size(s::Sim, i::Int) = clamp(s.n - (i - 1) * s.bs, 0, s.bs)

chunk(d::Dagger.DArray{T,1}, i::Int) where {T} = d.chunks[i]
chunk2(d::Dagger.DArray{T,2}, i::Int) where {T} = d.chunks[1, i]

"""
    partition_scope(s, i) -> Dagger.AbstractScope

Scope pinning a task to the processor that owns partition `i`.  It is built
from the very processor objects the `DArray` blocks were assigned to, which the
runtime enumerated itself, so the same code targets a `ThreadProc` under the
distributed backend and the owning rank's processor under Dagger's MPI
"""
partition_scope(s::Sim, i::Int) = partition_scope(s.lay, i, s.cfg.backend)

"""
Tiles are scoped to the *process*, not to a thread.
"""

"Tile ranges of partition `i`, at the configured granularity."
partition_tiles(s::Sim, i::Int) =
    tiles(partition_size(s, i), max(1, min(s.ntile, partition_size(s, i))))


"""
    partition_options(s, i, return_type)

Task options for partition `i`: pinned to the processor owning its data, and
carrying the task's concrete result type.
"""
partition_options(s::Sim, i::Int, return_type::Type) =
    Dagger.Options(; scope=partition_scope(s, i), return_type)

"""
    build_sim(cfg, ps) -> Sim

Distribute a particle set.  The particles are Morton-sorted globally first, so
every partition receives a contiguous Morton range of (almost) equal size --
the balanced initial distribution the benchmark requires, and what makes a
`DArray` block and a local octree cover exactly the same particles.
"""
function build_sim(cfg::Config, ps::ParticleSet)
    n = length(ps)
    # One partition per process; the threads inside a process are used by the
    # tiles of that partition, not by extra partitions.
    lay = build_layout(cfg)
    p = lay.npart
    bs = cld(n, p)
    procs = lay.procs
    assignment = copy(lay.procs)

    # Global Morton sort of the input.  Input time is excluded from the reported
    # runtimes, and the benchmark explicitly allows reading the data set on one
    # node before distributing it.
    dom = bounding_domain((minimum(ps.x), minimum(ps.y), minimum(ps.z)),
                          (maximum(ps.x), maximum(ps.y), maximum(ps.z)))
    keys = Vector{UInt64}(undef, n)
    compute_keys!(keys, dom, ps.x, ps.y, ps.z)
    sort_partition!(keys, ps.id, ps.m, ps.x, ps.y, ps.z, ps.vx, ps.vy, ps.vz)

    # Blocks are padded to `bs` particles so that every block has the same size;
    # the padding slots carry zero mass and are never referenced, because every
    # loop and every octree runs over the partition's real count.
    buf = zeros(Float64, p * NFIELD * bs)
    meta = zeros(UInt64, p * NMETA * bs)
    for i in 1:p
        off = (i - 1) * NFIELD * bs
        moff = (i - 1) * NMETA * bs
        lo = (i - 1) * bs
        cnt = clamp(n - lo, 0, bs)
        for j in 1:cnt
            g = lo + j
            buf[off + (F_M - 1) * bs + j] = ps.m[g]
            buf[off + (F_X - 1) * bs + j] = ps.x[g]
            buf[off + (F_Y - 1) * bs + j] = ps.y[g]
            buf[off + (F_Z - 1) * bs + j] = ps.z[g]
            buf[off + (F_VX - 1) * bs + j] = ps.vx[g]
            buf[off + (F_VY - 1) * bs + j] = ps.vy[g]
            buf[off + (F_VZ - 1) * bs + j] = ps.vz[g]
            meta[moff + (M_KEY - 1) * bs + j] = keys[g]
            meta[moff + (M_ID - 1) * bs + j] = UInt64(ps.id[g])
        end
    end

    d_buf = fetch(Dagger.DVector(buf, Dagger.Blocks(NFIELD * bs), assignment))
    d_meta = fetch(Dagger.DVector(meta, Dagger.Blocks(NMETA * bs), assignment))
    d_sbuf = fetch(Dagger.DVector(zeros(Float64, p * NFIELD * bs),
                                  Dagger.Blocks(NFIELD * bs), assignment))
    d_smeta = fetch(Dagger.DVector(zeros(UInt64, p * NMETA * bs),
                                   Dagger.Blocks(NMETA * bs), assignment))
    # One `NRED` row group per tile: tiles of a partition accumulate their
    # counters side by side instead of contending for one slot, which is what
    # removes the ordering between them.  `reduced_sum` sums the groups back up.
    ntile = tile_count(cfg, lay, bs)
    # Room for the largest tile count the configuration permits, not just the
    # current one: a tile writing past the end would be a silent wrong answer,
    # because `wait` on a failed `DTask` does not re-raise the way `fetch` does.
    nred = NRED * max(ntile, cfg.tiles > 0 ? cfg.tiles : cfg.max_tiles)
    d_red = fetch(Dagger.DMatrix(zeros(nred, p), Dagger.Blocks(nred, 1),
                                 reshape(assignment, 1, p)))

    return Sim(cfg, physics(cfg), Timers(cfg.detailed_timers), n, p, bs, lay, procs,
               assignment, d_buf, d_meta, d_sbuf, d_smeta, d_red, zeros(nred, p), dom,
               fill(EMPTY_BOX, p), zeros(Int, p), zeros(p),
               [Box[] for _ in 1:p], [UInt64[] for _ in 1:p],
               build_coarse_tree(dom, PartitionSummary[]),
               Vector{Any}(undef, p), Vector{Any}(undef, p),
               Tuple{Int,String}[], Float64[], String[], 0.0, 0, Float64[], true,
               ntile, 0.0, 0.0)
end

"Record which process and processor each partition's work actually runs on."
function probe_execution!(s::Sim)
    ts = [Dagger.@spawn scope=partition_scope(s, i) return_type=Tuple{Int,String} task_where(
              chunk(s.buf, i)) for i in 1:s.p]
    # Sequential, and at startup: this runs once, right after addprocs, while
    # Distributed is still forming its mesh -- the worst possible moment to ask
    # for P simultaneous connections, and nothing here is on a hot path.
    s.executed_on = Tuple{Int,String}[fetch(t) for t in ts]
    return s.executed_on
end

# ---------------------------------------------------------------------------
# timestep phases
# ---------------------------------------------------------------------------

"""
    update_domain!(s, dt) -> Domain

Reduce the per-partition post-drift bounding boxes to one cubic domain.  Pure
tasks, gathered by a single Dagger task and fetched once, so every process ends
up with the same domain -- and the same subsequent task graph -- without any
explicit communication.
"""
function update_domain!(s::Sim, dt::Float64)
    ts = [Dagger.@spawn scope=partition_scope(s, i) return_type=PartitionExtent task_post_drift_box(
              chunk(s.buf, i), s.bs, partition_size(s, i), dt) for i in 1:s.p]
    # Same reasoning as `collect_summaries`, including the MPI split.
    parts = s.cfg.backend === :mpi ?
        fetch(Dagger.@spawn task_gather_extents(ts...)) :
        PartitionExtent[fetch(t) for t in ts]
    lo = (Inf, Inf, Inf)
    hi = (-Inf, -Inf, -Inf)
    for i in 1:s.p
        b, cnt, mass = parts[i]
        s.boxes[i] = b
        s.npart[i] = cnt
        s.pmass[i] = mass
        isempty_box(b) && continue
        lo = (min(lo[1], b.xlo), min(lo[2], b.ylo), min(lo[3], b.zlo))
        hi = (max(hi[1], b.xhi), max(hi[2], b.yhi), max(hi[3], b.zhi))
    end
    s.dom = bounding_domain(lo, hi)
    return s.dom
end

"Half kick, drift, Morton keys and local sort: one Datadeps task per partition."
function advance_partitions!(s::Sim, dt::Float64; kick_h::Float64=dt / 2)
    dom = s.dom
    bs = s.bs
    # Kick, drift and re-key are elementwise in the particle index and tile like
    # the force walk: disjoint ranges, no ordering, no Datadeps region.  The sort
    # is not elementwise and stays one task per partition, which is why the two
    # are separate stages with a wait between them.
    phase!(s.tm, "integ.move") do
        moves = Any[]
        for i in 1:s.p
            n = partition_size(s, i)
            n == 0 && continue
            for (t, r) in enumerate(partition_tiles(s, i))
                push!(moves, Dagger.@spawn scope=partition_scope(s, i) return_type=Nothing task_move!(
                    chunk(s.buf, i), chunk(s.meta, i), bs, n, dom, dt, kick_h,
                    first(r), last(r)))
            end
        end
        # Inside the phase: the sort must not start until every tile has
        # re-keyed, and the timer has to cover the work rather than the spawn.
        foreach(wait, moves)
    end
    phase!(s.tm, "integ.sort") do
        sort_partitions!(s)
    end
    return nothing
end

"""
    sort_partitions!(s)

Put every partition back into Morton order, in parallel.
"""
function sort_partitions!(s::Sim)
    bs = s.bs
    pending = Any[]
    for i in 1:s.p
        n = partition_size(s, i)
        n == 0 && continue
        tl = partition_tiles(s, i)
        k = length(tl)
        sc = partition_scope(s, i)
        # Off by default.  The sample sort wins on a few threads but loses badly at
        # high core counts: it costs `3k+4` tasks per partition per step against one,
        # and the driver is charged per task at a rate that grows with the processor
        # count.
        if !s.cfg.parallel_sort || k <= 1 || n <= 1 || s.lay.nthreads < 4
            push!(pending, Dagger.@spawn scope=sc return_type=Nothing task_sort!(
                chunk(s.buf, i), chunk(s.meta, i), bs, n))
            continue
        end
        sp = Dagger.@spawn scope=sc return_type=Vector{UInt64} task_sort_splitters(
            chunk(s.meta, i), bs, n, k)
        counts = [Dagger.@spawn scope=sc return_type=Vector{Int} task_sort_count(
                      chunk(s.meta, i), bs, n, first(r), last(r), sp)
                  for r in tl]
        dest = Dagger.@spawn scope=sc return_type=Vector{Int} task_sort_place(counts...)
        totals = Dagger.@spawn scope=sc return_type=Vector{Int} task_sort_totals(counts...)
        scatters = [Dagger.@spawn scope=sc return_type=Nothing task_sort_scatter!(
                        chunk(s.sbuf, i), chunk(s.smeta, i), chunk(s.buf, i),
                        chunk(s.meta, i), bs, n, first(r), last(r), dest, sp, t)
                    for (t, r) in enumerate(tl)]
        # The bucket sorts read what the scatters wrote, so they have to wait for
        # all of them -- a bucket takes particles from every tile.  Passing the
        # scatter tasks as arguments is what expresses that to the runtime.
        ranges = Dagger.@spawn scope=sc return_type=Vector{UnitRange{Int}} bucket_ranges(
            totals, scatters...)
        for b in 1:k
            push!(pending, Dagger.@spawn scope=sc return_type=Nothing task_sort_bucket_at!(
                chunk(s.buf, i), chunk(s.meta, i), chunk(s.sbuf, i),
                chunk(s.smeta, i), bs, n, ranges, b))
        end
    end
    foreach(wait, pending)
    return nothing
end

"""
    spawn_tree(s, i) -> DTask{Tree}

Partition `i`'s octree, as one task or as a three-stage graph.
"""
function spawn_tree(s::Sim, i::Int)
    cfg = s.cfg
    np = partition_size(s, i)
    # One subtree task per thread, with a floor of four above one thread and a
    # cap of sixteen.  Two tasks is the worst parallel point -- the planner's
    # pieces are uneven enough that two bins leave a noticeable imbalance and the
    # count pass adds about a third to the total work -- so a plain one-task-per-
    # thread rule would pick it exactly where the phase should start scaling.
    # The floor must not apply at one thread, where the decomposition is pure
    # overhead and the single-thread point is the speedup baseline.
    thr = max(s.lay.nthreads, 1)
    ntask = cfg.tree_tasks > 0 ? cfg.tree_tasks :
            thr == 1 ? 1 : clamp(thr, 4, 16)
    ntask = min(ntask, max(np, 1))
    sc = partition_scope(s, i)
    if ntask <= 1 || np == 0
        return Dagger.@spawn scope=sc return_type=Tree task_build_tree(
            chunk(s.buf, i), chunk(s.meta, i), s.bs, np, s.dom,
            cfg.leaf_capacity, cfg.max_depth)
    end
    # Five stages, and no copy anywhere:
    #
    #   plan     split the particle range by work, group the pieces into tasks
    #   count    how many nodes each piece will need   (parallel)
    #   reserve  grow one node array to the final size
    #   fill     build the pieces straight into it     (parallel, disjoint ranges)
    #   finish   centres of mass for the nodes above the split
    #
    # Counting first is what removes the copy: one extra pass over the keys, no
    # allocation, no writes.  `count` and `fill` take the plan, and `fill` the
    # offsets, as task arguments rather than fetched values, so the driver never
    # pulls anything back between stages.
    plan = Dagger.@spawn scope=sc return_type=TreePlan task_tree_plan(
        chunk(s.buf, i), chunk(s.meta, i), s.bs, np, s.dom,
        cfg.leaf_capacity, cfg.max_depth, ntask)
    counts = [Dagger.@spawn scope=sc return_type=Vector{Int} task_tree_count(
                  chunk(s.buf, i), chunk(s.meta, i), s.bs, np, s.dom,
                  cfg.leaf_capacity, cfg.max_depth, plan, k) for k in 1:ntask]
    # `reserve` takes `off` so the runtime sequences it after `offsets`: it grows
    # the upper tree in place, and `offsets` needs the size from before that.
    off = Dagger.@spawn scope=sc return_type=Vector{Int} task_tree_offsets(plan, counts...)
    tree = Dagger.@spawn scope=sc return_type=Tree task_tree_reserve(plan, off, counts...)
    fills = [Dagger.@spawn scope=sc return_type=Nothing task_tree_fill!(
                 tree, chunk(s.buf, i), chunk(s.meta, i), s.bs, np,
                 cfg.leaf_capacity, cfg.max_depth, plan, off, k) for k in 1:ntask]
    return Dagger.@spawn scope=sc return_type=Tree task_tree_finish!(tree, plan, fills...)
end

"Local octrees plus the coarse global tree assembled from their cell summaries."

function build_trees!(s::Sim)
    cfg = s.cfg
    tm = s.tm
    summaries = Vector{Any}(undef, s.p)
    # The three sub-phases separate driver cost from compute: `tree.spawn` and
    # `tree.coarse` run on the driver and grow with the partition count, while
    # `tree.collect` is the wait on the workers.
    phase!(tm, "tree.spawn") do
        for i in 1:s.p
            s.trees[i] = spawn_tree(s, i)
        end
        for i in 1:s.p
            summaries[i] = Dagger.@spawn scope=partition_scope(s, i) return_type=PartitionSummary task_summarize(
                s.trees[i], i, s.boxes[i], partition_size(s, i), branch_budget(cfg, s.p),
                chunk(s.buf, i), s.bs)
        end
    end
    # Wait for the octrees first and the summaries second so the two are timed
    # apart.  Both tasks are already spawned, so this only splits the waiting.
    phase!(tm, "tree.build"; sync = _ -> foreach(wait, s.trees)) do
        nothing
    end
    summ = phase!(tm, "tree.summarize") do
        collect_summaries(summaries, cfg.backend)
    end
    phase!(tm, "tree.coarse") do
        s.ct = build_coarse_tree(s.dom, summ)
        s.cells = target_cells(summ)
        s.branch_keys = [UInt64[b.key for b in ps.branches] for ps in summ]
        check_invariants!(s)
    end
    return s.ct
end

"""
    check_invariants!(s)

Assert, every timestep, that the coarse global tree accounts for exactly the
particles and the mass the partitions hold.
"""
function check_invariants!(s::Sim; rtol::Float64=1e-8)
    part_count = sum(s.npart)
    part_mass = sum(s.pmass)
    ct_count = coarse_total_count(s.ct)
    ct_mass = coarse_total_mass(s.ct)

    part_count == s.n || error("ownership invariant: partitions hold $part_count of " *
                               "$(s.n) particles")
    ct_count == s.n || error("coarse tree invariant: root accounts for $ct_count of " *
                             "$(s.n) particles (partitions hold $part_count)")
    rel = abs(ct_mass - part_mass) / max(abs(part_mass), floatmin())
    rel <= rtol || error("mass invariant: coarse tree root mass $ct_mass vs " *
                         "partition mass $part_mass (relative $rel > $rtol)")
    s.invariant_worst = max(s.invariant_worst, rel)
    return s
end

"""
    interaction_batch(cfg, bs) -> Int

Particles per interaction task.
"""
interaction_batch(cfg::Config, bs::Int) = cfg.task_batch_size > 0 ? cfg.task_batch_size : max(bs, 1)

"""
    branch_budget(cfg, p) -> Int

How many cells a partition publishes into the coarse global tree.
"""
function branch_budget(cfg::Config, p::Int)
    cfg.max_branches > 0 && return cfg.max_branches
    return clamp(2048 ÷ max(p, 1), 16, 128)
end

"""
    export_groups(cfg, p) -> Int
    import_fanin(cfg, p) -> Int

The two knobs of the LET exchange, both counted in Dagger *arguments*: an
argument costs the driver ~0.2 ms regardless of how much data it carries, and at
192 partitions the naive all-to-all is P^2 = 36 000 of them.
"""
function export_groups(cfg::Config, p::Int)
    cfg.export_groups > 0 && return clamp(cfg.export_groups, 1, p)
    # Eight groups, or one per partition when there are fewer.  What sets the
    # useful number of groups is how many requesters have to be pruned against,
    # which is the partition count, so this stays saturated at 8.
    return clamp(8, 1, p)
end

function import_fanin(cfg::Config, p::Int)
    cfg.import_fanin > 0 && return clamp(cfg.import_fanin, 1, p)
    return clamp(8, 1, p)
end

"Export bundles and the per-partition locally essential trees."
function build_lets!(s::Sim)
    tm = s.tm
    # Two sub-phases, both driver work: `let.requests` are the O(P^2 b^2) box
    # tests that decide who needs what from whom, `let.plan` builds the three
    # task stages.  The difference to the whole phase is the wait on the workers.
    requests = phase!(tm, "let.requests") do
        r = import_requests(s.ct, s.cells, s.cfg.theta)
        s.import_edges = import_edges(r)
        r
    end
    phase!(tm, "let.plan") do
        _build_lets_plan!(s, requests)
    end
    return nothing
end

function _build_lets_plan!(s::Sim, requests)
    # A single partition imports nothing and has no remote mass for the coarse
    # tree to stand in for: the local octree already spans the whole domain, so
    # it is this partition's locally essential tree.  Assembling a second copy
    # costs a full tree build per step and changes nothing.
    if s.p == 1
        s.lets[1] = s.trees[1]
        return s.lets
    end
    ng = export_groups(s.cfg, s.p)
    fanin = import_fanin(s.cfg, s.p)
    rsize = cld(s.p, ng)            # requesters per export group
    osize = cld(s.p, fanin)         # owners per merged bundle
    rgroup(i) = min(ng, (i - 1) ÷ rsize + 1)
    rmembers(g) = ((g - 1) * rsize + 1):min(g * rsize, s.p)
    omembers(f) = ((f - 1) * osize + 1):min(f * osize, s.p)

    # Stage 1 -- one bundle per (owner, requester group), pruned against that
    # group's requested keys and target cells only.
    bundles = Dict{Tuple{Int,Int},Any}()
    for j in 1:s.p, g in 1:ng
        gkeys = UInt64[]
        gcells = Box[]
        for i in rmembers(g)
            ks = get(requests[i], Int32(j), nothing)
            ks === nothing && continue
            append!(gkeys, ks)
            append!(gcells, s.cells[i])
        end
        isempty(gkeys) && continue
        unique!(sort!(gkeys))
        bundles[(j, g)] = Dagger.@spawn scope=partition_scope(s, j) return_type=Bundle task_export(
            s.trees[j], gkeys, gcells, s.cfg.theta, chunk(s.buf, j),
            chunk(s.meta, j), s.bs, partition_size(s, j), j)
    end

    # Stage 2 -- concatenate each owner group's bundles.  The merge runs inside
    # the requester group that is about to consume it, spread over its members
    # so that neither the work nor the merged bundle piles up on one process.
    merged = Dict{Tuple{Int,Int},Any}()
    for g in 1:ng, f in 1:fanin
        parts = Any[bundles[(j, g)] for j in omembers(f) if haskey(bundles, (j, g))]
        isempty(parts) && continue
        if length(parts) == 1
            merged[(f, g)] = parts[1]
            continue
        end
        members = rmembers(g)
        host = members[1 + (f - 1) % length(members)]
        merged[(f, g)] = Dagger.@spawn scope=partition_scope(s, host) return_type=Bundle task_merge_bundles(parts...)
    end

    # Stage 3 -- assemble each partition's LET from at most `fanin` bundles.  An
    # owner group is pulled in only if it holds an owner this partition actually
    # requested from.
    for i in 1:s.p
        g = rgroup(i)
        need = Set{Int}(Int(j) for j in import_owners(requests, i))
        imported = Any[merged[(f, g)] for f in 1:fanin
                       if haskey(merged, (f, g)) && any(in(need), omembers(f))]
        s.lets[i] = Dagger.spawn(task_assemble_let, partition_options(s, i, Tree), s.trees[i],
                                 s.ct, s.cells[i], s.branch_keys[i], s.cfg.theta, i,
                                 imported...)
    end
    return s.lets
end

"""
    compute_interactions!(s)

The interaction phase: one Dagger task per batch of target particles, pinned to
the process that owns them.
"""
function compute_interactions!(s::Sim)
    params = s.params
    bs = s.bs
    # Two sub-phases, because they scale in opposite directions and the sum hides
    # it.  `interact.spawn` is the driver handing every tile to the scheduler: it
    # is serial and grows with the tile count and the processor count.
    # `interact.wait` is the work itself and shrinks with the thread count.
    tspawn = time_ns()
    pending = phase!(s.tm, "interact.spawn") do
        out = Any[]
        for i in 1:s.p
            np = partition_size(s, i)
            np == 0 && continue
            for (t, r) in enumerate(partition_tiles(s, i))
                push!(out, Dagger.@spawn scope=partition_scope(s, i) return_type=Nothing task_forces!(
                    chunk(s.buf, i), chunk2(s.red, i), chunk(s.meta, i), bs, np,
                    s.lets[i], first(r), last(r), params, (t - 1) * NRED))
            end
        end
        out
    end
    s.last_spawn = (time_ns() - tspawn) / 1e9
    t0 = time_ns()
    phase!(s.tm, "interact.wait") do
        foreach(wait, pending)
    end
    s.last_wait = (time_ns() - t0) / 1e9
    return nothing
end


"""
    update_accelerations!(s)

Tree construction, LET exchange and the interaction phase: everything needed to
turn the current positions into accelerations and potential contributions.
"""
function update_accelerations!(s::Sim)
    tm = s.tm
    phase!(tm, "tree_construction") do
        build_trees!(s)   # ends with a collective fetch of the (small) summaries
    end
    # `wait`, not `fetch`: under MPI a `fetch` on a DTask is collective and would
    # replicate every tree to every rank.  The wait is a barrier and is not needed
    # for correctness -- `compute_interactions!` passes `s.lets[i]` as a Dagger
    # argument, so each partition's walk is already ordered after its own LET.
    # It exists so the waiting is attributed to the LET phase.  With
    # `overlap_phases` the barrier goes away.
    phase!(tm, "let_exchange"; sync = _ -> foreach(wait, s.lets)) do
        build_lets!(s)
    end
    phase!(tm, "particle_interactions") do
        compute_interactions!(s)
    end
    if s.cfg.verify_tree
        for i in 1:s.p
            r = fetch(Dagger.@spawn scope=partition_scope(s, i) task_check_tree(
                s.lets[i], sum(s.pmass)))
            r.ok || push!(s.tree_report, "partition $i LET: $(r.message)")
        end
    end
    return nothing
end

"Second half kick, one Datadeps task per partition."
function apply_kick!(s::Sim, dt::Float64)
    bs = s.bs
    Dagger.spawn_datadeps() do
        for i in 1:s.p
            Dagger.@spawn task_kick!(Dagger.InOut(chunk(s.buf, i)), bs,
                                     partition_size(s, i), dt)
        end
    end
    return nothing
end

"""
    step!(s, dt)

One kick-drift-kick leapfrog step.  With `dt == 0` the state is unchanged,
which is how the warm-up iterations exercise every code path (and force
compilation) without perturbing the initial condition.
"""
function step!(s::Sim, dt::Float64; sync::Bool=true)
    t0 = time_ns()
    # Kick-drift-kick with the two half kicks either side of a step boundary
    # fused: both use the accelerations just computed, so the pair is one full
    # kick and fusing them removes a Datadeps region per step.  The velocity
    # then lags half a step behind the position; `sync` performs the trailing
    # half kick so energies, snapshots and the final state see synchronised
    # velocities.
    kick_h = s.v_synced ? dt / 2 : dt
    # Two sub-phases, both worker time: `integ.domain` is the O(n) pass that
    # predicts where the particles land after the drift, `integ.advance` the
    # kick / drift / re-key / sort.
    phase!(s.tm, "integration") do
        phase!(s.tm, "integ.domain") do
            update_domain!(s, dt)
        end
        phase!(s.tm, "integ.advance") do
            advance_partitions!(s, dt; kick_h=kick_h)
        end
    end
    s.v_synced = false
    update_accelerations!(s)
    if sync
        phase!(s.tm, "integration") do
            phase!(s.tm, "integ.kick") do
                apply_kick!(s, dt)
            end
        end
        s.v_synced = true
    end
    dtime = (time_ns() - t0) / 1e9
    push!(s.step_times, dtime)
    return dtime
end

"Bring velocities back onto the position's time level, if they are behind."
function synchronize_velocities!(s::Sim, dt::Float64)
    s.v_synced && return s
    phase!(s.tm, "integration") do
        apply_kick!(s, dt)
    end
    s.v_synced = true
    return s
end

"""
    energies(s) -> (kinetic, potential)

Global energies, reduced from the per-partition contributions.  The potential
follows equation 12 of the white paper: `0.5 G sum_i m_i sum_j (-m_j / r)`.
"""
function energies(s::Sim)
    bs = s.bs
    Dagger.spawn_datadeps() do
        for i in 1:s.p
            Dagger.@spawn task_energy!(Dagger.InOut(chunk2(s.red, i)),
                                       Dagger.In(chunk(s.buf, i)), bs,
                                       partition_size(s, i))
        end
    end
    s.reduced = collect(s.red)
    ke = 0.0
    pe = 0.0
    for i in 1:s.p
        ke += s.reduced[RED_KE, i]
        pe += s.reduced[RED_PE, i]
    end
    return ke, 0.5 * s.params.G * pe
end

"Sum of a reduction slot over all partitions."
function reduced_sum(s::Sim, slot::Int)
    total = 0.0
    # Only the groups the last interaction phase actually wrote: the array is
    # sized for the largest granularity, and a run that has since moved to fewer
    # tiles would otherwise add the stale counters of the ones it dropped.
    ntile = min(s.ntile, size(s.reduced, 1) ÷ NRED)
    for i in 1:s.p, t in 0:(ntile - 1)
        total += s.reduced[t * NRED + slot, i]
    end
    return total
end

"""
    prepare(cfg) -> Sim

Set up the backend, load the input, distribute it, and compute the initial
accelerations -- without running a timestep or writing output.  Used by the
tests and by the scaling harness.
"""
function prepare(cfg::Config)
    setup_backend!(cfg)
    s = build_sim(cfg, load_particles(cfg))
    probe_execution!(s)
    update_domain!(s, 0.0)
    advance_partitions!(s, 0.0)
    update_accelerations!(s)
    s.reduced = collect(s.red)     # make the per-partition counters visible
    return s
end

"""
    gather_state(s) -> (ParticleSet, ax, ay, az, pot)

Collect the distributed state onto the calling process.  For tests and
verification only -- it replicates the particle set and is never used on a
simulation path.
"""
function gather_state(s::Sim)
    buf = collect(s.buf)
    meta = collect(s.meta)
    ps = ParticleSet(s.n)
    ax = zeros(s.n); ay = zeros(s.n); az = zeros(s.n); pot = zeros(s.n)
    g = 0
    for i in 1:s.p
        off = (i - 1) * NFIELD * s.bs
        moff = (i - 1) * NMETA * s.bs
        for j in 1:partition_size(s, i)
            g += 1
            ps.id[g] = Int64(meta[moff + (M_ID - 1) * s.bs + j])
            ps.m[g] = buf[off + (F_M - 1) * s.bs + j]
            ps.x[g] = buf[off + (F_X - 1) * s.bs + j]
            ps.y[g] = buf[off + (F_Y - 1) * s.bs + j]
            ps.z[g] = buf[off + (F_Z - 1) * s.bs + j]
            ps.vx[g] = buf[off + (F_VX - 1) * s.bs + j]
            ps.vy[g] = buf[off + (F_VY - 1) * s.bs + j]
            ps.vz[g] = buf[off + (F_VZ - 1) * s.bs + j]
            ax[g] = buf[off + (F_AX - 1) * s.bs + j]
            ay[g] = buf[off + (F_AY - 1) * s.bs + j]
            az[g] = buf[off + (F_AZ - 1) * s.bs + j]
            pot[g] = buf[off + (F_POT - 1) * s.bs + j]
        end
    end
    return ps, ax, ay, az, pot
end

# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

function write_final_state(s::Sim; merge::Bool=true, name::AbstractString="final_state.csv")
    dir = s.cfg.output
    mkpath(dir)
    stem = replace(name, ".csv" => "")
    pieces = String[joinpath(dir, "$stem.part$(lpad(i, 4, '0')).csv") for i in 1:s.p]
    ts = [Dagger.@spawn scope=partition_scope(s, i) return_type=String task_csv_piece(
              pieces[i], chunk(s.buf, i), chunk(s.meta, i), s.bs, partition_size(s, i))
          for i in 1:s.p]
    foreach(fetch, ts)
    # The pieces are written by the processes that own the particles; only the
    # reporting process concatenates them.
    merge || return ""
    return merge_csv_pieces(joinpath(dir, name), pieces)
end

"Directory the visualization files go to: `--vs_dir`, else `<output>/sim_out`."
vs_directory(cfg::Config) = isempty(cfg.vs_dir) ? joinpath(cfg.output, "sim_out") : cfg.vs_dir

"""
    write_snapshot(s, index, t)

One visualization snapshot, written by the processes that own the particles --
nothing is gathered onto the driver.
"""
function write_snapshot(s::Sim, index::Int, t::Float64)
    if s.cfg.vtk_format === :vtp
        dir = vs_directory(s.cfg)
        ts = [Dagger.@spawn scope=partition_scope(s, i) return_type=String task_vtp_piece(
                  dir, i, index, chunk(s.buf, i), chunk(s.meta, i), s.bs,
                  partition_size(s, i)) for i in 1:s.p]
        foreach(fetch, ts)
        push!(s.snapshot_times, t)
        return joinpath(dir, "time_series")
    end
    dir = joinpath(s.cfg.output, "vtk")
    base = "state_$(lpad(index, 6, '0'))"
    ts = [Dagger.@spawn scope=partition_scope(s, i) return_type=String task_vtk_piece(
              dir, base, i, s.p, chunk(s.buf, i), chunk(s.meta, i), s.bs,
              partition_size(s, i)) for i in 1:s.p]
    foreach(fetch, ts)
    push!(s.snapshot_times, t)
    return joinpath(dir, base * ".pvtu")
end

# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

"""
    load_particles(cfg) -> ParticleSet

Read or generate the input.  Every process of the run reads it (the `DArray`
constructors consume the same source array on every rank) and drops it again as
soon as the data is distributed.
"""
function load_particles(cfg::Config)
    ps = if isempty(cfg.input)
        generate_particles(cfg.generate; seed=cfg.seed, kind=cfg.generate_kind, G=cfg.G,
                           total_mass=cfg.total_mass, scale=cfg.length_scale)
    else
        read_csv(cfg.input)
    end
    cfg.recenter_momentum && recenter_momentum!(ps)
    return ps
end

"""
    run_simulation(cfg) -> Dict

Run the whole simulation, write the final state, the optional ParaView
snapshots and `metrics.toml`, and return the summary that went into the metrics
file.
"""
function run_simulation(cfg::Config)
    setup_backend!(cfg)
    report = is_reporter(cfg)
    tm = Timers(cfg.detailed_timers, disabled_timers(cfg))

    t_total = time_ns()
    ps = phase!(tm, "input") do
        load_particles(cfg)
    end
    s = phase!(tm, "input") do
        build_sim(cfg, ps)
    end
    s.tm = tm
    ps = nothing                    # the distributed copy is authoritative now

    report && !cfg.quiet && @info "configuration" particles=s.n partitions=s.p
    probe_execution!(s)

    # Warm-up steps with dt = 0: the same task graph and the same kernels, but no
    # change to the state, so compilation and first-touch costs stay out of the
    # timings and the measured run still starts from the initial condition.
    if cfg.warmup_steps > 0
        for _ in 1:cfg.warmup_steps
            step!(s, 0.0)
        end
        reset!(tm)
        empty!(s.step_times)
    else
        update_domain!(s, 0.0)
        advance_partitions!(s, 0.0)
        update_accelerations!(s)    # a_0 for the first kick
    end

    ke0, pe0 = cfg.energy_interval > 0 ? energies(s) : (NaN, NaN)
    # The reference writes the state before the first step as well, both as a
    # snapshot and as initial_state.csv, so the two runs can be diffed at t = 0.
    if cfg.initial_state_csv
        phase!(tm, "output") do
            write_final_state(s; merge=report, name="initial_state.csv")
        end
    end
    if cfg.vtk_interval > 0
        phase!(tm, "output") do
            write_snapshot(s, 0, 0.0)
        end
    end

    ke = ke0
    pe = pe0
    energy_log = Tuple{Int,Float64,Float64}[]
    for stepno in 1:cfg.steps
        # The trailing half kick is only needed when something is about to read
        # the velocities; otherwise it fuses into the next step's leading half
        # kick and one Datadeps region disappears.
        want_energy = cfg.energy_interval > 0 && stepno % cfg.energy_interval == 0
        want_vtk = cfg.vtk_interval > 0 && stepno % cfg.vtk_interval == 0
        step!(s, cfg.dt; sync=want_energy || want_vtk || stepno == cfg.steps)
        stepno == 1 && snapshot_first_step!(tm)
        if want_energy
            ke, pe = phase!(tm, "energy") do
                energies(s)
            end
            push!(energy_log, (stepno, ke, pe))
        end
        if want_vtk
            phase!(tm, "output") do
                write_snapshot(s, length(s.snapshot_times), stepno * cfg.dt)
            end
        end
    end
    synchronize_velocities!(s, cfg.dt)
    if cfg.energy_interval > 0 && (isempty(energy_log) || energy_log[end][1] != cfg.steps)
        ke, pe = energies(s)
    end
    # The guideline asks for a snapshot of the *final* state whenever
    # visualization is on.  A stride rarely divides the step count, so the last
    # scheduled snapshot is usually not the last step: write one if it is not.
    if cfg.vtk_interval > 0 && cfg.steps % cfg.vtk_interval != 0
        phase!(tm, "output") do
            write_snapshot(s, length(s.snapshot_times), cfg.steps * cfg.dt)
        end
    end

    if cfg.csv_output
        phase!(tm, "output") do
            write_final_state(s; merge=report)
        end
    end
    if cfg.vtk_interval > 0 && cfg.vtk_format === :vtp && report
        write_pvd(vs_directory(cfg), s.snapshot_times, s.p)
    end
    record!(tm, "load_balancing", 0.0)
    s.reduced = collect(s.red)      # pick up the last step's interaction counters
    tm.total = (time_ns() - t_total) / 1e9

    summary = build_summary(s, ke0, pe0, ke, pe, energy_log)
    if report
        write_metrics(joinpath(cfg.output, "metrics.toml"), summary)
        cfg.quiet || print_summary(s, summary)
    end
    isempty(s.tree_report) || @warn "tree invariant problems" s.tree_report
    return summary
end

function build_summary(s::Sim, ke0, pe0, ke, pe, energy_log)
    cfg = s.cfg
    tm = s.tm
    e0 = ke0 + pe0
    e1 = ke + pe
    drift = isfinite(e0) && e0 != 0 ? (e1 - e0) / abs(e0) : NaN
    npart_per = s.npart

    run = Dict{String,Any}(
        "backend" => String(cfg.backend),
        "julia_version" => string(VERSION),
        "dagger_version" => string(pkgversion(Dagger)),
        "dagger_acceleration" => string(Dagger.accel_kind(Dagger.current_acceleration())),
        "transport" => transport_info(cfg),
        "julia_processes" => nprocs(),
        "threads_per_process" => Threads.nthreads(),
        "dagger_cpu_processors" => length(s.procs),
        "compute_owners" => length(unique(proc_owner.(s.procs))),
        "partitions" => s.p,
        # How the partition was actually cut for its threads.  It is derived
        # from --tiles / --tiles_per_thread and the thread count, so recording
        # the flags alone would not say what ran.
        "tiles" => tile_count(cfg, s.lay, partition_size(s, 1)),
        "tree_tasks" => min(cfg.tree_tasks, max(partition_size(s, 1), 1)),
        "particles" => s.n,
        "block_size" => s.bs,
        "particles_per_partition_min" => minimum(npart_per),
        "particles_per_partition_max" => maximum(npart_per),
        "executed_on_processes" => sort!(unique(first.(s.executed_on))),
        "executed_on_processors" => sort!(unique(last.(s.executed_on))),
        "steps" => cfg.steps,
        "warmup_steps" => cfg.warmup_steps,
    )
    timings = timings_dict(tm)
    timings["first_step"] = isempty(s.step_times) ? 0.0 : s.step_times[1]
    timings["first_step_phases"] = first_step_dict(tm)
    timings["steps_total"] = sum(s.step_times; init=0.0)
    timings["step_mean"] = isempty(s.step_times) ? 0.0 :
                           sum(s.step_times) / length(s.step_times)
    timings["step_min"] = isempty(s.step_times) ? 0.0 : minimum(s.step_times)
    timings["step_max"] = isempty(s.step_times) ? 0.0 : maximum(s.step_times)

    energy = Dict{String,Any}(
        "kinetic_initial" => ke0, "potential_initial" => pe0, "total_initial" => e0,
        "kinetic_final" => ke, "potential_final" => pe, "total_final" => e1,
        "relative_drift" => drift,
        "virial_final" => isfinite(pe) && pe != 0 ? 2 * ke / abs(pe) : NaN,
        "history_steps" => [t[1] for t in energy_log],
        "history_kinetic" => [t[2] for t in energy_log],
        "history_potential" => [t[3] for t in energy_log],
        "history_total" => [t[2] + t[3] for t in energy_log],
    )
    interactions = Dict{String,Any}(
        "direct" => reduced_sum(s, RED_DIRECT),
        "approximate" => reduced_sum(s, RED_APPROX),
    )
    tree = Dict{String,Any}(
        "coarse_nodes" => coarse_nnodes(s.ct),
        "coarse_total_mass" => coarse_total_mass(s.ct),
        "global_mass" => sum(s.pmass),
        # Worst relative disagreement between the coarse tree's root mass and the
        # mass the partitions hold, over every timestep.  The run aborts above
        # 1e-8, so a recorded value is proof the check ran and passed.
        "mass_invariant_worst_relative" => s.invariant_worst,
        "let_nodes_total" => reduced_sum(s, RED_LETNODES),
        "imported_particles_total" => reduced_sum(s, RED_NIMPORT),
        "import_edges" => s.import_edges,
        "export_groups" => export_groups(cfg, s.p),
        "task_batch_size_effective" => interaction_batch(cfg, s.bs),
        "max_branches_effective" => branch_budget(cfg, s.p),
        "import_fanin" => import_fanin(cfg, s.p),
        "domain_half_width" => s.dom.half,
    )
    return Dict{String,Any}("parameters" => config_dict(cfg), "run" => run,
                            "timings_seconds" => timings, "energy" => energy,
                            "interactions" => interactions, "tree" => tree)
end

function print_summary(s::Sim, summary::Dict{String,Any})
    run = summary["run"]
    t = summary["timings_seconds"]
    e = summary["energy"]
    it = summary["interactions"]
    println()
    @printf("particles %d  partitions %d  processors %d  backend %s\n",
            run["particles"], run["partitions"], run["dagger_cpu_processors"],
            run["backend"])
    @printf("executed on processes %s, processors %s\n",
            run["executed_on_processes"], run["executed_on_processors"])
    @printf("total %.4f s   steps %.4f s   first step %.4f s   mean step %.4f s\n",
            t["total"], t["steps_total"], t["first_step"], t["step_mean"])
    if s.cfg.detailed_timers
        for p in TIMER_PHASES
            @printf("  %-24s %.4f s\n", p, get(t, p, 0.0))
        end
    end
    @printf("energy total %.10g -> %.10g (relative drift %.3e)\n",
            e["total_initial"], e["total_final"], e["relative_drift"])
    @printf("interactions direct %.0f  approximate %.0f\n", it["direct"], it["approximate"])
    println()
    return nothing
end
