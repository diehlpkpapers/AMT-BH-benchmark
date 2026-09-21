"""
Linear hashable oct-tree (HOT) with bucket leaves, plus the coarse global tree
and the locally essential tree (LET) built on top of it.
"""

# ---------------------------------------------------------------------------
# axis-aligned boxes
# ---------------------------------------------------------------------------

"Axis-aligned bounding box of a set of particles."
struct Box
    xlo::Float64; ylo::Float64; zlo::Float64
    xhi::Float64; yhi::Float64; zhi::Float64
end

const EMPTY_BOX = Box(Inf, Inf, Inf, -Inf, -Inf, -Inf)

function particle_box(x::AbstractVector{Float64}, y::AbstractVector{Float64},
                      z::AbstractVector{Float64})
    isempty(x) && return EMPTY_BOX
    xlo = ylo = zlo = Inf
    xhi = yhi = zhi = -Inf
    @inbounds for i in eachindex(x)
        xlo = min(xlo, x[i]); xhi = max(xhi, x[i])
        ylo = min(ylo, y[i]); yhi = max(yhi, y[i])
        zlo = min(zlo, z[i]); zhi = max(zhi, z[i])
    end
    return Box(xlo, ylo, zlo, xhi, yhi, zhi)
end

isempty_box(b::Box) = b.xhi < b.xlo

function box_union(a::Box, b::Box)
    isempty_box(a) && return b
    isempty_box(b) && return a
    return Box(min(a.xlo, b.xlo), min(a.ylo, b.ylo), min(a.zlo, b.zlo),
               max(a.xhi, b.xhi), max(a.yhi, b.yhi), max(a.zhi, b.zhi))
end

"Squared distance between the closest points of two boxes (0 if they overlap)."
@inline function dist2_boxes(a::Box, b::Box)
    dx = max(0.0, max(a.xlo - b.xhi, b.xlo - a.xhi))
    dy = max(0.0, max(a.ylo - b.yhi, b.ylo - a.yhi))
    dz = max(0.0, max(a.zlo - b.zhi, b.zlo - a.zhi))
    return dx * dx + dy * dy + dz * dz
end

# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------

mutable struct Tree
    dom::Domain
    key::Vector{UInt64}
    level::Vector{Int8}
    mass::Vector{Float64}
    comx::Vector{Float64}
    comy::Vector{Float64}
    comz::Vector{Float64}
    size::Vector{Float64}
    childflat::Vector{Int32}   # 8 entries per node, 0 == absent
    nchild::Vector{Int8}
    lfirst::Vector{Int32}
    lcount::Vector{Int32}
    ifirst::Vector{Int32}
    icount::Vector{Int32}
    ipx::Vector{Float64}
    ipy::Vector{Float64}
    ipz::Vector{Float64}
    ipm::Vector{Float64}
    ipid::Vector{UInt64}
end

Tree(dom::Domain) = Tree(dom, UInt64[], Int8[], Float64[], Float64[], Float64[],
                         Float64[], Float64[], Int32[], Int8[], Int32[], Int32[],
                         Int32[], Int32[], Float64[], Float64[], Float64[],
                         Float64[], UInt64[])

nnodes(t::Tree) = length(t.key)
@inline child(t::Tree, node::Integer, oct::Integer) = t.childflat[8 * (node - 1) + oct + 1]
@inline set_child!(t::Tree, node::Integer, oct::Integer, kid::Integer) =
    (t.childflat[8 * (node - 1) + oct + 1] = Int32(kid))
@inline nparticles(t::Tree, node::Integer) = Int(t.lcount[node]) + Int(t.icount[node])
nimported(t::Tree) = length(t.ipx)

function push_node!(t::Tree, key::UInt64, level::Integer)
    push!(t.key, key)
    push!(t.level, Int8(level))
    push!(t.mass, 0.0)
    push!(t.comx, 0.0); push!(t.comy, 0.0); push!(t.comz, 0.0)
    push!(t.size, cell_size(t.dom, level))
    for _ in 1:8
        push!(t.childflat, Int32(0))
    end
    push!(t.nchild, Int8(0))
    push!(t.lfirst, Int32(0)); push!(t.lcount, Int32(0))
    push!(t.ifirst, Int32(0)); push!(t.icount, Int32(0))
    return length(t.key)
end

"""
    find_node(tree, key) -> Int

Node index of `key`, or `0` if the cell is not in the tree.  Found by
descending the key's octant digits, which costs `O(level)` and needs no
auxiliary index.
"""
function find_node(t::Tree, key::UInt64)
    nnodes(t) == 0 && return 0
    L = key_level(key)
    nd = 1
    for j in 1:L
        oct = Int((key >> (3 * (L - j))) & 0x7)
        nd = Int(child(t, nd, oct))
        nd == 0 && return 0
    end
    return nd
end

# ---------------------------------------------------------------------------
# local tree construction
# ---------------------------------------------------------------------------

"""
    build_local_tree(dom, keys, x, y, z, m, leaf_capacity, max_depth) -> Tree

Build the local oct-tree of one partition.  `keys` must hold the `MAX_LEVEL`
Morton keys of the partition's particles in ascending order, with `x, y, z, m`
permuted the same way (see `sort_partition!`).  Cells split until they hold at
most `leaf_capacity` particles or `max_depth` is reached, so duplicate
"""
function build_local_tree(dom::Domain, keys::AbstractVector{UInt64},
                          x::AbstractVector{Float64}, y::AbstractVector{Float64},
                          z::AbstractVector{Float64}, m::AbstractVector{Float64},
                          leaf_capacity::Integer, max_depth::Integer)
    t = Tree(dom)
    n = length(keys)
    n == 0 && return t
    depth = min(Int(max_depth), MAX_LEVEL)
    # Reserve the node arrays before descending: every node appends to thirteen
    # of them, so growing a push at a time regrows each array repeatedly.  The
    # estimate only has to be close -- a `sizehint!` that falls short just grows
    # as before.  1.5 nodes per particle sits just above the measured count.
    reserve_nodes!(t, max(8, cld(3 * n, 2 * max(Int(leaf_capacity), 1))))
    root = push_node!(t, ROOT_KEY, 0)
    _build!(t, root, keys, x, y, z, m, 1, n, Int(leaf_capacity), depth)
    return t
end

function _build!(t::Tree, node::Int, keys, x, y, z, m, lo::Int, hi::Int,
                 leaf_capacity::Int, max_depth::Int)
    level = Int(t.level[node])
    cnt = hi - lo + 1
    if cnt <= leaf_capacity || level >= max_depth
        t.lfirst[node] = Int32(lo)
        t.lcount[node] = Int32(cnt)
        mass = 0.0; sx = 0.0; sy = 0.0; sz = 0.0
        @inbounds for i in lo:hi
            mass += m[i]; sx += m[i] * x[i]; sy += m[i] * y[i]; sz += m[i] * z[i]
        end
        _set_com!(t, node, mass, sx, sy, sz)
        return
    end

    # Split by the octant digit of the next level.  The keys are sorted, so each
    # octant occupies a contiguous run and one scan suffices.
    shift = 3 * (MAX_LEVEL - level - 1)
    key = t.key[node]
    mass = 0.0; sx = 0.0; sy = 0.0; sz = 0.0
    nkids = 0
    p = lo
    for oct in 0:7
        a = p
        @inbounds while p <= hi && Int((keys[p] >> shift) & 0x7) == oct
            p += 1
        end
        b = p - 1
        b < a && continue
        kid = push_node!(t, key_child(key, oct), level + 1)
        set_child!(t, node, oct, kid)
        nkids += 1
        _build!(t, kid, keys, x, y, z, m, a, b, leaf_capacity, max_depth)
        mk = t.mass[kid]
        mass += mk
        sx += mk * t.comx[kid]; sy += mk * t.comy[kid]; sz += mk * t.comz[kid]
    end
    @assert p == hi + 1 "Morton keys are not sorted inside cell $(key)"
    t.nchild[node] = Int8(nkids)
    _set_com!(t, node, mass, sx, sy, sz)
    return
end

@inline function _set_com!(t::Tree, node::Integer, mass::Float64,
                           sx::Float64, sy::Float64, sz::Float64)
    t.mass[node] = mass
    if mass > 0
        t.comx[node] = sx / mass; t.comy[node] = sy / mass; t.comz[node] = sz / mass
    else
        gx, gy, gz, _ = cell_geometry(t.dom, t.key[node])
        t.comx[node] = gx; t.comy[node] = gy; t.comz[node] = gz
    end
    return
end

# ---------------------------------------------------------------------------
# branch nodes and the coarse global tree
# ---------------------------------------------------------------------------

"""
Summary of one locally owned cell (a *branch node*).
"""
struct BranchNode
    key::UInt64
    mass::Float64
    comx::Float64
    comy::Float64
    comz::Float64
    count::Int32
    box::Box
end

"What one partition publishes about itself every timestep."
struct PartitionSummary
    owner::Int32
    box::Box
    npart::Int32
    mass::Float64
    branches::Vector{BranchNode}
end

"""
    subtree_counts(tree) -> Vector{Int32}

Number of particles below every node, computed bottom-up in one pass.  Nodes
are created parent-before-child, so a reverse sweep sees every child before its
parent.
"""
function subtree_counts(t::Tree)
    cnt = Vector{Int32}(undef, nnodes(t))
    for nd in nnodes(t):-1:1
        c = Int32(nparticles(t, nd))
        for oct in 0:7
            k = child(t, nd, oct)
            k != 0 && (c += cnt[k])
        end
        cnt[nd] = c
    end
    return cnt
end

"""
    branch_nodes(tree; max_branches) -> Vector{Int}

A cut through the local tree with at most `max_branches` cells that together
hold all local particles.  The cut is refined adaptively: the cell holding the
most particles is split first (ties broken by Morton key, so the result is
deterministic), which gives a cut of roughly equal-sized cells instead of a
"""
branch_nodes(t::Tree; max_branches::Integer=128) =
    branch_nodes(t, subtree_counts(t); max_branches)

"""
    branch_nodes(t, cnt; max_branches)

The frontier, given subtree counts that have already been computed.
"""
function branch_nodes(t::Tree, cnt::Vector{Int32}; max_branches::Integer=128)
    nnodes(t) == 0 && return Int[]
    frontier = [1]
    while length(frontier) < max_branches
        best = 0
        bestcnt = Int32(-1)
        bestkey = typemax(UInt64)
        bestkids = 0
        for (slot, nd) in enumerate(frontier)
            kids = Int(t.nchild[nd])
            kids == 0 && continue
            length(frontier) - 1 + kids > max_branches && continue
            c = cnt[nd]
            if c > bestcnt || (c == bestcnt && t.key[nd] < bestkey)
                best = slot; bestcnt = c; bestkey = t.key[nd]; bestkids = kids
            end
        end
        best == 0 && break
        nd = frontier[best]
        deleteat!(frontier, best)
        for oct in 0:7
            c = child(t, nd, oct)
            c != 0 && push!(frontier, Int(c))
        end
    end
    sort!(frontier; by=nd -> t.key[nd])
    return frontier
end

function summarize_partition(t::Tree, owner::Integer, box::Box, npart::Integer,
                             x::AbstractVector{Float64}, y::AbstractVector{Float64},
                             z::AbstractVector{Float64}; max_branches::Integer=128)
    counts = subtree_counts(t)
    nds = branch_nodes(t, counts; max_branches)
    # The branch nodes are a cut of the tree in key order and the particles are
    # in Morton order, so each branch owns a contiguous run and the runs follow
    # one another.  Walking them is all the bounding boxes need.
    branches = Vector{BranchNode}(undef, length(nds))
    total = 0.0
    at = 1
    for (i, nd) in enumerate(nds)
        c = Int(counts[nd])
        r = at:(at + c - 1)
        bx = c > 0 ? particle_box(view(x, r), view(y, r), view(z, r)) : EMPTY_BOX
        branches[i] = BranchNode(t.key[nd], t.mass[nd], t.comx[nd], t.comy[nd],
                                 t.comz[nd], counts[nd], bx)
        total += t.mass[nd]
        at += c
    end
    at == length(x) + 1 || at == 1 ||
        error("branch cut covers $(at - 1) of $(length(x)) particles")
    return PartitionSummary(Int32(owner), box, Int32(npart), total, branches)
end

"""
    subtree_boxes(tree, x, y, z) -> Vector{Box}

Bounding box of the particles below every node, in one bottom-up pass (nodes
are created parent-before-child, so a reverse sweep sees every child first).
"""
function subtree_boxes(t::Tree, x::AbstractVector{Float64}, y::AbstractVector{Float64},
                       z::AbstractVector{Float64})
    boxes = Vector{Box}(undef, nnodes(t))
    for nd in nnodes(t):-1:1
        b = EMPTY_BOX
        lc = Int(t.lcount[nd])
        if lc > 0
            lf = Int(t.lfirst[nd])
            r = lf:(lf + lc - 1)
            b = box_union(b, particle_box(view(x, r), view(y, r), view(z, r)))
        end
        for oct in 0:7
            c = child(t, nd, oct)
            c != 0 && (b = box_union(b, boxes[c]))
        end
        boxes[nd] = b
    end
    return boxes
end

"""
    target_cells(summaries) -> Vector{Vector{Box}}

The target geometry of every partition: the bounding boxes of the cells it
published.  A node accepted for all of them is accepted by every particle of
that partition, so it can stay a summary node in the partition's LET.
"""
function target_cells(summaries::Vector{PartitionSummary})
    return [Box[b.box for b in s.branches if !isempty_box(b.box)] for s in summaries]
end

"""
    prune_accept(cx, cy, cz, size, cells, theta2) -> Bool

Can a cell centred at `(cx, cy, cz)` with edge length `size` be accepted as a
single pseudo-body by *every* target cell in `cells`?
"""
@inline function prune_accept(cx::Float64, cy::Float64, cz::Float64, size::Float64,
                              cells::Vector{Box}, theta2::Float64)
    theta2 <= 0 && return false
    isempty(cells) && return true
    half = 0.5 * size
    node = Box(cx - half, cy - half, cz - half, cx + half, cy + half, cz + half)
    s2 = size * size
    @inbounds for c in cells
        s2 < theta2 * dist2_boxes(c, node) || return false
    end
    return true
end

"Centre of the `oct`-th child of a cell centred at `(cx, cy, cz)` with edge `size`."
@inline function child_centre(cx::Float64, cy::Float64, cz::Float64, size::Float64,
                              oct::Integer)
    q = 0.25 * size
    return (cx + ((oct >> 2) & 1 == 1 ? q : -q),
            cy + ((oct >> 1) & 1 == 1 ? q : -q),
            cz + ((oct & 1) == 1 ? q : -q))
end

"""
The coarse global tree: the top of the global oct-tree, from the root down to
the branch nodes published by every partition.  It has `O(P * max_branches)`
nodes and is identical on every process, which is what lets each partition
decide -- locally and without communication -- which remote subtrees it needs.
"""
struct CoarseTree
    dom::Domain
    keys::Vector{UInt64}            # sorted ascending
    mass::Vector{Float64}
    comx::Vector{Float64}
    comy::Vector{Float64}
    comz::Vector{Float64}
    size::Vector{Float64}
    count::Vector{Int32}
    isbranch::Vector{Bool}
    owners::Vector{Vector{Int32}}   # partitions owning the subtree below the key
    index::Dict{UInt64,Int32}
end

coarse_nnodes(ct::CoarseTree) = length(ct.keys)
coarse_index(ct::CoarseTree, key::UInt64) = get(ct.index, key, Int32(0))
coarse_total_mass(ct::CoarseTree) = isempty(ct.mass) ? 0.0 : ct.mass[coarse_index(ct, ROOT_KEY)]
"Particles accounted for by the coarse tree's root cell."
coarse_total_count(ct::CoarseTree) =
    isempty(ct.count) ? 0 : Int(ct.count[coarse_index(ct, ROOT_KEY)])

"""
    build_coarse_tree(dom, summaries) -> CoarseTree

Merge the branch nodes published by all partitions and accumulate them into
every ancestor cell up to the root.  A cell published by several partitions
(which happens where a cell straddles a partition boundary) is merged and keeps
its full owner list, so an opening request goes to *all* owners and no mass is
"""
function build_coarse_tree(dom::Domain, summaries::Vector{PartitionSummary})
    macc = Dict{UInt64,Float64}()
    sx = Dict{UInt64,Float64}(); sy = Dict{UInt64,Float64}(); sz = Dict{UInt64,Float64}()
    cnt = Dict{UInt64,Int32}()
    owners = Dict{UInt64,Vector{Int32}}()
    branch = Set{UInt64}()

    for s in summaries
        for b in s.branches
            push!(branch, b.key)
            ow = get!(() -> Int32[], owners, b.key)
            s.owner in ow || push!(ow, s.owner)
            k = b.key
            while true
                macc[k] = get(macc, k, 0.0) + b.mass
                sx[k] = get(sx, k, 0.0) + b.mass * b.comx
                sy[k] = get(sy, k, 0.0) + b.mass * b.comy
                sz[k] = get(sz, k, 0.0) + b.mass * b.comz
                cnt[k] = get(cnt, k, Int32(0)) + b.count
                k == ROOT_KEY && break
                k = key_parent(k)
            end
        end
    end

    allkeys = sort!(collect(UInt64, keys(macc)))
    n = length(allkeys)
    ct = CoarseTree(dom, allkeys,
                    Vector{Float64}(undef, n), Vector{Float64}(undef, n),
                    Vector{Float64}(undef, n), Vector{Float64}(undef, n),
                    Vector{Float64}(undef, n), Vector{Int32}(undef, n),
                    Vector{Bool}(undef, n), Vector{Vector{Int32}}(undef, n),
                    Dict{UInt64,Int32}())
    for (i, k) in enumerate(allkeys)
        mk = macc[k]
        ct.mass[i] = mk
        if mk > 0
            ct.comx[i] = sx[k] / mk; ct.comy[i] = sy[k] / mk; ct.comz[i] = sz[k] / mk
        else
            gx, gy, gz, _ = cell_geometry(dom, k)
            ct.comx[i] = gx; ct.comy[i] = gy; ct.comz[i] = gz
        end
        ct.size[i] = cell_size(dom, key_level(k))
        ct.count[i] = cnt[k]
        ct.isbranch[i] = k in branch
        ct.owners[i] = get(owners, k, Int32[])
        ct.index[k] = Int32(i)
    end
    return ct
end

# ---------------------------------------------------------------------------
# import planning
# ---------------------------------------------------------------------------

"""
    import_requests(ct, cells, theta) -> requests

`requests[i]` maps each owner partition `j` to the list of `j`'s published cell
keys that partition `i` may have to open.  It follows from the (replicated)
coarse tree alone, so every process computes the same answer without
communicating -- which is what keeps the task graph SPMD-uniform under MPI.
"""
function import_requests(ct::CoarseTree, cells::Vector{Vector{Box}}, theta::Float64)
    P = length(cells)
    theta2 = theta * theta
    requests = [Dict{Int32,Vector{UInt64}}() for _ in 1:P]
    for n in 1:coarse_nnodes(ct)
        ct.isbranch[n] || continue
        owners = ct.owners[n]
        key = ct.keys[n]
        gx, gy, gz, _ = cell_geometry(ct.dom, key)
        for i in 1:P
            isempty(cells[i]) && continue
            # A cell that every target cell of partition `i` accepts as a
            # pseudo-body never has to be opened, so nothing has to be imported.
            prune_accept(gx, gy, gz, ct.size[n], cells[i], theta2) && continue
            for o in owners
                Int(o) == i && continue     # our own subtree is already local
                push!(get!(() -> UInt64[], requests[i], o), key)
            end
        end
    end
    for r in requests
        for v in values(r)
            sort!(v)
        end
    end
    return requests
end

"Owners partition `i` imports from, in ascending order."
import_owners(requests::Vector{Dict{Int32,Vector{UInt64}}}, i::Integer) =
    sort!(collect(keys(requests[i])))

"Total number of (requester, owner) import pairs -- reported as a diagnostic."
import_edges(requests::Vector{Dict{Int32,Vector{UInt64}}}) =
    sum(length, requests; init=0)

# ---------------------------------------------------------------------------
# export bundles
# ---------------------------------------------------------------------------

"""
Pruned set of tree nodes that one partition ships to the partitions which
requested it: a flat, key-addressed node list plus the particles of exported
leaves.  Child links are not transferred; the receiver rebuilds them from keys.
"""
struct Bundle
    owner::Int32
    nowner::Vector{Int32}       # owner of each node: a merged bundle mixes several
    key::Vector{UInt64}
    mass::Vector{Float64}
    comx::Vector{Float64}
    comy::Vector{Float64}
    comz::Vector{Float64}
    pfirst::Vector{Int32}
    pcount::Vector{Int32}
    px::Vector{Float64}
    py::Vector{Float64}
    pz::Vector{Float64}
    pm::Vector{Float64}
    pid::Vector{UInt64}
end

Bundle(owner::Integer) = Bundle(Int32(owner), Int32[], UInt64[], Float64[], Float64[],
                                Float64[], Float64[], Int32[], Int32[], Float64[],
                                Float64[], Float64[], Float64[], UInt64[])

bundle_nnodes(b::Bundle) = length(b.key)
bundle_nparticles(b::Bundle) = length(b.px)

"""
    export_bundle(tree, keys, cells, theta, x, y, z, m, id, owner) -> Bundle

Export the subtrees below `keys`, pruned against the target cells of the
requesting partitions: a node is expanded only when at least one requesting
cell fails to accept it, so every requester receives all nodes its own
traversal can reach.  Leaves carry their particles.
"""
function export_bundle(t::Tree, request_keys::Vector{UInt64}, cells::Vector{Box},
                       theta::Float64, x::AbstractVector{Float64},
                       y::AbstractVector{Float64}, z::AbstractVector{Float64},
                       m::AbstractVector{Float64}, id::AbstractVector{<:Integer},
                       owner::Integer)
    b = Bundle(owner)
    (nnodes(t) == 0 || isempty(request_keys)) && return b
    theta2 = theta * theta
    for k in request_keys
        nd = find_node(t, k)
        nd == 0 && continue
        gx, gy, gz, _ = cell_geometry(t.dom, k)
        _export!(b, t, nd, gx, gy, gz, cells, theta2, x, y, z, m, id, true)
    end
    resize!(b.nowner, length(b.key))
    fill!(b.nowner, Int32(owner))
    return b
end

function _export!(b::Bundle, t::Tree, node::Int, cx::Float64, cy::Float64, cz::Float64,
                  cells::Vector{Box}, theta2::Float64, x, y, z, m, id, isroot::Bool)
    push!(b.key, t.key[node])
    push!(b.mass, isroot ? 0.0 : t.mass[node])
    push!(b.comx, t.comx[node]); push!(b.comy, t.comy[node]); push!(b.comz, t.comz[node])

    # A node every requesting cell accepts as a pseudo-body is a stopping point:
    # neither its children nor its particles are needed, since the receiver uses
    # its centre of mass.
    accepted = prune_accept(cx, cy, cz, t.size[node], cells, theta2)
    lc = accepted ? 0 : Int(t.lcount[node])
    if lc > 0
        lf = Int(t.lfirst[node])
        push!(b.pfirst, Int32(length(b.px) + 1))
        push!(b.pcount, Int32(lc))
        @inbounds for i in lf:(lf + lc - 1)
            push!(b.px, x[i]); push!(b.py, y[i]); push!(b.pz, z[i])
            push!(b.pm, m[i]); push!(b.pid, UInt64(id[i]))
        end
    else
        push!(b.pfirst, Int32(0)); push!(b.pcount, Int32(0))
    end

    (accepted || t.nchild[node] == 0) && return
    for oct in 0:7
        c = child(t, node, oct)
        c == 0 && continue
        kx, ky, kz = child_centre(cx, cy, cz, t.size[node], oct)
        _export!(b, t, Int(c), kx, ky, kz, cells, theta2, x, y, z, m, id, false)
    end
    return
end

"""
    merge_bundles(bundles) -> Bundle

Concatenate export bundles into one.
"""
function merge_bundles(bundles::Vector{Bundle})
    length(bundles) == 1 && return bundles[1]
    isempty(bundles) && return Bundle(0)
    out = Bundle(bundles[1].owner)
    nn = sum(bundle_nnodes, bundles)
    np = sum(bundle_nparticles, bundles)
    for v in (out.key, out.mass, out.comx, out.comy, out.comz)
        sizehint!(v, nn)
    end
    for v in (out.nowner, out.pfirst, out.pcount)
        sizehint!(v, nn)
    end
    for v in (out.px, out.py, out.pz, out.pm, out.pid)
        sizehint!(v, np)
    end
    poff = 0
    for b in bundles
        append!(out.nowner, b.nowner)
        append!(out.key, b.key); append!(out.mass, b.mass)
        append!(out.comx, b.comx); append!(out.comy, b.comy); append!(out.comz, b.comz)
        @inbounds for nd in 1:bundle_nnodes(b)
            c = b.pcount[nd]
            push!(out.pfirst, c > 0 ? b.pfirst[nd] + Int32(poff) : Int32(0))
            push!(out.pcount, c)
        end
        append!(out.px, b.px); append!(out.py, b.py); append!(out.pz, b.pz)
        append!(out.pm, b.pm); append!(out.pid, b.pid)
        poff += bundle_nparticles(b)
    end
    return out
end

# ---------------------------------------------------------------------------
# locally essential tree
# ---------------------------------------------------------------------------

"""
Per-key merge state used while assembling a LET.
"""
mutable struct MergeState
    index::Dict{UInt64,Int32}
    key::Vector{UInt64}
    mass::Vector{Float64}
    sx::Vector{Float64}
    sy::Vector{Float64}
    sz::Vector{Float64}
    lfirst::Vector{Int32}
    lcount::Vector{Int32}
    ihead::Vector{Int32}        # first imported run of the node, 0 if none
    iprun_bundle::Vector{Int32}
    iprun_first::Vector{Int32}
    iprun_count::Vector{Int32}
    iprun_next::Vector{Int32}
end

function MergeState(nhint::Int)
    ms = MergeState(Dict{UInt64,Int32}(), UInt64[], Float64[], Float64[], Float64[],
                    Float64[], Int32[], Int32[], Int32[], Int32[], Int32[], Int32[],
                    Int32[])
    sizehint!(ms.index, nhint)
    for v in (ms.key, ms.mass, ms.sx, ms.sy, ms.sz, ms.lfirst, ms.lcount, ms.ihead)
        sizehint!(v, nhint)
    end
    return ms
end

Base.haskey(ms::MergeState, k::UInt64) = haskey(ms.index, k)
merge_index(ms::MergeState, k::UInt64) = get(ms.index, k, Int32(0))

"Add a contribution for `key`, returning its node index."
function merge_add!(ms::MergeState, key::UInt64, mass::Float64, comx::Float64,
                    comy::Float64, comz::Float64)
    idx = get(ms.index, key, Int32(0))
    if idx == 0
        push!(ms.key, key)
        push!(ms.mass, mass)
        push!(ms.sx, mass * comx); push!(ms.sy, mass * comy); push!(ms.sz, mass * comz)
        push!(ms.lfirst, Int32(0)); push!(ms.lcount, Int32(0))
        push!(ms.ihead, Int32(0))
        idx = Int32(length(ms.key))
        ms.index[key] = idx
    else
        ms.mass[idx] += mass
        ms.sx[idx] += mass * comx; ms.sy[idx] += mass * comy; ms.sz[idx] += mass * comz
    end
    return idx
end

"Attach an imported particle run (a slice of bundle `bi`) to node `idx`."
function merge_import!(ms::MergeState, idx::Int32, bi::Integer, first::Integer,
                       count::Integer)
    push!(ms.iprun_bundle, Int32(bi))
    push!(ms.iprun_first, Int32(first))
    push!(ms.iprun_count, Int32(count))
    push!(ms.iprun_next, ms.ihead[idx])
    ms.ihead[idx] = Int32(length(ms.iprun_bundle))
    return ms
end

"""
    assemble_let(ct, local_tree, bundles, cells, branch_keys, theta) -> Tree

Build the locally essential tree of one partition: the coarse global tree,
refined by this partition's own subtree and by the imported subtrees, pruned
with the cell-level opening criterion.
"""
function assemble_let(ct::CoarseTree, local_tree::Tree, bundles::Vector{Bundle},
                      cells::Vector{Box}, branch_keys::Vector{UInt64}, theta::Float64,
                      self::Integer=0)
    t = Tree(ct.dom)
    theta2 = theta * theta

    nbundle = sum(bundle_nnodes, bundles; init=0)
    nimport = sum(bundle_nparticles, bundles; init=0)
    ms = MergeState(nnodes(local_tree) + nbundle)

    # The local subtree contributes exactly like an imported bundle: nodes at or
    # above this partition's branch cut carry mass 0.0 (the coarse tree already
    # accounts for them), nodes below the cut carry their real mass.
    if nnodes(local_tree) > 0
        _add_local!(ms, local_tree, 1, Set{UInt64}(branch_keys), false)
    end
    # A merged bundle can contain this partition's own exported subtree, because
    # owner groups are shared by a whole group of requesters.  Adding it would
    # count our mass twice: `_add_local!` above already contributed it.
    me = Int32(self)
    for (bi, b) in enumerate(bundles)
        for nd in 1:bundle_nnodes(b)
            b.nowner[nd] == me && continue
            idx = merge_add!(ms, b.key[nd], b.mass[nd], b.comx[nd], b.comy[nd], b.comz[nd])
            b.pcount[nd] > 0 && merge_import!(ms, idx, bi, b.pfirst[nd], b.pcount[nd])
        end
    end

    # Preallocate: the LET holds at most the coarse tree plus every available
    # node, and at most every imported particle.
    reserve_nodes!(t, coarse_nnodes(ct) + length(ms.key))
    for v in (t.ipx, t.ipy, t.ipz, t.ipm)
        sizehint!(v, nimport)
    end
    sizehint!(t.ipid, nimport)

    root = push_node!(t, ROOT_KEY, 0)
    gx, gy, gz, _ = cell_geometry(ct.dom, ROOT_KEY)
    _emit_let!(t, root, ROOT_KEY, gx, gy, gz, ct, ms, bundles, cells, theta2)
    return t
end

"Reserve capacity for `n` nodes without creating them."
function reserve_nodes!(t::Tree, n::Integer)
    sizehint!(t.key, n); sizehint!(t.level, n); sizehint!(t.mass, n)
    sizehint!(t.comx, n); sizehint!(t.comy, n); sizehint!(t.comz, n)
    sizehint!(t.size, n); sizehint!(t.childflat, 8n); sizehint!(t.nchild, n)
    sizehint!(t.lfirst, n); sizehint!(t.lcount, n)
    sizehint!(t.ifirst, n); sizehint!(t.icount, n)
    return t
end

"""
Add the local subtree's contributions to `avail`.  `below` becomes true once the
walk has passed this partition's branch cut; at or above the cut the mass is
already contained in the coarse tree, so only the particles are contributed.
"""
function _add_local!(ms::MergeState, t::Tree, node::Int, branchset::Set{UInt64},
                     below::Bool)
    key = t.key[node]
    mass = below ? t.mass[node] : 0.0
    idx = merge_add!(ms, key, mass, t.comx[node], t.comy[node], t.comz[node])
    ms.lfirst[idx] = t.lfirst[node]
    ms.lcount[idx] = t.lcount[node]
    nowbelow = below || (key in branchset)
    for oct in 0:7
        c = child(t, node, oct)
        c != 0 && _add_local!(ms, t, Int(c), branchset, nowbelow)
    end
    return ms
end

function _emit_let!(t::Tree, node::Int, key::UInt64, cx::Float64, cy::Float64,
                    cz::Float64, ct::CoarseTree, ms::MergeState,
                    bundles::Vector{Bundle}, cells::Vector{Box}, theta2::Float64)
    ci = coarse_index(ct, key)
    ai = merge_index(ms, key)
    (ci != 0 || ai != 0) || return false

    # A cell's mass is the sum of two disjoint contributions: what the coarse
    # tree knows, and what the sources hold below their own cut (bundle and
    # local-subtree contributions, whose roots carry mass 0.0).  Every particle
    # is therefore counted exactly once per cell.
    mass = 0.0; sx = 0.0; sy = 0.0; sz = 0.0
    if ci != 0
        m = ct.mass[ci]
        mass += m
        sx += m * ct.comx[ci]; sy += m * ct.comy[ci]; sz += m * ct.comz[ci]
    end
    if ai != 0
        mass += ms.mass[ai]
        sx += ms.sx[ai]; sy += ms.sy[ai]; sz += ms.sz[ai]
    end
    if mass > 0
        t.comx[node] = sx / mass; t.comy[node] = sy / mass; t.comz[node] = sz / mass
    else
        t.comx[node] = cx; t.comy[node] = cy; t.comz[node] = cz
    end
    t.mass[node] = mass

    has_kids = false
    for oct in 0:7
        ck = key_child(key, oct)
        if haskey(ms, ck) || haskey(ct.index, ck)
            has_kids = true
            break
        end
    end

    # Accepted by every target cell: keep as a summary node and drop everything
    # below it.
    if has_kids && prune_accept(cx, cy, cz, t.size[node], cells, theta2)
        return true
    end

    # Attach leaf particles (a requester that opens a leaf sums it directly).
    if ai != 0
        if ms.lcount[ai] > 0
            t.lfirst[node] = ms.lfirst[ai]
            t.lcount[node] = ms.lcount[ai]
        end
        run = ms.ihead[ai]
        if run != 0
            first_idx = nimported(t) + 1
            total = 0
            while run != 0
                b = bundles[ms.iprun_bundle[run]]
                lo = Int(ms.iprun_first[run])
                cnt = Int(ms.iprun_count[run])
                hi = lo + cnt - 1
                append!(t.ipx, view(b.px, lo:hi)); append!(t.ipy, view(b.py, lo:hi))
                append!(t.ipz, view(b.pz, lo:hi)); append!(t.ipm, view(b.pm, lo:hi))
                append!(t.ipid, view(b.pid, lo:hi))
                total += cnt
                run = ms.iprun_next[run]
            end
            t.ifirst[node] = Int32(first_idx)
            t.icount[node] = Int32(total)
        end
    end

    has_kids || return true

    nkids = 0
    for oct in 0:7
        ck = key_child(key, oct)
        (haskey(ms, ck) || haskey(ct.index, ck)) || continue
        kx, ky, kz = child_centre(cx, cy, cz, t.size[node], oct)
        kid = push_node!(t, ck, Int(t.level[node]) + 1)
        if _emit_let!(t, kid, ck, kx, ky, kz, ct, ms, bundles, cells, theta2)
            set_child!(t, node, oct, kid)
            nkids += 1
        else
            _pop_node!(t)
        end
    end
    t.nchild[node] = Int8(nkids)
    return true
end

function _pop_node!(t::Tree)
    pop!(t.key); pop!(t.level); pop!(t.mass)
    pop!(t.comx); pop!(t.comy); pop!(t.comz); pop!(t.size)
    resize!(t.childflat, length(t.childflat) - 8)
    pop!(t.nchild)
    pop!(t.lfirst); pop!(t.lcount); pop!(t.ifirst); pop!(t.icount)
    return
end

# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------

"""
    check_tree(tree) -> NamedTuple

Verify the structural invariants relied on elsewhere: a node's key level
matches its stored level, every child key is the parent's key extended by its
octant, `nchild` agrees with the child links, and an internal node's mass
equals the sum of its children's masses plus any particles it carries directly.
"""
function check_tree(t::Tree; mass_tol::Float64=1e-10, expect_mass::Float64=NaN)
    nnodes(t) == 0 && return (; ok=true, nodes=0, mass=0.0, particles=0, message="empty")
    problems = String[]
    total_particles = 0
    for nd in 1:nnodes(t)
        key = t.key[nd]
        lvl = Int(t.level[nd])
        key_level(key) == lvl || push!(problems, "node $nd: level $lvl != key level $(key_level(key))")
        childmass = 0.0
        nkids = 0
        for oct in 0:7
            c = Int(child(t, nd, oct))
            c == 0 && continue
            nkids += 1
            t.key[c] == key_child(key, oct) || push!(problems, "node $nd: child $oct key mismatch")
            childmass += t.mass[c]
        end
        nkids == Int(t.nchild[nd]) || push!(problems, "node $nd: nchild $(t.nchild[nd]) != $nkids")
        if nkids > 0 && nparticles(t, nd) == 0
            rel = abs(childmass - t.mass[nd]) / max(abs(t.mass[nd]), 1e-300)
            rel > mass_tol && push!(problems,
                "node $nd: mass $(t.mass[nd]) != child sum $childmass (rel $rel)")
        end
        total_particles += nparticles(t, nd)
    end
    # The root of a locally essential tree carries the mass of the whole system.
    # This is the check that catches a subtree counted twice or not at all,
    # which the per-node tests above cannot see: they stay consistent when a
    # whole subtree is duplicated.
    if !isnan(expect_mass) && expect_mass > 0
        rel = abs(t.mass[1] - expect_mass) / expect_mass
        rel > 1e-8 && push!(problems,
            "root mass $(t.mass[1]) != global mass $expect_mass (rel $rel)")
    end

    return (; ok=isempty(problems), nodes=nnodes(t), mass=t.mass[1],
            particles=total_particles,
            message=isempty(problems) ? "ok" : join(problems, "; "))
end

# ---------------------------------------------------------------------------
# Dagger integration
# ---------------------------------------------------------------------------

const TREE_ARRAY_FIELDS = (:key, :level, :mass, :comx, :comy, :comz, :size,
                           :childflat, :nchild, :lfirst, :lcount, :ifirst, :icount,
                           :ipx, :ipy, :ipz, :ipm, :ipid)

"""
    Dagger.move!(dep_mod, to_space, from_space, to::Tree, from::Tree)

In-place data movement for trees inside a Datadeps region.  Tree and LET tasks
are pinned to the processor owning their partition, so this is normally not
needed; defining it means a scheduling decision that does place a consumer
elsewhere copies the tree instead of failing.
"""
function Dagger.move!(dep_mod, to_space::Dagger.MemorySpace, from_space::Dagger.MemorySpace,
                      to::Tree, from::Tree)
    to.dom = from.dom
    for f in TREE_ARRAY_FIELDS
        dst = getfield(to, f)
        src = getfield(from, f)
        resize!(dst, length(src))
        copyto!(dst, src)
    end
    return nothing
end
