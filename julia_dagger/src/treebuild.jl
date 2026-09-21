"""
Local tree construction as a Dagger task graph.
"""

"A cell left for a subtree task: its node index in the upper tree and its particle range."
struct TreeStub
    node::Int
    lo::Int
    hi::Int
end

"""
Upper tree plus the stubs whose subtrees are still missing.
"""
struct TreePlan
    upper::Tree
    stubs::Vector{TreeStub}
    group_first::Vector{Int}      # length ngroup+1, the usual CSR-style offsets
end

TreePlan(upper::Tree, stubs::Vector{TreeStub}) =
    TreePlan(upper, stubs, collect(1:(length(stubs) + 1)))

"Number of subtree tasks this plan asks for."
ngroups(plan::TreePlan) = length(plan.group_first) - 1

"Stub indices belonging to subtree task `k`."
group_range(plan::TreePlan, k::Int) = plan.group_first[k]:(plan.group_first[k + 1] - 1)

"""
    task_tree_plan(buf, meta, bs, n, dom, leaf_capacity, max_depth, target) -> TreePlan

Descend until at least `target` cells are pending, leaving them as stubs.
"""
function task_tree_plan(buf, meta, bs::Int, n::Int, dom::Domain,
                        leaf_capacity::Int, max_depth::Int, target::Int)::TreePlan
    p = pfields(buf, bs, n)
    keys = mview(meta, bs, M_KEY, n)
    depth = min(max_depth, MAX_LEVEL)
    t = Tree(dom)
    n == 0 && return TreePlan(t, TreeStub[])
    root = push_node!(t, ROOT_KEY, 0)
    target = max(1, target)

    # Split until no pending cell holds more than its fair share of the particles,
    # not until there are `target` cells.  Counting cells does nothing on a
    # concentrated distribution: one octant of the root can hold almost
    # everything, the count reaches the target after one split, and one piece
    # still holds most of the tree.  Splitting by work produces more pieces than
    # tasks, which is what `group_first` is for.
    limit = max(leaf_capacity, cld(n, target))
    pending = [TreeStub(root, 1, n)]
    # Descending forever on degenerate data would cost the driver more than the
    # imbalance does, so cap the pieces well above the task count but not
    # unboundedly.
    maxpieces = 16 * target
    while length(pending) < maxpieces
        best = 0
        bestcnt = -1
        for (slot, st) in enumerate(pending)
            level = Int(t.level[st.node])
            cnt = st.hi - st.lo + 1
            (cnt <= leaf_capacity || level >= depth || cnt <= limit) && continue
            cnt > bestcnt && (best = slot; bestcnt = cnt)
        end
        best == 0 && break                       # everything is small enough

        st = pending[best]
        level = Int(t.level[st.node])
        shift = 3 * (MAX_LEVEL - level - 1)
        key = t.key[st.node]
        kids = TreeStub[]
        q = st.lo
        for oct in 0:7
            a = q
            @inbounds while q <= st.hi && Int((keys[q] >> shift) & 0x7) == oct
                q += 1
            end
            b = q - 1
            b < a && continue
            kid = push_node!(t, key_child(key, oct), level + 1)
            set_child!(t, st.node, oct, kid)
            push!(kids, TreeStub(kid, a, b))
        end
        t.nchild[st.node] = Int8(length(kids))
        deleteat!(pending, best)
        append!(pending, kids)
    end

    # Cells the real stop rule closes are finished here, exactly as the serial
    # build would; only the rest go out as stubs.
    stubs = TreeStub[]
    for st in pending
        level = Int(t.level[st.node])
        cnt = st.hi - st.lo + 1
        if cnt <= leaf_capacity || level >= depth
            _close_leaf!(t, st.node, st.lo, st.hi, p.x, p.y, p.z, p.m)
        else
            push!(stubs, st)
        end
    end
    isempty(stubs) && return TreePlan(t, stubs, [1])

    # Pack the pieces into `target` tasks, largest first onto the lightest task.
    # The pieces are wildly uneven, and longest-processing-time-first bounds the
    # imbalance by one piece's worth of overshoot.
    ngroup = min(target, length(stubs))
    order = sortperm(stubs; by = st -> -(st.hi - st.lo + 1))
    load = zeros(Int, ngroup)
    assign = [Int[] for _ in 1:ngroup]
    for idx in order
        g = argmin(load)
        push!(assign[g], idx)
        load[g] += stubs[idx].hi - stubs[idx].lo + 1
    end
    # Lay the stubs out so each task's are contiguous, which is what lets the
    # splice walk plan and results in the same order without a lookup table.
    ordered = Vector{TreeStub}(undef, length(stubs))
    first = Vector{Int}(undef, ngroup + 1)
    at = 1
    for g in 1:ngroup
        first[g] = at
        for idx in assign[g]
            ordered[at] = stubs[idx]
            at += 1
        end
    end
    first[ngroup + 1] = at
    return TreePlan(t, ordered, first)
end

"Number of non-empty octants of a sorted key range, without creating nodes."
function _octant_count(keys, lo::Int, hi::Int, shift::Int)
    nkids = 0
    q = lo
    for oct in 0:7
        a = q
        @inbounds while q <= hi && Int((keys[q] >> shift) & 0x7) == oct
            q += 1
        end
        q > a && (nkids += 1)
    end
    return nkids
end

"Close a cell the stop rule finishes, with its centre of mass, as the serial build does."
@inline function _close_leaf!(t::Tree, node::Int, lo::Int, hi::Int, x, y, z, m)
    t.lfirst[node] = Int32(lo)
    t.lcount[node] = Int32(hi - lo + 1)
    mass = 0.0; sx = 0.0; sy = 0.0; sz = 0.0
    @inbounds for i in lo:hi
        mass += m[i]; sx += m[i] * x[i]; sy += m[i] * y[i]; sz += m[i] * z[i]
    end
    _set_com!(t, node, mass, sx, sy, sz)
    return
end

"""
    task_tree_subtree(buf, meta, bs, n, dom, leaf_capacity, max_depth, plan, k) -> Tree

Build stub `k`'s subtree as a standalone `Tree` whose node 1 is the stub itself.
Returns an empty tree when `k` is past the end, so the driver can spawn a fixed
number of tasks without first fetching the plan -- a fetch would be a round trip
per partition per step, and a collective under MPI.
"""
function task_tree_subtree(buf, meta, bs::Int, n::Int, dom::Domain,
                           leaf_capacity::Int, max_depth::Int,
                           plan::TreePlan, k::Int)::Vector{Tree}
    k > ngroups(plan) && return Tree[]
    p = pfields(buf, bs, n)
    keys = mview(meta, bs, M_KEY, n)
    depth = min(max_depth, MAX_LEVEL)
    r = group_range(plan, k)
    out = Vector{Tree}(undef, length(r))
    for (j, i) in enumerate(r)
        st = plan.stubs[i]
        sub = Tree(dom)
        # Reserve before descending, for the same reason `build_local_tree` does:
        # every node appends to thirteen arrays and a group can hold dozens of
        # subtrees.
        reserve_nodes!(sub, max(8, cld(3 * (st.hi - st.lo + 1),
                                       2 * max(leaf_capacity, 1))))
        root = push_node!(sub, plan.upper.key[st.node], plan.upper.level[st.node])
        _build!(sub, root, keys, p.x, p.y, p.z, p.m, st.lo, st.hi,
                leaf_capacity, depth)
        out[j] = sub
    end
    return out
end

"""
    task_tree_splice(plan, subs...) -> Tree

Concatenate the subtrees into the upper tree and propagate mass upward.
"""
# ---------------------------------------------------------------------------
# Copy-free build: count, then write straight into the final node array
# ---------------------------------------------------------------------------
#
# Building each subtree into its own `Tree` and copying it in afterwards costs
# twice the allocation and a full pass over every node, which is more than
# splitting the build saves.  The subtrees are written directly into disjoint
# index ranges of one node array instead.  That needs each subtree's node
# count up front, which `_count_nodes` supplies with the same recursion the
# build uses, minus the writing.

"""
    _count_nodes(keys, lo, hi, level, leaf_capacity, max_depth) -> Int

How many nodes `_build_at!` will create for `lo:hi`, itself included.
"""
function _count_nodes(keys, lo::Int, hi::Int, level::Int,
                      leaf_capacity::Int, max_depth::Int)
    cnt = hi - lo + 1
    (cnt <= leaf_capacity || level >= max_depth) && return 1
    total = 1
    shift = 3 * (MAX_LEVEL - level - 1)
    q = lo
    for oct in 0:7
        a = q
        @inbounds while q <= hi && Int((keys[q] >> shift) & 0x7) == oct
            q += 1
        end
        b = q - 1
        b < a && continue
        total += _count_nodes(keys, a, b, level + 1, leaf_capacity, max_depth)
    end
    return total
end

"Initialise slot `i` of an already-sized tree, as `push_node!` would a new one."
@inline function _place_node!(t::Tree, i::Int, key::UInt64, level::Integer)
    @inbounds begin
        t.key[i] = key;            t.level[i] = Int8(level)
        t.mass[i] = 0.0
        t.comx[i] = 0.0;           t.comy[i] = 0.0;          t.comz[i] = 0.0
        t.size[i] = cell_size(t.dom, level)
        t.nchild[i] = Int8(0)
        t.lfirst[i] = Int32(0);    t.lcount[i] = Int32(0)
        t.ifirst[i] = Int32(0);    t.icount[i] = Int32(0)
        for c in 1:8
            t.childflat[8 * (i - 1) + c] = Int32(0)
        end
    end
    return i
end

"""
    _build_at!(t, node, cur, ...) -> Int

Build the subtree for `lo:hi` into `t` with its root already placed at `node`,
taking further slots from the cursor `cur`.  Mirrors `_build!` exactly; only
where the nodes come from differs.
"""
function _build_at!(t::Tree, node::Int, cur::Base.RefValue{Int}, keys, x, y, z, m,
                    lo::Int, hi::Int, leaf_capacity::Int, max_depth::Int)
    level = Int(t.level[node])
    cnt = hi - lo + 1
    if cnt <= leaf_capacity || level >= max_depth
        _close_leaf!(t, node, lo, hi, x, y, z, m)
        return node
    end
    shift = 3 * (MAX_LEVEL - level - 1)
    key = t.key[node]
    mass = 0.0; sx = 0.0; sy = 0.0; sz = 0.0
    nkids = 0
    q = lo
    for oct in 0:7
        a = q
        @inbounds while q <= hi && Int((keys[q] >> shift) & 0x7) == oct
            q += 1
        end
        b = q - 1
        b < a && continue
        kid = cur[]
        cur[] += 1
        _place_node!(t, kid, key_child(key, oct), level + 1)
        set_child!(t, node, oct, kid)
        nkids += 1
        _build_at!(t, kid, cur, keys, x, y, z, m, a, b, leaf_capacity, max_depth)
        mk = t.mass[kid]
        mass += mk
        sx += mk * t.comx[kid]; sy += mk * t.comy[kid]; sz += mk * t.comz[kid]
    end
    t.nchild[node] = Int8(nkids)
    _set_com!(t, node, mass, sx, sy, sz)
    return node
end

"""
    task_tree_count(buf, meta, bs, n, dom, leaf_capacity, max_depth, plan, k)

Node count of every stub in subtree task `k`'s group, in group order.  Cheap
next to the build -- one pass over the keys, no allocation and no writes -- and
it is what lets the build skip the copy.
"""
function task_tree_count(buf, meta, bs::Int, n::Int, dom::Domain,
                         leaf_capacity::Int, max_depth::Int,
                         plan::TreePlan, k::Int)::Vector{Int}
    k > ngroups(plan) && return Int[]
    keys = mview(meta, bs, M_KEY, n)
    depth = min(max_depth, MAX_LEVEL)
    r = group_range(plan, k)
    out = Vector{Int}(undef, length(r))
    for (j, i) in enumerate(r)
        st = plan.stubs[i]
        out[j] = _count_nodes(keys, st.lo, st.hi, Int(plan.upper.level[st.node]),
                              leaf_capacity, depth)
    end
    return out
end

"""
    task_tree_reserve(plan, counts...) -> Tree

The upper tree grown to its final size, with every stub's slice reserved.
"""
function task_tree_reserve(plan::TreePlan, off::Vector{Int}, counts::Vector{Int}...)::Tree
    t = plan.upper
    isempty(plan.stubs) && return t
    flat = Int[]
    for c in counts
        append!(flat, c)
    end
    length(flat) == length(plan.stubs) ||
        error("reserve got $(length(flat)) counts for $(length(plan.stubs)) stubs")
    # Derived from the offsets, never from `nnodes(t)`: this task grows the very
    # array whose length that would read.  It takes `off` as an argument so the
    # runtime orders it after `task_tree_offsets` -- both otherwise depend only
    # on the counts, so their order would be undefined.
    total = off[end] + flat[end] - 1      # each root lands in its stub's own slot
    _grow_tree!(t, total)
    return t
end

"""
    task_tree_offsets(plan, counts...) -> Vector{Int}

Where each stub's extra nodes go, derived from the same counts in the same
order as `task_tree_reserve`.
"""
function task_tree_offsets(plan::TreePlan, counts::Vector{Int}...)::Vector{Int}
    flat = Int[]
    for c in counts
        append!(flat, c)
    end
    off = Vector{Int}(undef, length(flat))
    at = nnodes(plan.upper)
    for k in eachindex(flat)
        off[k] = at
        at += flat[k] - 1
    end
    return off
end

"""
    task_tree_fill!(tree, buf, meta, bs, n, leaf_capacity, max_depth, plan, off, k)

Build subtree task `k`'s stubs straight into `tree`.
"""
function task_tree_fill!(tree::Tree, buf, meta, bs::Int, n::Int,
                         leaf_capacity::Int, max_depth::Int,
                         plan::TreePlan, off::Vector{Int}, k::Int)::Nothing
    k > ngroups(plan) && return nothing
    p = pfields(buf, bs, n)
    keys = mview(meta, bs, M_KEY, n)
    depth = min(max_depth, MAX_LEVEL)
    for i in group_range(plan, k)
        st = plan.stubs[i]
        # `off[i]` is the slot before this stub's first: the stub's own root stays
        # in the upper tree at `st.node` and its descendants start one later.
        cur = Ref(off[i] + 1)
        last = _build_at!(tree, st.node, cur, keys, p.x, p.y, p.z, p.m,
                          st.lo, st.hi, leaf_capacity, depth)
        # Cheap and worth keeping: a cursor that ran past its slice means the
        # count and the build disagree, and the symptom without this is a tree
        # that is silently wrong rather than an error.
        cur[] - 1 <= (i < length(plan.stubs) ? off[i + 1] : nnodes(tree)) ||
            error("subtree $i overran its slice: cursor $(cur[] - 1)")
        last
    end
    return nothing
end

"""
    task_tree_finish!(tree, plan, _fills...) -> Tree

Mass and centre of mass for the nodes above the split, once every subtree is in
place.  The `_fills` arguments carry no data; they are what makes this task wait
for the writes.
"""
function task_tree_finish!(tree::Tree, plan::TreePlan, _fills...)::Tree
    isempty(plan.stubs) && return tree
    _propagate_com!(tree, 1, Set(st.node for st in plan.stubs))
    return tree
end

function task_tree_splice(plan::TreePlan, groups::Vector{Tree}...)::Tree
    t = plan.upper
    isempty(plan.stubs) && return t
    # Flatten the per-task results back into stub order.  `task_tree_plan` lays
    # the stubs out so each task's are contiguous, so concatenating the groups
    # in task order is stub order.
    subs = Vector{Tree}(undef, length(plan.stubs))
    at = 1
    for g in groups, sub in g
        subs[at] = sub
        at += 1
    end
    at == length(plan.stubs) + 1 ||
        error("splice got $(at - 1) subtrees for $(length(plan.stubs)) stubs")

    base = nnodes(t)
    off = Vector{Int}(undef, length(plan.stubs))
    total = base
    for k in eachindex(plan.stubs)
        off[k] = total
        total += nnodes(subs[k]) - 1
    end
    _grow_tree!(t, total)
    for k in eachindex(plan.stubs)
        sub = subs[k]; st = plan.stubs[k]; o = off[k]
        _copy_node!(t, st.node, sub, 1, o)
        @inbounds for j in 2:nnodes(sub)
            _copy_node!(t, o + j - 1, sub, j, o)
        end
    end
    _propagate_com!(t, 1, Set(st.node for st in plan.stubs))
    return t
end

"Grow every per-node array of `t` to `n` nodes."
function _grow_tree!(t::Tree, n::Int)
    resize!(t.key, n); resize!(t.level, n); resize!(t.mass, n)
    resize!(t.comx, n); resize!(t.comy, n); resize!(t.comz, n)
    resize!(t.size, n); resize!(t.childflat, 8n); resize!(t.nchild, n)
    resize!(t.lfirst, n); resize!(t.lcount, n)
    resize!(t.ifirst, n); resize!(t.icount, n)
    return t
end

"Copy node `j` of `sub` into slot `i` of `t`, shifting child indices by `o`."
@inline function _copy_node!(t::Tree, i::Int, sub::Tree, j::Int, o::Int)
    @inbounds begin
        t.key[i] = sub.key[j];       t.level[i] = sub.level[j]
        t.mass[i] = sub.mass[j]
        t.comx[i] = sub.comx[j];     t.comy[i] = sub.comy[j];   t.comz[i] = sub.comz[j]
        t.size[i] = sub.size[j];     t.nchild[i] = sub.nchild[j]
        t.lfirst[i] = sub.lfirst[j]; t.lcount[i] = sub.lcount[j]
        t.ifirst[i] = sub.ifirst[j]; t.icount[i] = sub.icount[j]
        for c in 1:8
            kid = sub.childflat[8 * (j - 1) + c]
            t.childflat[8 * (i - 1) + c] = kid == 0 ? Int32(0) : Int32(o + kid - 1)
        end
    end
    return
end

"""
Mass and centre of mass for the upper nodes, bottom up.
"""
function _propagate_com!(t::Tree, node::Int, stubs::Set{Int})
    (node in stubs || t.nchild[node] == 0) && return
    mass = 0.0; sx = 0.0; sy = 0.0; sz = 0.0
    @inbounds for oct in 0:7
        kid = Int(child(t, node, oct))
        kid == 0 && continue
        _propagate_com!(t, kid, stubs)
        mk = t.mass[kid]
        mass += mk
        sx += mk * t.comx[kid]; sy += mk * t.comy[kid]; sz += mk * t.comz[kid]
    end
    _set_com!(t, node, mass, sx, sy, sz)
    return
end
