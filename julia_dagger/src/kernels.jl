"""
Compute kernels: Morton keying and sorting of a partition, the Barnes-Hut tree
traversal, the `O(N^2)` brute-force reference, energies, and the leapfrog
updates.
"""

"""
    PhysicsParams

Numerical parameters of the model.  `G` and `softening` carry the units of the
input data set; `soften_potential` selects whether the potential energy uses
the softened distance as well (equation 12 of the white paper does not state
it, so it is configurable and off by default).
"""
struct PhysicsParams
    G::Float64
    softening::Float64
    theta::Float64
    soften_potential::Bool
end

"Counters used to report the interaction mix (and to compare implementations)."
mutable struct InteractionCounts
    direct::Int64
    approx::Int64
end
InteractionCounts() = InteractionCounts(0, 0)

# ---------------------------------------------------------------------------
# Morton keying and sorting of one partition
# ---------------------------------------------------------------------------

"""
    compute_keys!(keys, dom, x, y, z)

Fill `keys` with the `MAX_LEVEL` Morton key of every particle.
"""
function compute_keys!(keys::AbstractVector{UInt64}, dom::Domain,
                       x::AbstractVector{Float64}, y::AbstractVector{Float64},
                       z::AbstractVector{Float64},
                       range::UnitRange{Int}=1:length(x))
    @inbounds for i in range
        keys[i] = morton_key(dom, x[i], y[i], z[i])
    end
    return keys
end

"""
    key_sortperm(keys) -> Vector{Int32}

The stable sorting permutation of `keys`, the same as
`sortperm(keys; alg = Base.Sort.DEFAULT_STABLE)`, by an LSD radix sort that
carries each key's index along. Digits that all keys share are skipped.
"""
function key_sortperm(keys::AbstractVector{UInt64})
    n = length(keys)
    # Small inputs gain nothing, and the Int32 payload bounds the size.
    (n < 1024 || n > typemax(Int32)) &&
        return Vector{Int32}(sortperm(keys; alg = Base.Sort.DEFAULT_STABLE))
    kmin = typemax(UInt64)
    kmax = typemin(UInt64)
    @inbounds for k in keys
        kmin = min(kmin, k); kmax = max(kmax, k)
    end
    bits = 64 - leading_zeros(kmax - kmin)
    ka = Vector{UInt64}(undef, n); ia = Vector{Int32}(undef, n)
    kb = Vector{UInt64}(undef, n); ib = Vector{Int32}(undef, n)
    @inbounds for i in 1:n
        ka[i] = keys[i] - kmin
        ia[i] = Int32(i)
    end
    R = 11
    mask = UInt64((1 << R) - 1)
    cnt = Vector{Int}(undef, 1 << R)
    shift = 0
    @inbounds while shift < bits
        fill!(cnt, 0)
        for i in 1:n
            cnt[Int((ka[i] >> shift) & mask) + 1] += 1
        end
        if maximum(cnt) == n          # every key has this digit
            shift += R
            continue
        end
        at = 0
        for d in eachindex(cnt)
            c = cnt[d]; cnt[d] = at; at += c
        end
        for i in 1:n
            d = Int((ka[i] >> shift) & mask) + 1
            p = cnt[d] + 1
            cnt[d] = p
            kb[p] = ka[i]; ib[p] = ia[i]
        end
        ka, kb = kb, ka
        ia, ib = ib, ia
        shift += R
    end
    return ia
end

"""
    sort_partition!(keys, id, arrays...) -> keys

Sort a partition's particles by `(key, id)` and apply the permutation in place
to `keys`, `id` and every array in `arrays`.  The id tie-break makes the order
deterministic for duplicate coordinates, which is what keeps runs
reproducible; Morton-sorted order is also what lets a cell map onto a
contiguous slice of the partition.
"""
function sort_partition!(keys::AbstractVector{UInt64}, id::AbstractVector{<:Integer},
                         arrays::Vararg{AbstractVector{Float64}})
    n = length(keys)
    n <= 1 && return keys
    # Sort the keys themselves rather than indices compared through a closure.
    perm = key_sortperm(keys)
    # Equal keys stay in input order, which would make the result depend on the
    # incoming arrangement.  Ordering each run of equal keys by `id` restores a
    # total order.  Two particles share a key only by landing in the same
    # deepest-level cell.
    i = 1
    @inbounds while i < n
        k = keys[perm[i]]
        j = i
        while j < n && keys[perm[j + 1]] == k
            j += 1
        end
        j > i && sort!(view(perm, i:j); by = p -> id[p])
        i = j + 1
    end
    issorted(perm) && return keys

    fbuf = Vector{Float64}(undef, n)
    for arr in arrays
        @inbounds for i in 1:n
            fbuf[i] = arr[perm[i]]
        end
        copyto!(arr, fbuf)
    end
    kbuf = Vector{UInt64}(undef, n)
    ibuf = Vector{eltype(id)}(undef, n)
    @inbounds for i in 1:n
        kbuf[i] = keys[perm[i]]
        ibuf[i] = id[perm[i]]
    end
    copyto!(keys, kbuf)
    copyto!(id, ibuf)
    return keys
end

# ---------------------------------------------------------------------------
# Barnes-Hut traversal
# ---------------------------------------------------------------------------

"""
    traverse_batch!(ax, ay, az, pot, tree, range, x, y, z, m, id, key, params, counts, stack)

Accumulate accelerations and per-particle potential contributions for the
target particles `range` of one partition by traversing its locally essential
`tree`.

A node is accepted when `size < theta * d` and it does not contain the target;
the containment test is only needed for theta > 1/sqrt(3). Nodes without
children and particles are summary nodes the LET builder already accepted for
the whole partition. `Val(false)` for the potential skips it, leaving `pot`
untouched. With `lanes = Val(K)`, `K > 1`, the walk runs `K` particles at a time
in SIMD lanes (`_traverse_packet!`), with bit-identical results.
"""
function traverse_batch!(ax::AbstractVector{Float64}, ay::AbstractVector{Float64},
                         az::AbstractVector{Float64}, pot::AbstractVector{Float64},
                         tree::Tree, range::UnitRange{Int},
                         x::AbstractVector{Float64}, y::AbstractVector{Float64},
                         z::AbstractVector{Float64}, m::AbstractVector{Float64},
                         id::AbstractVector{<:Integer},
                         key::AbstractVector{UInt64}, params::PhysicsParams,
                         counts::InteractionCounts=InteractionCounts(),
                         stack::Vector{Int32}=Int32[],
                         potv::Val{POT}=Val(true),
                         fast::Val=Val(false),
                         compact::Val=Val(false),
                         lanes::Val{K}=Val(0)) where {POT,K}
    # A target sits at most s*sqrt(3) from the centre of mass of a cell of edge
    # s that contains it, so for theta <= 1/sqrt(3) the containment test is
    # redundant. Both it and the import path are compiled away when not needed.
    guard = 3.0 * params.theta * params.theta > 1.0
    imports = nimported(tree) > 0
    if K > 1 && !guard && fast === Val(false)
        P = (ax, ay, az, pot, tree, range, x, y, z, m, id, params, counts, potv)
        return imports ? _traverse_packet!(P..., Val(true), lanes) :
                         _traverse_packet!(P..., Val(false), lanes)
    end
    A = (ax, ay, az, pot, tree, range, x, y, z, m, id, key, params, counts, stack,
         potv, fast, compact)
    return guard ?
        (imports ? _traverse!(A..., Val(true),  Val(true))
                 : _traverse!(A..., Val(true),  Val(false))) :
        (imports ? _traverse!(A..., Val(false), Val(true))
                 : _traverse!(A..., Val(false), Val(false)))
end

function _traverse!(ax::AbstractVector{Float64}, ay::AbstractVector{Float64},
                    az::AbstractVector{Float64}, pot::AbstractVector{Float64},
                    tree::Tree, range::UnitRange{Int},
                    x::AbstractVector{Float64}, y::AbstractVector{Float64},
                    z::AbstractVector{Float64}, m::AbstractVector{Float64},
                    id::AbstractVector{<:Integer},
                    key::AbstractVector{UInt64}, params::PhysicsParams,
                    counts::InteractionCounts, stack::Vector{Int32},
                    ::Val{POT}, fast::Val, compact::Val,
                    ::Val{GUARD}, ::Val{IMPORTS}) where {POT,GUARD,IMPORTS}
    nnodes(tree) == 0 && return counts
    theta2 = params.theta * params.theta
    eps2 = params.softening * params.softening
    G = params.G
    ndirect = 0
    napprox = 0

    # The depth is bounded by 8 * MAX_LEVEL, so the stack is a fixed-size
    # `MVector` on the stack frame. The `stack` argument is unused.
    stk = MVector{8 * MAX_LEVEL + 8, Int32}(undef)

    hot = tree.hot
    length(hot) == nnodes(tree) ||
        error("tree.hot holds $(length(hot)) nodes for a tree of $(nnodes(tree)); call pack_hot!")

    # Bound to locals: field access on the mutable `Tree` would reload them.
    tkey = tree.key; tchild = tree.childflat; tkids = tree.kids
    tlfirst = tree.lfirst; tifirst = tree.ifirst
    tipx = tree.ipx; tipy = tree.ipy; tipz = tree.ipz
    tipm = tree.ipm; tipid = tree.ipid

    # Squared cell edge per level, so the acceptance test needs only the level.
    el = edge_length(tree.dom)
    s2lev = Vector{Float64}(undef, MAX_LEVEL + 1)
    @inbounds for l in 0:MAX_LEVEL
        c = el / (1 << l)
        s2lev[l + 1] = c * c
    end

    @inbounds for i in range
        px = x[i]; py = y[i]; pz = z[i]; pid = id[i]
        pkey = GUARD ? key[i] : UInt64(0)
        accx = 0.0; accy = 0.0; accz = 0.0; phi = 0.0
        top = 1
        stk[1] = Int32(1)
        while top > 0
            nd = Int(stk[top]); top -= 1
            # Field by field rather than copying the whole record, so nothing
            # stays in a register before it is needed.
            nkid = Int(hot[nd].nchild)
            dx = hot[nd].comx - px; dy = hot[nd].comy - py; dz = hot[nd].comz - pz
            r2 = dx * dx + dy * dy + dz * dz

            if nkid == 0 && hot[nd].lcount == 0 && hot[nd].icount == 0
                # Summary node: accepted for the whole partition box.
                mass = hot[nd].mass
                if mass != 0.0
                    inv = inv_r3(r2, eps2, fast)
                    accx += mass * inv * dx; accy += mass * inv * dy; accz += mass * inv * dz
                    POT && (phi -= mass * _inv_dist(r2, eps2, params.soften_potential, fast))
                    napprox += 1
                end
                continue
            end

            lev = Int(hot[nd].lev)
            if s2lev[lev + 1] < theta2 * r2 &&
               !(GUARD && pkey >> (3 * (MAX_LEVEL - lev)) == tkey[nd])
                mass = hot[nd].mass
                inv = inv_r3(r2, eps2, fast)
                accx += mass * inv * dx; accy += mass * inv * dy; accz += mass * inv * dz
                POT && (phi -= mass * _inv_dist(r2, eps2, params.soften_potential, fast))
                napprox += 1
                continue
            end

            npart = Int(hot[nd].lcount) + Int(hot[nd].icount)
            if npart > 0
                lf = Int(tlfirst[nd]); lc = Int(hot[nd].lcount)
                for j in lf:(lf + lc - 1)
                    id[j] == pid && continue
                    ddx = x[j] - px; ddy = y[j] - py; ddz = z[j] - pz
                    rr2 = ddx * ddx + ddy * ddy + ddz * ddz
                    inv = inv_r3(rr2, eps2, fast)
                    mj = m[j]
                    accx += mj * inv * ddx; accy += mj * inv * ddy; accz += mj * inv * ddz
                    POT && (phi -= mj * _inv_dist(rr2, eps2, params.soften_potential, fast))
                    ndirect += 1
                end
                if IMPORTS && hot[nd].icount > 0
                    ifst = Int(tifirst[nd]); icnt = Int(hot[nd].icount)
                    for j in ifst:(ifst + icnt - 1)
                        tipid[j] == pid && continue
                        ddx = tipx[j] - px; ddy = tipy[j] - py; ddz = tipz[j] - pz
                        rr2 = ddx * ddx + ddy * ddy + ddz * ddz
                        inv = inv_r3(rr2, eps2, fast)
                        mj = tipm[j]
                        accx += mj * inv * ddx; accy += mj * inv * ddy; accz += mj * inv * ddz
                        POT && (phi -= mj * _inv_dist(rr2, eps2, params.soften_potential, fast))
                        ndirect += 1
                    end
                end
            end

            if nkid > 0
                if compact === Val(true)
                    base = Int(hot[nd].kidfirst)
                    for j in (base + nkid - 1):-1:base
                        top += 1
                        stk[top] = tkids[j]
                    end
                else
                    base = 8 * (nd - 1)
                    for oct in 7:-1:0
                        c = tchild[base + oct + 1]
                        if c != 0
                            top += 1
                            stk[top] = c
                        end
                    end
                end
            end
        end
        ax[i] = G * accx; ay[i] = G * accy; az[i] = G * accz
        POT && (pot[i] = m[i] * phi)
    end

    counts.direct += ndirect
    counts.approx += napprox
    return counts
end

"""
    _traverse_packet!(..., Val(K))

`_traverse!` for `K` consecutive target particles at once: one shared
depth-first walk with the particles in the lanes of `K`-wide vectors.

Each stack entry carries a mask of the lanes still exploring that node. At a
node the lanes that accept it interact with its centre of mass; the others go on
into its bodies or children. Every lane visits exactly the nodes the scalar walk
visits for its particle, in the same order and with the same arithmetic, so the
result is bit-identical to `_traverse!`. The lanes share the per-node work --
stack, node load, acceptance test -- since neighbours in Morton order open
almost the same nodes. Used without the containment guard and without fast math;
`traverse_batch!` sends anything else to `_traverse!`.
"""
function _traverse_packet!(ax::AbstractVector{Float64}, ay::AbstractVector{Float64},
                           az::AbstractVector{Float64}, pot::AbstractVector{Float64},
                           tree::Tree, range::UnitRange{Int},
                           x::AbstractVector{Float64}, y::AbstractVector{Float64},
                           z::AbstractVector{Float64}, m::AbstractVector{Float64},
                           id::AbstractVector{<:Integer}, params::PhysicsParams,
                           counts::InteractionCounts,
                           ::Val{POT}, ::Val{IMPORTS}, ::Val{K}) where {POT,IMPORTS,K}
    nnodes(tree) == 0 && return counts
    K <= 32 || error("packet width $K: masks are at most 32 bits")
    MT = K <= 8 ? UInt8 : K <= 16 ? UInt16 : UInt32
    V = SIMD.Vec{K,Float64}
    theta2 = params.theta * params.theta
    eps2 = params.softening * params.softening
    G = params.G
    soften = params.soften_potential
    ndirect = 0
    napprox = 0

    hot = tree.hot
    length(hot) == nnodes(tree) ||
        error("tree.hot holds $(length(hot)) nodes for a tree of $(nnodes(tree)); call pack_hot!")
    tkids = tree.kids; tlfirst = tree.lfirst; tifirst = tree.ifirst
    tipx = tree.ipx; tipy = tree.ipy; tipz = tree.ipz
    tipm = tree.ipm; tipid = tree.ipid

    el = edge_length(tree.dom)
    s2lev = Vector{Float64}(undef, MAX_LEVEL + 1)
    @inbounds for l in 0:MAX_LEVEL
        c = el / (1 << l)
        s2lev[l + 1] = c * c
    end

    stk = MVector{8 * MAX_LEVEL + 8, Int32}(undef)
    smask = MVector{8 * MAX_LEVEL + 8, MT}(undef)
    lanebit = SIMD.Vec{K,MT}(ntuple(l -> MT(1) << (l - 1), Val(K)))
    full = K == 8 * sizeof(MT) ? typemax(MT) : (MT(1) << K) - one(MT)
    none = zero(MT)

    hi = last(range)
    # A fresh `i` per packet: a reassigned variable captured by the lane loads
    # would be boxed and allocate on every packet.
    @inbounds for i in range[1:K:end]
        nlive = min(K, hi - i + 1)
        live = nlive == K ? full : (MT(1) << nlive) - one(MT)
        # Lanes past the end of the range repeat its last particle and stay
        # inactive.
        px = _lanes(x, i, hi, Val(K))
        py = _lanes(y, i, hi, Val(K))
        pz = _lanes(z, i, hi, Val(K))
        pid = _lanes_u64(id, i, hi, Val(K))
        accx = zero(V); accy = zero(V); accz = zero(V); phi = zero(V)
        top = 1
        stk[1] = Int32(1); smask[1] = live
        while top > 0
            nd = Int(stk[top]); am = smask[top]; top -= 1
            nkid = Int(hot[nd].nchild)
            dx = hot[nd].comx - px; dy = hot[nd].comy - py; dz = hot[nd].comz - pz
            r2 = dx * dx + dy * dy + dz * dz

            if nkid == 0 && hot[nd].lcount == 0 && hot[nd].icount == 0
                # Summary node: accepted by every lane that reaches it.
                mass = hot[nd].mass
                if mass != 0.0
                    act = (lanebit & am) != none
                    inv = _inv_r3v(r2, eps2)
                    accx = SIMD.vifelse(act, accx + mass * inv * dx, accx)
                    accy = SIMD.vifelse(act, accy + mass * inv * dy, accy)
                    accz = SIMD.vifelse(act, accz + mass * inv * dz, accz)
                    POT && (phi = SIMD.vifelse(act, phi - mass * _inv_distv(r2, eps2, soften), phi))
                    napprox += count_ones(am)
                end
                continue
            end

            lev = Int(hot[nd].lev)
            acc = SIMD.bitmask(s2lev[lev + 1] < theta2 * r2) % MT & am
            if acc != none
                act = (lanebit & acc) != none
                mass = hot[nd].mass
                inv = _inv_r3v(r2, eps2)
                accx = SIMD.vifelse(act, accx + mass * inv * dx, accx)
                accy = SIMD.vifelse(act, accy + mass * inv * dy, accy)
                accz = SIMD.vifelse(act, accz + mass * inv * dz, accz)
                POT && (phi = SIMD.vifelse(act, phi - mass * _inv_distv(r2, eps2, soften), phi))
                napprox += count_ones(acc)
            end
            om = am & ~acc                    # lanes that open this node
            om == none && continue
            opn = (lanebit & om) != none

            npart = Int(hot[nd].lcount) + Int(hot[nd].icount)
            if npart > 0
                lf = Int(tlfirst[nd]); lc = Int(hot[nd].lcount)
                for j in lf:(lf + lc - 1)
                    use = opn & (pid != UInt64(id[j]))
                    ddx = x[j] - px; ddy = y[j] - py; ddz = z[j] - pz
                    rr2 = ddx * ddx + ddy * ddy + ddz * ddz
                    inv = _inv_r3v(rr2, eps2)
                    mj = m[j]
                    accx = SIMD.vifelse(use, accx + mj * inv * ddx, accx)
                    accy = SIMD.vifelse(use, accy + mj * inv * ddy, accy)
                    accz = SIMD.vifelse(use, accz + mj * inv * ddz, accz)
                    POT && (phi = SIMD.vifelse(use, phi - mj * _inv_distv(rr2, eps2, soften), phi))
                    ndirect += count_ones(SIMD.bitmask(use))
                end
                if IMPORTS && hot[nd].icount > 0
                    ifst = Int(tifirst[nd]); icnt = Int(hot[nd].icount)
                    for j in ifst:(ifst + icnt - 1)
                        use = opn & (pid != UInt64(tipid[j]))
                        ddx = tipx[j] - px; ddy = tipy[j] - py; ddz = tipz[j] - pz
                        rr2 = ddx * ddx + ddy * ddy + ddz * ddz
                        inv = _inv_r3v(rr2, eps2)
                        mj = tipm[j]
                        accx = SIMD.vifelse(use, accx + mj * inv * ddx, accx)
                        accy = SIMD.vifelse(use, accy + mj * inv * ddy, accy)
                        accz = SIMD.vifelse(use, accz + mj * inv * ddz, accz)
                        POT && (phi = SIMD.vifelse(use, phi - mj * _inv_distv(rr2, eps2, soften), phi))
                        ndirect += count_ones(SIMD.bitmask(use))
                    end
                end
            end

            if nkid > 0
                # Children in the same order as the scalar walk pops them.
                base = Int(hot[nd].kidfirst)
                for j in (base + nkid - 1):-1:base
                    top += 1
                    stk[top] = tkids[j]
                    smask[top] = om
                end
            end
        end
        for l in 1:nlive
            ii = i + l - 1
            ax[ii] = G * accx[l]; ay[ii] = G * accy[l]; az[ii] = G * accz[l]
            POT && (pot[ii] = m[ii] * phi[l])
        end
    end

    counts.direct += ndirect
    counts.approx += napprox
    return counts
end

"Particles `i .. i+K-1` of `v` in lanes, the last one repeated past `hi`."
@inline _lanes(v::AbstractVector{Float64}, i::Int, hi::Int, ::Val{K}) where {K} =
    SIMD.Vec{K,Float64}(ntuple(l -> @inbounds(v[min(i + l - 1, hi)]), Val(K)))
@inline _lanes_u64(v::AbstractVector{<:Integer}, i::Int, hi::Int, ::Val{K}) where {K} =
    SIMD.Vec{K,UInt64}(ntuple(l -> UInt64(@inbounds(v[min(i + l - 1, hi)])), Val(K)))

"`inv_r3` lane by lane, with the same rounding."
@inline function _inv_r3v(r2::SIMD.Vec{K,Float64}, eps2::Float64) where {K}
    ir = 1 / sqrt(r2 + eps2)
    return ir * ir * ir
end

"`_inv_dist` lane by lane."
@inline function _inv_distv(r2::SIMD.Vec{K,Float64}, eps2::Float64, soften::Bool) where {K}
    soften && return 1 / sqrt(r2 + eps2)
    return SIMD.vifelse(r2 == 0.0, zero(r2), 1 / sqrt(r2))
end

"""
    inv_r3(r2, eps2)

The softened `1 / (r^2 + eps^2)^(3/2)` factor of equation 3, as one sqrt and
one division.  Shared with the brute-force reference, so that `theta = 0`
reproduces it exactly.
"""
@inline function inv_r3(r2::Float64, eps2::Float64)
    # The argument cannot be negative, so `sqrt`'s domain check is skipped.
    ir = 1 / Base.sqrt_llvm(r2 + eps2)
    return ir * ir * ir
end

"""
    inv_r3(r2, eps2, Val(fast))

`Val(true)` lets the compiler relax the reciprocal square root. Changes the
last bits, so it is off by default (`--fastmath`).
"""
@inline inv_r3(r2::Float64, eps2::Float64, ::Val{false}) = inv_r3(r2, eps2)
@inline function inv_r3(r2::Float64, eps2::Float64, ::Val{true})
    ir = @fastmath 1 / sqrt(r2 + eps2)
    return ir * ir * ir
end

"Reciprocal distance used for the potential energy (softened only on request)."
@inline function _inv_dist(r2::Float64, eps2::Float64, soften::Bool)
    if soften
        return 1 / Base.sqrt_llvm(r2 + eps2)
    else
        return r2 == 0.0 ? 0.0 : 1 / Base.sqrt_llvm(r2)
    end
end

@inline _inv_dist(r2::Float64, eps2::Float64, soften::Bool, ::Val{false}) =
    _inv_dist(r2, eps2, soften)
@inline function _inv_dist(r2::Float64, eps2::Float64, soften::Bool, ::Val{true})
    soften && return @fastmath 1 / sqrt(r2 + eps2)
    r2 == 0.0 && return 0.0
    return @fastmath 1 / sqrt(r2)
end

# ---------------------------------------------------------------------------
# brute force reference
# ---------------------------------------------------------------------------

"""
    brute_force!(ax, ay, az, pot, x, y, z, m, params) -> nothing

Serial `O(N^2)` reference used as the correctness oracle.  `pot[i]` holds
`m_i * sum_j -m_j / r_ij`, so `0.5 * G * sum(pot)` is the potential energy of
equation 12.
"""
function brute_force!(ax::AbstractVector{Float64}, ay::AbstractVector{Float64},
                      az::AbstractVector{Float64}, pot::AbstractVector{Float64},
                      x::AbstractVector{Float64}, y::AbstractVector{Float64},
                      z::AbstractVector{Float64}, m::AbstractVector{Float64},
                      params::PhysicsParams)
    n = length(x)
    eps2 = params.softening * params.softening
    fill!(ax, 0.0); fill!(ay, 0.0); fill!(az, 0.0); fill!(pot, 0.0)
    @inbounds for i in 1:n
        accx = 0.0; accy = 0.0; accz = 0.0; phi = 0.0
        for j in 1:n
            i == j && continue
            dx = x[j] - x[i]; dy = y[j] - y[i]; dz = z[j] - z[i]
            r2 = dx * dx + dy * dy + dz * dz
            inv = inv_r3(r2, eps2)
            accx += m[j] * inv * dx; accy += m[j] * inv * dy; accz += m[j] * inv * dz
            phi -= m[j] * _inv_dist(r2, eps2, params.soften_potential)
        end
        ax[i] = params.G * accx; ay[i] = params.G * accy; az[i] = params.G * accz
        pot[i] = m[i] * phi
    end
    return nothing
end

# ---------------------------------------------------------------------------
# energies and integration
# ---------------------------------------------------------------------------

"Kinetic energy `sum 0.5 m |v|^2` of a particle range."
function kinetic_energy(m::AbstractVector{Float64}, vx::AbstractVector{Float64},
                        vy::AbstractVector{Float64}, vz::AbstractVector{Float64},
                        range::UnitRange{Int}=1:length(m))
    e = 0.0
    @inbounds for i in range
        e += m[i] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i])
    end
    return 0.5 * e
end

"""
    kick_by!(vx, vy, vz, ax, ay, az, h[, range])

`v += a*h`, with `h` the actual time increment.  The leapfrog uses half-step
kicks either side of a drift, but the trailing half kick of one step and the
leading half kick of the next use the same accelerations and are fused into a
single full-step kick, so both increments have to be expressible.
"""
function kick_by!(vx::AbstractVector{Float64}, vy::AbstractVector{Float64},
                  vz::AbstractVector{Float64}, ax::AbstractVector{Float64},
                  ay::AbstractVector{Float64}, az::AbstractVector{Float64},
                  h::Float64, range::UnitRange{Int}=1:length(vx))
    @inbounds for i in range
        vx[i] += ax[i] * h; vy[i] += ay[i] * h; vz[i] += az[i] * h
    end
    return nothing
end

"Half-step kick: `v += a*dt/2`, taking the *full* step `dt`."
function kick!(vx::AbstractVector{Float64}, vy::AbstractVector{Float64},
               vz::AbstractVector{Float64}, ax::AbstractVector{Float64},
               ay::AbstractVector{Float64}, az::AbstractVector{Float64},
               dt::Float64, range::UnitRange{Int}=1:length(vx))
    return kick_by!(vx, vy, vz, ax, ay, az, 0.5 * dt, range)
end

"Drift: `p += v * dt`."
function drift!(x::AbstractVector{Float64}, y::AbstractVector{Float64},
                z::AbstractVector{Float64}, vx::AbstractVector{Float64},
                vy::AbstractVector{Float64}, vz::AbstractVector{Float64},
                dt::Float64, range::UnitRange{Int}=1:length(x))
    @inbounds for i in range
        x[i] += vx[i] * dt; y[i] += vy[i] * dt; z[i] += vz[i] * dt
    end
    return nothing
end
