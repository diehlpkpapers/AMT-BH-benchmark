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
    sort_partition!(keys, id, arrays...) -> keys

Sort a partition's particles by `(key, id)` and apply the permutation in place
to `keys`, `id` and every array in `arrays`.  The id tie-break makes the order
deterministic for duplicate coordinates, which is what keeps runs
reproducible; Morton-sorted order is also what lets a cell map onto a
"""
function sort_partition!(keys::AbstractVector{UInt64}, id::AbstractVector{<:Integer},
                         arrays::Vararg{AbstractVector{Float64}})
    n = length(keys)
    n <= 1 && return keys
    # Sort the keys themselves rather than indices compared through a closure.
    # An indirect comparator defeats the cache and hides from the compiler that
    # this is an integer sort, so the radix path for `UInt64` never runs.
    perm = sortperm(keys; alg = Base.Sort.DEFAULT_STABLE)
    # `sortperm` leaves equal keys in input order, which would make the result
    # depend on the incoming arrangement.  Ordering each run of equal keys by
    # `id` restores a total order.  Two particles share a key only by landing
    # in the same deepest-level cell.
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
"""
function traverse_batch!(ax::AbstractVector{Float64}, ay::AbstractVector{Float64},
                         az::AbstractVector{Float64}, pot::AbstractVector{Float64},
                         tree::Tree, range::UnitRange{Int},
                         x::AbstractVector{Float64}, y::AbstractVector{Float64},
                         z::AbstractVector{Float64}, m::AbstractVector{Float64},
                         id::AbstractVector{<:Integer},
                         key::AbstractVector{UInt64}, params::PhysicsParams,
                         counts::InteractionCounts=InteractionCounts(),
                         stack::Vector{Int32}=Int32[])
    nnodes(tree) == 0 && return counts
    theta2 = params.theta * params.theta
    # A target sits at most s*sqrt(3) from the centre of mass of a cell of edge
    # s that contains it, so for theta <= 1/sqrt(3) the criterion can never
    # accept such a cell and the containment test below is provably redundant.
    guard = 3.0 * theta2 > 1.0
    eps2 = params.softening * params.softening
    G = params.G
    ndirect = 0
    napprox = 0

    @inbounds for i in range
        px = x[i]; py = y[i]; pz = z[i]; pid = id[i]; pkey = key[i]
        accx = 0.0; accy = 0.0; accz = 0.0; phi = 0.0
        empty!(stack)
        push!(stack, Int32(1))
        while !isempty(stack)
            nd = Int(pop!(stack))
            nkid = Int(tree.nchild[nd])
            npart = nparticles(tree, nd)

            if nkid == 0 && npart == 0
                # Summary node: accepted for the whole partition box.
                dx = tree.comx[nd] - px; dy = tree.comy[nd] - py; dz = tree.comz[nd] - pz
                r2 = dx * dx + dy * dy + dz * dz
                mass = tree.mass[nd]
                if mass != 0.0
                    inv = inv_r3(r2, eps2)
                    accx += mass * inv * dx; accy += mass * inv * dy; accz += mass * inv * dz
                    phi -= mass * _inv_dist(r2, eps2, params.soften_potential)
                    napprox += 1
                end
                continue
            end

            dx = tree.comx[nd] - px; dy = tree.comy[nd] - py; dz = tree.comz[nd] - pz
            r2 = dx * dx + dy * dy + dz * dz
            s = tree.size[nd]
            if theta2 > 0 && s * s < theta2 * r2 &&
               !(guard && pkey >> (3 * (MAX_LEVEL - Int(tree.level[nd]))) == tree.key[nd])
                mass = tree.mass[nd]
                inv = inv_r3(r2, eps2)
                accx += mass * inv * dx; accy += mass * inv * dy; accz += mass * inv * dz
                phi -= mass * _inv_dist(r2, eps2, params.soften_potential)
                napprox += 1
                continue
            end

            if npart > 0
                lf = Int(tree.lfirst[nd]); lc = Int(tree.lcount[nd])
                for j in lf:(lf + lc - 1)
                    id[j] == pid && continue
                    ddx = x[j] - px; ddy = y[j] - py; ddz = z[j] - pz
                    rr2 = ddx * ddx + ddy * ddy + ddz * ddz
                    inv = inv_r3(rr2, eps2)
                    mj = m[j]
                    accx += mj * inv * ddx; accy += mj * inv * ddy; accz += mj * inv * ddz
                    phi -= mj * _inv_dist(rr2, eps2, params.soften_potential)
                    ndirect += 1
                end
                if tree.icount[nd] > 0
                    ifst = Int(tree.ifirst[nd]); icnt = Int(tree.icount[nd])
                    for j in ifst:(ifst + icnt - 1)
                        tree.ipid[j] == pid && continue
                        ddx = tree.ipx[j] - px; ddy = tree.ipy[j] - py; ddz = tree.ipz[j] - pz
                        rr2 = ddx * ddx + ddy * ddy + ddz * ddz
                        inv = inv_r3(rr2, eps2)
                        mj = tree.ipm[j]
                        accx += mj * inv * ddx; accy += mj * inv * ddy; accz += mj * inv * ddz
                        phi -= mj * _inv_dist(rr2, eps2, params.soften_potential)
                        ndirect += 1
                    end
                end
            end

            if nkid > 0
                for oct in 7:-1:0
                    c = child(tree, nd, oct)
                    c != 0 && push!(stack, c)
                end
            end
        end
        ax[i] = G * accx; ay[i] = G * accy; az[i] = G * accz
        pot[i] = m[i] * phi
    end

    counts.direct += ndirect
    counts.approx += napprox
    return counts
end

"""
    inv_r3(r2, eps2)

The softened `1 / (r^2 + eps^2)^(3/2)` factor of equation 3, as one sqrt and
one division.
"""
@inline function inv_r3(r2::Float64, eps2::Float64)
    ir = 1 / sqrt(r2 + eps2)
    return ir * ir * ir
end

"Reciprocal distance used for the potential energy (softened only on request)."
@inline function _inv_dist(r2::Float64, eps2::Float64, soften::Bool)
    if soften
        return 1 / sqrt(r2 + eps2)
    else
        return r2 == 0.0 ? 0.0 : 1 / sqrt(r2)
    end
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
