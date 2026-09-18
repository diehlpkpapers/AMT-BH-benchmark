"""
Morton (Z-order) keys and the cubic bounding domain.
"""

const MAX_LEVEL = 21
const ROOT_KEY = UInt64(1)

"""
    Domain(cx, cy, cz, half)

Cubic simulation domain centred at `(cx, cy, cz)` with half edge length `half`,
i.e. covering `[c - half, c + half]` in every dimension.  Barnes-Hut needs a
*cubic* domain so that a cell's edge length is direction independent.
"""
struct Domain
    cx::Float64
    cy::Float64
    cz::Float64
    half::Float64
end

edge_length(d::Domain) = 2 * d.half
origin(d::Domain) = (d.cx - d.half, d.cy - d.half, d.cz - d.half)

"""
    bounding_domain(minv, maxv; pad=1e-9)

Cubic domain containing the axis-aligned box `[minv, maxv]`.  The box is
inflated by `pad` (relative) so that particles on the upper boundary still map
into a valid cell index, and degenerate (zero size) inputs get a unit domain.
"""
function bounding_domain(minv::NTuple{3,Float64}, maxv::NTuple{3,Float64}; pad::Float64=1e-9)
    cx = 0.5 * (minv[1] + maxv[1])
    cy = 0.5 * (minv[2] + maxv[2])
    cz = 0.5 * (minv[3] + maxv[3])
    half = 0.0
    for k in 1:3
        half = max(half, 0.5 * (maxv[k] - minv[k]))
    end
    if !isfinite(half) || half <= 0
        half = 1.0
    end
    half *= (1 + pad)
    return Domain(cx, cy, cz, half)
end

"""
    cell_index(dom, x, y, z, level) -> (ix, iy, iz)

Integer cell coordinates of the point at `level`, clamped into the domain.
"""
@inline function cell_index(dom::Domain, x::Float64, y::Float64, z::Float64, level::Int)
    n = 1 << level
    ox, oy, oz = origin(dom)
    s = edge_length(dom) / n
    ix = clamp(floor(Int, (x - ox) / s), 0, n - 1)
    iy = clamp(floor(Int, (y - oy) / s), 0, n - 1)
    iz = clamp(floor(Int, (z - oz) / s), 0, n - 1)
    return ix, iy, iz
end

"Octant of `(ix, iy, iz)` bits at `shift`: `4*x + 2*y + z`."
@inline octant(ix::Int, iy::Int, iz::Int, shift::Int) =
    ((ix >> shift) & 1) << 2 | ((iy >> shift) & 1) << 1 | ((iz >> shift) & 1)

"""
    morton_key(dom, x, y, z, level=MAX_LEVEL) -> UInt64

Sentinel-prefixed Morton key of the cell of `level` that contains the point.
"""
@inline function morton_key(dom::Domain, x::Float64, y::Float64, z::Float64,
                            level::Int=MAX_LEVEL)
    ix, iy, iz = cell_index(dom, x, y, z, level)
    key = ROOT_KEY
    for l in (level - 1):-1:0
        key = (key << 3) | UInt64(octant(ix, iy, iz, l))
    end
    return key
end

"Level of a sentinel-prefixed key (the root is level 0)."
@inline key_level(key::UInt64) = (63 - leading_zeros(key)) ÷ 3

@inline key_parent(key::UInt64) = key >> 3
@inline key_child(key::UInt64, oct::Integer) = (key << 3) | UInt64(oct)
@inline key_ancestor(key::UInt64, levels::Int) = key >> (3 * levels)

"Is `anc` an ancestor of (or equal to) `key`?"
@inline function key_is_ancestor(anc::UInt64, key::UInt64)
    la, lk = key_level(anc), key_level(key)
    la <= lk || return false
    return key_ancestor(key, lk - la) == anc
end

"""
    key_range(key) -> (lo, hi)

Inclusive range of `MAX_LEVEL` keys contained in the cell `key`.  Used to map a
cell onto a contiguous slice of Morton-sorted particles.
"""
@inline function key_range(key::UInt64)
    d = MAX_LEVEL - key_level(key)
    lo = key << (3 * d)
    hi = ((key + 1) << (3 * d)) - 1
    return lo, hi
end

"""
    cell_geometry(dom, key) -> (cx, cy, cz, size)

Centre and edge length of the cell identified by `key`.
"""
function cell_geometry(dom::Domain, key::UInt64)
    level = key_level(key)
    ix = iy = iz = 0
    k = key
    for l in 0:(level - 1)
        oct = Int(k & 0x7)
        ix |= ((oct >> 2) & 1) << l
        iy |= ((oct >> 1) & 1) << l
        iz |= (oct & 1) << l
        k >>= 3
    end
    n = 1 << level
    s = edge_length(dom) / n
    ox, oy, oz = origin(dom)
    return (ox + (ix + 0.5) * s, oy + (iy + 0.5) * s, oz + (iz + 0.5) * s, s)
end

"Edge length of a cell at `level`."
@inline cell_size(dom::Domain, level::Integer) = edge_length(dom) / (1 << level)
