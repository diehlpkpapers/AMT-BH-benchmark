"""
Parallel sort of a partition into Morton order.
"""

"Bucket of `key` given ascending `splitters`; 1 .. length(splitters)+1."
@inline sort_bucket(splitters::Vector{UInt64}, key::UInt64) =
    searchsortedlast(splitters, key) + 1

"""
    task_sort_splitters(meta, bs, n, k) -> Vector{UInt64}

`k-1` keys that cut the partition into `k` roughly equal buckets.
"""
function task_sort_splitters(meta, bs::Int, n::Int, k::Int)::Vector{UInt64}
    (k <= 1 || n <= 1) && return UInt64[]
    key = mview(meta, bs, M_KEY, n)
    want = min(n, 32 * k)
    step = max(1, n ÷ want)
    sample = UInt64[key[i] for i in 1:step:n]
    sort!(sample)
    ns = length(sample)
    out = Vector{UInt64}(undef, k - 1)
    @inbounds for j in 1:(k - 1)
        out[j] = sample[clamp(round(Int, j * ns / k), 1, ns)]
    end
    return out
end

"Particles of tile `lo:hi` per bucket."
function task_sort_count(meta, bs::Int, n::Int, lo::Int, hi::Int,
                         splitters::Vector{UInt64})::Vector{Int}
    k = length(splitters) + 1
    c = zeros(Int, k)
    n == 0 && return c
    key = mview(meta, bs, M_KEY, n)
    @inbounds for i in lo:hi
        c[sort_bucket(splitters, key[i])] += 1
    end
    return c
end

"""
    task_sort_place(counts...) -> Vector{Int}

Where every (tile, bucket) run starts in the output, flattened tile-major.
"""
function task_sort_place(counts::Vector{Int}...)::Vector{Int}
    nt = length(counts)
    nt == 0 && return Int[]
    k = length(counts[1])
    dest = Vector{Int}(undef, nt * k)
    at = 1
    for b in 1:k
        for t in 1:nt
            dest[(t - 1) * k + b] = at
            at += counts[t][b]
        end
    end
    return dest
end

"Copy one particle from slot `i` of a partition to slot `j` of another."
@inline function _move_particle!(dp, dkey, did, sp, skey, sid, j::Int, i::Int)
    @inbounds begin
        dp.m[j] = sp.m[i]
        dp.x[j] = sp.x[i];   dp.y[j] = sp.y[i];   dp.z[j] = sp.z[i]
        dp.vx[j] = sp.vx[i]; dp.vy[j] = sp.vy[i]; dp.vz[j] = sp.vz[i]
        dkey[j] = skey[i];   did[j] = sid[i]
    end
    return
end

"""
    task_sort_scatter!(sbuf, smeta, buf, meta, bs, n, lo, hi, dest, splitters, t)

Move tile `t`'s particles into the scratch buffer at their sorted destinations.
"""
function task_sort_scatter!(sbuf, smeta, buf, meta, bs::Int, n::Int,
                            lo::Int, hi::Int, dest::Vector{Int},
                            splitters::Vector{UInt64}, t::Int)::Nothing
    n == 0 && return nothing
    k = length(splitters) + 1
    sp = pfields(buf, bs, n)
    skey = mview(meta, bs, M_KEY, n); sid = mview(meta, bs, M_ID, n)
    dp = pfields(sbuf, bs, n)
    dkey = mview(smeta, bs, M_KEY, n); did = mview(smeta, bs, M_ID, n)
    cur = dest[((t - 1) * k + 1):(t * k)]        # this tile's cursor per bucket
    @inbounds for i in lo:hi
        b = sort_bucket(splitters, skey[i])
        _move_particle!(dp, dkey, did, sp, skey, sid, cur[b], i)
        cur[b] += 1
    end
    return nothing
end

"""
    task_sort_bucket!(buf, meta, sbuf, smeta, bs, n, lo, hi)

Sort one contiguous group of buckets in the scratch buffer and copy it back.
"""
function task_sort_bucket!(buf, meta, sbuf, smeta, bs::Int, n::Int,
                           lo::Int, hi::Int)::Nothing
    (n == 0 || hi < lo) && return nothing
    sp = pfields(sbuf, bs, n)
    skey = mview(smeta, bs, M_KEY, n); sid = mview(smeta, bs, M_ID, n)
    rng = lo:hi
    sort_partition!(view(skey, rng), view(sid, rng),
                    view(sp.m, rng), view(sp.x, rng), view(sp.y, rng),
                    view(sp.z, rng), view(sp.vx, rng), view(sp.vy, rng),
                    view(sp.vz, rng))
    dp = pfields(buf, bs, n)
    dkey = mview(meta, bs, M_KEY, n); did = mview(meta, bs, M_ID, n)
    @inbounds for i in rng
        _move_particle!(dp, dkey, did, sp, skey, sid, i, i)
    end
    return nothing
end

"""
    bucket_ranges(totals, _scatters...) -> Vector{UnitRange{Int}}

Output range of every bucket.
"""
function bucket_ranges(totals::Vector{Int}, _scatters...)
    out = Vector{UnitRange{Int}}(undef, length(totals))
    at = 1
    for b in eachindex(totals)
        out[b] = at:(at + totals[b] - 1)
        at += totals[b]
    end
    return out
end

"Sort bucket `b` and copy it back into the partition buffer."
function task_sort_bucket_at!(buf, meta, sbuf, smeta, bs::Int, n::Int,
                              ranges::Vector{UnitRange{Int}}, b::Int)::Nothing
    b > length(ranges) && return nothing
    r = ranges[b]
    return task_sort_bucket!(buf, meta, sbuf, smeta, bs, n, first(r), last(r))
end

"""
    bucket_groups(counts, k) -> Vector{UnitRange{Int}}

Contiguous output ranges for the `k` sort tasks, one per bucket.
"""
function bucket_groups(total::Vector{Int})
    out = Vector{UnitRange{Int}}(undef, length(total))
    at = 1
    for b in eachindex(total)
        out[b] = at:(at + total[b] - 1)
        at += total[b]
    end
    return out
end

"Bucket sizes from the per-tile counts, in the same order `task_sort_place` uses."
function task_sort_totals(counts::Vector{Int}...)::Vector{Int}
    isempty(counts) && return Int[]
    k = length(counts[1])
    tot = zeros(Int, k)
    for c in counts, b in 1:k
        tot[b] += c[b]
    end
    return tot
end
