#include "morton_keys.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>



// Spreads the lower 21 bits of an integer to a 63-bit value,
// inserting two zero bits between each original bit
// This is used to interleave the x, y, and z coordinates
uint64_t spread21(uint32_t v) {
    uint64_t x = v & 0x1FFFFF; // Mask to 21 bits
    x = (x | (x << 32)) & 0x1F00000000FFFFULL;
    x = (x | (x << 16)) & 0x1F0000FF0000FFULL;
    x = (x | (x <<  8)) & 0x100F00F00F00F00FULL;
    x = (x | (x <<  4)) & 0x10C30C30C30C30C3ULL;
    x = (x | (x <<  2)) & 0x1249249249249249ULL;
    return x;
}


uint64_t morton63(uint32_t xi, uint32_t yi, uint32_t zi) {
    // Interleave the spread bits of x, y, and z coordinates
    return spread21(xi) | (spread21(yi) << 1) | (spread21(zi) << 2);
}


uint64_t mortonPrefix(uint64_t code, int depth) {
    if (depth == 0) return 0;
    int shift = 63 - 3 * depth;
    return code & (~0ULL << shift);
}

// Pulls the 21 x, y, or z bits back out of a 63-bit Morton code
uint32_t compact21(uint64_t m) {
    m &= 0x1249249249249249ULL;
    m = (m ^ (m >>  2)) & 0x10c30c30c30c30c3ULL;
    m = (m ^ (m >>  4)) & 0x100f00f00f00f00fULL;
    m = (m ^ (m >>  8)) & 0x1f0000ff0000ffULL;
    m = (m ^ (m >> 16)) & 0x1f00000000ffffULL;
    m = (m ^ (m >> 32)) & 0x1fffff;
    return static_cast<uint32_t>(m);
}

// Deinterleaves the top 3*depth bits of the prefix into (ix,iy,iz) integer coordinates
void decodePrefix(uint64_t prefix, int depth, uint32_t &ix, uint32_t &iy, uint32_t &iz) {
    if (depth == 0) { ix = iy = iz = 0; return; }

    // Move the relevant 3*d bits down to the LSBs and un-shuffle.
    uint64_t bits = prefix >> (63 - 3 * depth);
    ix = compact21(bits >> 0);
    iy = compact21(bits >> 1);
    iz = compact21(bits >> 2);
}

// Decodes a morton key to the position of its minimum corner in normalized [0,1) space.
Position key_to_normalized_position(uint64_t code, int depth) {
    uint32_t ix, iy, iz;
    decodePrefix(code, depth, ix, iy, iz);
    const double scale = 1.0 / static_cast<double>(1ULL << depth); // 1 / 2^d
    return { ix * scale, iy * scale, iz * scale };
}


std::vector<uint64_t> generateMortonCodes(const std::vector<Position>& raw, const BoundingBox& box)
{
    const std::size_t n = raw.size();
    std::vector<uint64_t> out_codes(n);
    if (!n) return out_codes;

    const double dx = (box.max.x == box.min.x) ? 1.0 : (box.max.x - box.min.x);
    const double dy = (box.max.y == box.min.y) ? 1.0 : (box.max.y - box.min.y);
    const double dz = (box.max.z == box.min.z) ? 1.0 : (box.max.z - box.min.z);

    constexpr uint32_t Q = (1u << 21);

    auto norm = [](double v, double lo, double span)
    {
        double t = (v - lo) / span;
        if (t <= 0.0) return 0.0;
        if (t >= 1.0) return std::nextafter(1.0, 0.0);
        return t;
    };

    auto body = [&](std::size_t i)
    {
        double xn = norm(raw[i].x, box.min.x, dx);
        double yn = norm(raw[i].y, box.min.y, dy);
        double zn = norm(raw[i].z, box.min.z, dz);

        uint32_t xi = static_cast<uint32_t>(xn * Q);
        uint32_t yi = static_cast<uint32_t>(yn * Q);
        uint32_t zi = static_cast<uint32_t>(zn * Q);
        out_codes[i] = morton63(xi, yi, zi);
    };

    // Per-element work here is a handful of flops, so HPX's default
    // automatic chunking - many small chunks - lets dispatch overhead
    // dominate at high thread counts (measured: this loop got SLOWER from
    // 32 to 128 threads). An explicit, coarser chunk size fixes that;
    // 4096 was the empirically best of {2048, 4096, 8192, 16384} tested on
    // kamand1 (2x AMD EPYC 7H12) at n ~ 3*10^5 - but only above the chunk
    // size itself: static_chunk_size(4096) segfaults inside HPX's
    // chunk_size_iterator when the range is smaller than one chunk, so
    // smaller inputs fall back to default chunking, which is already cheap
    // enough there (see compute_local_bbox for the same fix).
    constexpr std::size_t kMortonStaticChunkSize = 4096;
    if (n >= kMortonStaticChunkSize) {
        hpx::experimental::for_loop(
            hpx::execution::par.with(hpx::execution::experimental::static_chunk_size(kMortonStaticChunkSize)),
            std::size_t(0), n, body);
    } else {
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), n, body);
    }
    return out_codes;
}

