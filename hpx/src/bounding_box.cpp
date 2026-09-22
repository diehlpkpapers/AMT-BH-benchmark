#include "body.h"
#include "linear_octree.h"
#include "morton_keys.h"
#include "hpx_collectives.h"
#include <vector>
#include <limits>
#include <algorithm>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/runtime_local/get_os_thread_count.hpp>


double min_distance_sq(const BoundingBox& b1, const BoundingBox& b2) {
    // For each axis, find the separation distance. If the boxes overlap on an
    // axis, the distance is 0
    double dx = std::max(0.0, std::max(b1.min.x - b2.max.x, b2.min.x - b1.max.x));
    double dy = std::max(0.0, std::max(b1.min.y - b2.max.y, b2.min.y - b1.max.y));
    double dz = std::max(0.0, std::max(b1.min.z - b2.max.z, b2.min.z - b1.max.z));

    return dx * dx + dy * dy + dz * dz;
}

/**
 * @brief Computes the tight bounding box for a vector of local positions.
 */
BoundingBox compute_local_bbox(const std::vector<Position>& local_pos) {
    if (local_pos.empty()) {
        return BoundingBox{};
    }

    // hpx::transform_reduce: map each Position to its own degenerate
    // (point) BoundingBox, then combine pairwise via min/max - HPX's own
    // partitioning/reduction-tree implementation, rather than a hand-rolled
    // chunk-and-fold. `init` is a valid combine identity (a real box at
    // local_pos[0]), not a placeholder - combining it with local_pos[0]'s
    // own transformed box again is harmless, since min/max(x, x) == x.
    auto combine = [](const BoundingBox& a, const BoundingBox& b) -> BoundingBox {
        BoundingBox out;
        out.min.x = std::min(a.min.x, b.min.x);
        out.min.y = std::min(a.min.y, b.min.y);
        out.min.z = std::min(a.min.z, b.min.z);
        out.max.x = std::max(a.max.x, b.max.x);
        out.max.y = std::max(a.max.y, b.max.y);
        out.max.z = std::max(a.max.z, b.max.z);
        return out;
    };
    auto transform = [](const Position& p) -> BoundingBox { return BoundingBox{p, p}; };
    BoundingBox init{local_pos[0], local_pos[0]};

    // Explicit chunk size for the same reason as generateMortonCodes: tiny
    // per-element work lets default auto-chunking's dispatch overhead
    // dominate at high thread counts. Same empirically-tuned value - but
    // only above the chunk size itself: static_chunk_size(4096) segfaults
    // inside HPX's chunk_size_iterator when the range is smaller than one
    // chunk (reproduced down to a single element), so smaller inputs fall
    // back to default chunking, which is already cheap enough there.
    constexpr std::size_t kBboxStaticChunkSize = 4096;
    if (local_pos.size() >= kBboxStaticChunkSize) {
        return hpx::transform_reduce(
            hpx::execution::par.with(hpx::execution::experimental::static_chunk_size(kBboxStaticChunkSize)),
            local_pos.begin(), local_pos.end(), init, combine, transform);
    }
    return hpx::transform_reduce(hpx::execution::par,
        local_pos.begin(), local_pos.end(), init, combine, transform);
}

BoundingBox compute_global_bbox(const BoundingBox& local_bb, hpxc::Ctx& ctx) {
    using hpxc::this_site_arg, hpxc::generation_arg;

    std::array<double, 3> local_min{local_bb.min.x, local_bb.min.y, local_bb.min.z};
    std::array<double, 3> local_max{local_bb.max.x, local_bb.max.y, local_bb.max.z};

    auto min_fut = hpx::collectives::all_reduce(ctx.comm_bbox_min, local_min,
        hpxc::MinArr3{}, this_site_arg(ctx.rank), generation_arg(ctx.gen_bbox_min++));
    auto max_fut = hpx::collectives::all_reduce(ctx.comm_bbox_max, local_max,
        hpxc::MaxArr3{}, this_site_arg(ctx.rank), generation_arg(ctx.gen_bbox_max++));

    std::array<double, 3> global_min = min_fut.get();
    std::array<double, 3> global_max = max_fut.get();

    BoundingBox global_bb;
    global_bb.min = {global_min[0], global_min[1], global_min[2]};
    global_bb.max = {global_max[0], global_max[1], global_max[2]};
    return global_bb;
}


BoundingBox getBoundingBoxForCell(const OctreeKey& k,
                                const BoundingBox& global_bb) {
    // Must match generateMortonCodes' normalization exactly (same L for all
    // three axes, not each axis by its own extent) - otherwise a key decodes
    // to a different box than the one it was actually built from.
    double dx_raw = global_bb.max.x - global_bb.min.x;
    double dy_raw = global_bb.max.y - global_bb.min.y;
    double dz_raw = global_bb.max.z - global_bb.min.z;
    double L = std::max({dx_raw, dy_raw, dz_raw});
    double dx = L, dy = L, dz = L;

    // Calculate the real world size of a cell at this depth
    double sizeX = dx / (1 << k.depth);
    double sizeY = dy / (1 << k.depth);
    double sizeZ = dz / (1 << k.depth);

    Position norm_min = key_to_normalized_position(k.prefix, k.depth);

    // Scale the normalized coordinates to the real-world minimum corner position
    double min_x = global_bb.min.x + norm_min.x * dx;
    double min_y = global_bb.min.y + norm_min.y * dy;
    double min_z = global_bb.min.z + norm_min.z * dz;

    return {{min_x, min_y, min_z}, {min_x + sizeX, min_y + sizeY, min_z + sizeZ}};
}

double pointBoxDistanceSq(double x,double y,double z, const BoundingBox& b) {
    /* distance along each axis (0 if inside) */
    double dx = std::max(0.0, std::max(b.min.x - x, x - b.max.x));
    double dy = std::max(0.0, std::max(b.min.y - y, y - b.max.y));
    double dz = std::max(0.0, std::max(b.min.z - z, z - b.max.z));
    return dx*dx + dy*dy + dz*dz;
}


