#include "traversal.h"

#include <array>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <algorithm>
#include <iostream>
#include <iomanip>


void computeAccelerations(
    const OctreeMap&                tree,
    const std::vector<uint64_t>&    key,
    const std::vector<Position>&    pos,
    const std::vector<double>&      mass,
    double                          theta,
    double                          G,
    double                          soft2,
    const BoundingBox&              global_bb,
    std::vector<Acceleration>&      out,
    int                             rank)
{
    const size_t N = pos.size();
    out.resize(N);
    const double theta2 = theta * theta;

    constexpr int MAX_L = 21;

    // root side length
    double dx = global_bb.max.x - global_bb.min.x;
    double dy = global_bb.max.y - global_bb.min.y;
    double dz = global_bb.max.z - global_bb.min.z;
    double L  = std::max({dx, dy, dz});   // physical side length of root cell

    // precompute side^2 per depth (the per-depth Morton stride the old 8-way
    // probe needed to build child keys is gone with the flat child indices)
    std::array<double, MAX_L+1> side2;
    for (int d = 0; d <= MAX_L; ++d) {
        double side = L / double(1ULL << d);
        side2[d] = side * side;
    }

    constexpr int STACK_MAX = 256; // safe for depth <=21

    // Fix 2: one flatten of the hashmap per force call (same cadence as the
    // tree build itself), after which the N per-body walks below are pure
    // array indexing - no hash, no shard select, no probe. Children of a node
    // sit at consecutive indices, in the same ascending-octant order the old
    // 8-way probe pushed them in, so the walk order (and therefore every
    // floating-point sum) is unchanged.
    const FlatTree flat = flattenTree(tree);
    const FlatNode* const nodes = flat.data();

    hpx::experimental::for_loop(hpx::execution::par, size_t(0), N, [&](size_t i) {
        double ax = 0.0, ay = 0.0, az = 0.0;

        std::array<uint32_t, STACK_MAX> stack;
        int top = 0;
        if (!flat.empty()) stack[top++] = 0;  // index 0 is the root; empty => empty walk

        while (top) {
            const FlatNode& node = nodes[stack[--top]];
            const uint64_t pref = node.prefix;
            const uint8_t  dep  = node.depth;
            if (node.mass == 0.0) continue;

            double dx = node.comX - pos[i].x;
            double dy = node.comY - pos[i].y;
            double dz = node.comZ - pos[i].z;
            double r2 = dx*dx + dy*dy + dz*dz + soft2;

            // A leaf is never opened, even if it fails the BH test below or
            // holds body i itself: instead resolve it by exact pairwise
            // summation over its own bodies. This takes priority over both
            // the BH test and the self-skip below. At the default
            // leaf_capacity == 1 a leaf holds exactly one body (unless bodies
            // share a Morton code), so this reduces to "skip myself, or add
            // that one body as a monopole" - the same value the old
            // subdivide-to-depth-21 walk reached one chain of identical
            // single-body cells further down.
            if (node.is_leaf) {
                const bool self_leaf = node.body_count > 0 &&
                    i >= node.body_begin &&
                    i <  static_cast<size_t>(node.body_begin) + node.body_count;

                if (self_leaf || side2[dep] >= theta2 * r2) {
                    double own_m = 0.0;
                    for (uint32_t jj = 0; jj < node.body_count; ++jj) {
                        size_t j = static_cast<size_t>(node.body_begin) + jj;
                        own_m += mass[j];
                        if (j == i) continue;
                        double ddx = pos[j].x - pos[i].x;
                        double ddy = pos[j].y - pos[i].y;
                        double ddz = pos[j].z - pos[i].z;
                        double r2j = ddx*ddx + ddy*ddy + ddz*ddz + soft2;
                        if (!(r2j > 0.0)) continue;
                        double invR  = 1.0 / std::sqrt(r2j);
                        double invR3 = invR * invR * invR;
                        double s     = G * mass[j] * invR3;
                        ax += s * ddx;  ay += s * ddy;  az += s * ddz;
                    }

                    // Mass merged onto this key from another locality (e.g.
                    // --fc tree, or a LET record overlapping this leaf) isn't
                    // covered by the loop above; recover its centre of mass
                    // exactly from the merge algebra and add it as one monopole.
                    double extra = node.mass - own_m;
                    if (extra > 1e-12 * node.mass) {
                        double own_wx = 0.0, own_wy = 0.0, own_wz = 0.0;
                        for (uint32_t jj = 0; jj < node.body_count; ++jj) {
                            size_t j = static_cast<size_t>(node.body_begin) + jj;
                            own_wx += mass[j] * pos[j].x;
                            own_wy += mass[j] * pos[j].y;
                            own_wz += mass[j] * pos[j].z;
                        }
                        double ex = (node.mass * node.comX - own_wx) / extra;
                        double ey = (node.mass * node.comY - own_wy) / extra;
                        double ez = (node.mass * node.comZ - own_wz) / extra;
                        double edx = ex - pos[i].x, edy = ey - pos[i].y, edz = ez - pos[i].z;
                        double er2 = edx*edx + edy*edy + edz*edz + soft2;
                        if (er2 > 0.0) {
                            double invR  = 1.0 / std::sqrt(er2);
                            double invR3 = invR * invR * invR;
                            double s     = G * extra * invR3;
                            ax += s * edx;  ay += s * edy;  az += s * edz;
                        }
                    }
                    continue;
                }
                // is_leaf but passes BH and doesn't contain body i: falls
                // through to the monopole branch below (node.is_leaf is one
                // of its acceptance conditions).
            }

            // Self-skip for a depth-21 cell that is not a local leaf - only
            // reachable for nodes that arrived over the wire, since every
            // cell this locality built at depth 21 is is_leaf and was already
            // resolved (with its own exact self-exclusion) above.
            if (dep == MAX_L && pref == key[i]) continue;

            // Barnes–Hut acceptance
            if (node.is_pseudo || node.is_leaf || dep == MAX_L || side2[dep] < theta2 * r2) {
                double invR  = 1.0 / std::sqrt(r2);
                double invR3 = invR * invR * invR;
                double s     = G * node.mass * invR3;
                ax += s * dx;  ay += s * dy;  az += s * dz;
            } else {
                // Open cell: its children are already contiguous, so this is
                // pointer arithmetic instead of up to 8 hashmap lookups (of
                // which ~27% used to be guaranteed misses).
                if (top + node.child_count > STACK_MAX)
                    throw std::runtime_error("BH stack overflow");
                for (uint32_t c = 0; c < node.child_count; ++c)
                    stack[top++] = node.children_offset + c;
            }
        }

        out[i] = {ax, ay, az};
    });
}



void computeAccelerationsWithLET(
    OctreeMap&                     local_tree,
    const std::vector<NodeRecord>& remote_nodes,
    const std::vector<uint64_t>&   bodyKey,
    const std::vector<Position>&   pos,
    const std::vector<double>&     mass,
    double                         theta,
    double                         G,
    double                         soft2,
    const BoundingBox&             global_bb,
    std::vector<Acceleration>&     out,
    int                            rank)
{
    // Skip building the remote tree when there's nothing to merge - this
    // used to shadow an identical, unused build every step regardless.
    if (!remote_nodes.empty()) {
        OctreeMap remote_tree = buildTreeFromPseudoLeaves(remote_nodes);
        mergeRecordsIntoTree(local_tree, serializeTreeToRecords(remote_tree));
    }

    computeAccelerations(local_tree, bodyKey, pos, mass, theta, G, soft2, global_bb, out, rank);

}

/**
 * @brief Computes accelerations using a local BH traversal and a remote direct force summation.
 *
 * This function provides a hybrid approach for force calculation. It performs:
 * 1. A standard Barnes-Hut traversal on the local_tree for local interactions.
 * 2. A direct O(N*M) summation where the force from each of the M remote_nodes
 *    is calculated for each of the N local bodies.
 *
 * This method avoids the overhead of building a secondary tree for remote nodes, which
 * can be faster if the number of remote pseudo-leaves is small
 *
 * @param local_tree The pre-built octree for the local rank's particles
 * @param remote_nodes The vector of pseudo-leaves received from other ranks
 * @param key Morton keys of the local bodies
 * @param pos Positions of the local bodies
 * @param theta The Barnes-Hut opening angle for the local traversal
 * @param G The gravitational constant
 * @param soft2 The softening factor squared
 * @param[out] out The output vector where the final calculated accelerations are stored
 */
void computeAccelerationsWithRemoteDirectSum(
    const OctreeMap&               local_tree,
    const std::vector<NodeRecord>& remote_nodes,
    const std::vector<uint64_t>&   key,
    const std::vector<Position>&   pos,
    const std::vector<double>&     mass,
    double                         theta,
    double                         G,
    double                         soft2,
    const BoundingBox&             global_bb,
    std::vector<Acceleration>&     out,
    int                            rank)
{
    // const auto t0 = std::chrono::high_resolution_clock::now();
    // const auto p1_start = std::chrono::high_resolution_clock::now();

    // Pass 1 local Barnes–Hut walk
    computeAccelerations(local_tree, key, pos, mass, theta, G, soft2, global_bb, out,
                    rank);

    // const double p1_us = std::chrono::duration_cast<std::chrono::microseconds>(
    //                          std::chrono::high_resolution_clock::now() -
    //                          p1_start)
    //                          .count();
    // const auto p2_start = std::chrono::high_resolution_clock::now();

    // Pass 2 direct forces from remote pseudoleaves
    // using omp simd
    // build a structure of arrays copy so the compiler sees unit stride
    const std::size_t M = remote_nodes.size();
    std::vector<double> rx(M), ry(M), rz(M), rm(M);
    for (std::size_t j = 0; j < M; ++j) {
        rx[j] = remote_nodes[j].comX;
        ry[j] = remote_nodes[j].comY;
        rz[j] = remote_nodes[j].comZ;
        rm[j] = remote_nodes[j].mass * G;
    }

    // loop over local bodies
    hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), pos.size(), [&](std::size_t i) {

        double ax = 0.0, ay = 0.0, az = 0.0;
        const double px = pos[i].x;
        const double py = pos[i].y;
        const double pz = pos[i].z;

        // inner loop over remote nodes (left for the compiler to auto-vectorize;
        // HPX has no direct #pragma omp simd equivalent for a hand-rolled reduction)
        for (std::size_t j = 0; j < M; ++j) {
            double dx = rx[j] - px;
            double dy = ry[j] - py;
            double dz = rz[j] - pz;
            double r2 = dx*dx + dy*dy + dz*dz + soft2;
            if (r2 <= 1e-12) continue;

            double invR  = 1.0 / std::sqrt(r2);
            double invR3 = invR * invR * invR;
            double s     = rm[j] * invR3;
            ax += s * dx;
            ay += s * dy;
            az += s * dz;
        }

        out[i].x += ax;
        out[i].y += ay;
        out[i].z += az;
    });

    // const double p2_us = std::chrono::duration_cast<std::chrono::microseconds>(
    //                          std::chrono::high_resolution_clock::now() -
    //                          p2_start)
    //                          .count();

    // const double total_us =
    //     std::chrono::duration_cast<std::chrono::microseconds>(
    //         std::chrono::high_resolution_clock::now() - t0)
    //         .count();

    // if (rank == 0) {
    //     std::cout << std::fixed << std::setprecision(1)
    //               << "    - BH remote sum (Pass 2): " << p2_us << " µs ("
    //               << pct(p2_us, total_us) << "%)\n"
    //               << "    - BH local  walk (Pass 1): " << p1_us << " µs ("
    //               << pct(p1_us, total_us) << "%)\n"
    //               << "    - BH Dual Walk (total)   : " << total_us << " µs\n";
    // }
}


// static inline double pct(double part, double whole) {
//     return whole > 0.0 ? 100.0 * part / whole : 0.0;
// }
// static inline double ns2us(long long ns) { return ns / 1000.0; }

// void computeAccelerationsWithLET(
//     OctreeMap&                     local_tree,
//     const std::vector<NodeRecord>& remote_nodes,
//     const std::vector<uint64_t>&   bodyKey,
//     const std::vector<Position>&   pos,
//     double                         theta,
//     double                         G,
//     double                         soft2,
//     const BoundingBox&             global_bb,
//     std::vector<Acceleration>&     out,
//     int                            rank)
// {
//     // Local traversal
//     computeAccelerations(local_tree, bodyKey, pos, theta, G, soft2, global_bb, out, rank);

//     if (remote_nodes.empty()) {
//         return; // All force calculations are complete.
//     }

//     // Remote traversal
//     OctreeMap remote_tree = buildTreeFromPseudoLeaves(remote_nodes);

//     // Create a temporary vector to hold the accelerations from remote nodes.
//     std::vector<Acceleration> remote_acc;

//     // We need a dummy key vector for the remote traversal, since none of our
//     // local bodies are actually in the remote tree. This prevents the
//     // self interaction check from ever being true
//     std::vector<uint64_t> dummy_keys(pos.size(), ~uint64_t(0));

//     computeAccelerations(remote_tree, dummy_keys, pos, theta, G, soft2, global_bb, remote_acc, rank);

//     // Add the remote force contributions to the local ones
//     #pragma omp parallel for
//     for (size_t i = 0; i < pos.size(); ++i) {
//         out[i].x += remote_acc[i].x;
//         out[i].y += remote_acc[i].y;
//         out[i].z += remote_acc[i].z;
//     }
// }
