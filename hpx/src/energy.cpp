#include "energy.h"

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/collectives/all_reduce.hpp>
#include <hpx/collectives/all_gather.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

double computeLocalKineticEnergy(const std::vector<double>&   mass,
                                  const std::vector<Velocity>& vel)
{
    const std::size_t N = mass.size();
    std::vector<double> ke(N);

    hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), N, [&](std::size_t i) {
        const Velocity& v = vel[i];
        ke[i] = 0.5 * mass[i] * (v.x * v.x + v.y * v.y + v.z * v.z);
    });

    double sum = 0.0;
    for (std::size_t i = 0; i < N; ++i) sum += ke[i];
    return sum;
}

// Kept as a deliberate copy of computeAccelerations() rather than a shared
// template so the benchmarked force path stays untouched. If the acceptance
// criterion in traversal.cpp changes, change it here too or KE/PE stop being
// self-consistent with the forces.
double computeLocalPotentialEnergy(const OctreeMap&              tree,
                                    const std::vector<uint64_t>&  key,
                                    const std::vector<Position>&  pos,
                                    const std::vector<double>&    mass,
                                    double                        theta,
                                    double                        G,
                                    double                        soft2,
                                    const BoundingBox&            global_bb)
{
    const std::size_t N = pos.size();
    std::vector<double> phi(N, 0.0);
    const double theta2 = theta * theta;

    constexpr int MAX_L = 21;

    double dx0 = global_bb.max.x - global_bb.min.x;
    double dy0 = global_bb.max.y - global_bb.min.y;
    double dz0 = global_bb.max.z - global_bb.min.z;
    double L   = std::max({dx0, dy0, dz0});

    std::array<double, MAX_L + 1> side2;
    for (int d = 0; d <= MAX_L; ++d) {
        double side = L / double(1ULL << d);
        side2[d]  = side * side;
    }

    constexpr int STACK_MAX = 256;

    // Bodies are already Morton-sorted by the time this is called (main.cpp
    // step 4), so - unlike computeAccelerations() - no separate sort-order
    // pass is needed for cache locality to matter for correctness.

    // Mirrors computeAccelerations()'s walk (see there for why): one flatten
    // of the hashmap up front, after which the per-body walk is pure array
    // indexing and children are found at consecutive slots.
    const FlatTree flat = flattenTree(tree);
    const FlatNode* const nodes = flat.data();

    hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), N, [&](std::size_t i) {
        double p = 0.0;

        std::array<uint32_t, STACK_MAX> stack;
        int top = 0;
        if (!flat.empty()) stack[top++] = 0;

        while (top) {
            const FlatNode& node = nodes[stack[--top]];
            const uint64_t pref = node.prefix;
            const uint8_t  dep  = node.depth;
            if (node.mass == 0.0) continue;

            double dx = node.comX - pos[i].x;
            double dy = node.comY - pos[i].y;
            double dz = node.comZ - pos[i].z;
            double r2 = dx * dx + dy * dy + dz * dz + soft2;

            // Mirrors computeAccelerations() in traversal.cpp exactly - see
            // the comment there. A leaf is never opened; instead it is
            // resolved by exact pairwise potential summation, with any mass
            // merged in from another locality recovered via the same
            // residual-COM correction.
            if (node.is_leaf) {
                const bool self_leaf = node.body_count > 0 &&
                    i >= node.body_begin &&
                    i <  static_cast<std::size_t>(node.body_begin) + node.body_count;

                if (self_leaf || side2[dep] >= theta2 * r2) {
                    double own_m = 0.0;
                    for (uint32_t jj = 0; jj < node.body_count; ++jj) {
                        std::size_t j = static_cast<std::size_t>(node.body_begin) + jj;
                        own_m += mass[j];
                        if (j == i) continue;
                        double ddx = pos[j].x - pos[i].x;
                        double ddy = pos[j].y - pos[i].y;
                        double ddz = pos[j].z - pos[i].z;
                        double r2j = ddx * ddx + ddy * ddy + ddz * ddz + soft2;
                        if (!(r2j > 0.0)) continue;
                        p -= G * mass[j] / std::sqrt(r2j);
                    }

                    double extra = node.mass - own_m;
                    if (extra > 1e-12 * node.mass) {
                        double own_wx = 0.0, own_wy = 0.0, own_wz = 0.0;
                        for (uint32_t jj = 0; jj < node.body_count; ++jj) {
                            std::size_t j = static_cast<std::size_t>(node.body_begin) + jj;
                            own_wx += mass[j] * pos[j].x;
                            own_wy += mass[j] * pos[j].y;
                            own_wz += mass[j] * pos[j].z;
                        }
                        double ex = (node.mass * node.comX - own_wx) / extra;
                        double ey = (node.mass * node.comY - own_wy) / extra;
                        double ez = (node.mass * node.comZ - own_wz) / extra;
                        double edx = ex - pos[i].x, edy = ey - pos[i].y, edz = ez - pos[i].z;
                        double er2 = edx * edx + edy * edy + edz * edz + soft2;
                        if (er2 > 0.0) {
                            p -= G * extra / std::sqrt(er2);
                        }
                    }
                    continue;
                }
            }

            if (dep == MAX_L && pref == key[i]) continue;

            if (node.is_pseudo || node.is_leaf || dep == MAX_L || side2[dep] < theta2 * r2) {
                double invR = 1.0 / std::sqrt(r2);
                p -= G * node.mass * invR;
            } else {
                // Children are contiguous - no hashmap lookups, no blind
                // probing of the (measured ~27%) empty candidate octants.
                if (top + node.child_count > STACK_MAX)
                    throw std::runtime_error("BH stack overflow");
                for (uint32_t c = 0; c < node.child_count; ++c)
                    stack[top++] = node.children_offset + c;
            }
        }

        phi[i] = p;
    });

    double sum = 0.0;
    for (std::size_t i = 0; i < N; ++i) sum += mass[i] * phi[i];
    return 0.5 * sum;
}

double computeLocalPotentialEnergyWithRemoteDirectSum(
    const OctreeMap&                local_tree,
    const std::vector<NodeRecord>&  remote_nodes,
    const std::vector<uint64_t>&    key,
    const std::vector<Position>&    pos,
    const std::vector<double>&      mass,
    double                          theta,
    double                          G,
    double                          soft2,
    const BoundingBox&              global_bb)
{
    double local_part = computeLocalPotentialEnergy(local_tree, key, pos, mass, theta, G, soft2, global_bb);

    const std::size_t N = pos.size();
    const std::size_t M = remote_nodes.size();
    std::vector<double> rx(M), ry(M), rz(M), rm(M);
    for (std::size_t j = 0; j < M; ++j) {
        rx[j] = remote_nodes[j].comX;
        ry[j] = remote_nodes[j].comY;
        rz[j] = remote_nodes[j].comZ;
        rm[j] = remote_nodes[j].mass * G;
    }

    std::vector<double> phi_remote(N, 0.0);
    hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), N, [&](std::size_t i) {
        double p = 0.0;
        const double px = pos[i].x;
        const double py = pos[i].y;
        const double pz = pos[i].z;

        for (std::size_t j = 0; j < M; ++j) {
            double dx = rx[j] - px;
            double dy = ry[j] - py;
            double dz = rz[j] - pz;
            double r2 = dx * dx + dy * dy + dz * dz + soft2;
            if (r2 <= 1e-12) continue;

            p -= rm[j] / std::sqrt(r2);
        }
        phi_remote[i] = p;
    });

    double remote_sum = 0.0;
    for (std::size_t i = 0; i < N; ++i) remote_sum += mass[i] * phi_remote[i];

    return local_part + 0.5 * remote_sum;
}

double computeLocalExactPotentialEnergy(const std::vector<Position>&  pos,
                                         const std::vector<double>&   mass,
                                         const std::vector<uint64_t>& ids,
                                         double                       G,
                                         double                       soft2,
                                         hpxc::Ctx&                   ctx)
{
    const std::size_t N = pos.size();

    std::vector<hpxc::TransferBody> mine(N);
    for (std::size_t i = 0; i < N; ++i)
        mine[i] = {pos[i], mass[i], Velocity{0.0, 0.0, 0.0}, ids[i]};

    auto per_site = hpx::collectives::all_gather(ctx.comm_energy_gather, std::move(mine),
        hpxc::this_site_arg(ctx.rank), hpxc::generation_arg(ctx.gen_energy_gather++)).get();
    auto all = hpxc::flatten(std::move(per_site));
    const std::size_t M = all.size();

    std::vector<double> phi(N, 0.0);
    hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), N, [&](std::size_t i) {
        double p = 0.0;
        const double px = pos[i].x;
        const double py = pos[i].y;
        const double pz = pos[i].z;
        const uint64_t my_id = ids[i];

        for (std::size_t j = 0; j < M; ++j) {
            if (all[j].id == my_id) continue;
            double dx = all[j].pos.x - px;
            double dy = all[j].pos.y - py;
            double dz = all[j].pos.z - pz;
            double r2 = dx * dx + dy * dy + dz * dz + soft2;
            p -= G * all[j].mass / std::sqrt(r2);
        }
        phi[i] = p;
    });

    double sum = 0.0;
    for (std::size_t i = 0; i < N; ++i) sum += mass[i] * phi[i];
    return 0.5 * sum;
}

EnergySample computeGlobalEnergy(FCPolicy                        pol,
                                  bool                            exact_pe,
                                  const OctreeMap&                bh_tree,
                                  const std::vector<NodeRecord>&  remote_nodes,
                                  const std::vector<uint64_t>&    key,
                                  const std::vector<Position>&    pos,
                                  const std::vector<Velocity>&    vel,
                                  const std::vector<double>&      mass,
                                  const std::vector<uint64_t>&    ids,
                                  double                          theta,
                                  double                          G,
                                  double                          soft2,
                                  const BoundingBox&              global_bb,
                                  hpxc::Ctx&                      ctx)
{
    double ke = computeLocalKineticEnergy(mass, vel);

    double pe;
    if (exact_pe) {
        pe = computeLocalExactPotentialEnergy(pos, mass, ids, G, soft2, ctx);
    } else if (pol == FCPolicy::LET_DirectSum) {
        pe = computeLocalPotentialEnergyWithRemoteDirectSum(
            bh_tree, remote_nodes, key, pos, mass, theta, G, soft2, global_bb);
    } else {
        // FCPolicy::Tree or FCPolicy::LET: bh_tree already contains everything
        // (the full tree, or the local tree merged with the LET remote nodes).
        pe = computeLocalPotentialEnergy(bh_tree, key, pos, mass, theta, G, soft2, global_bb);
    }

    std::array<double, 2> local{ke, pe};
    auto g = hpx::collectives::all_reduce(ctx.comm_energy, local, hpxc::SumArr2{},
        hpxc::this_site_arg(ctx.rank), hpxc::generation_arg(ctx.gen_energy++)).get();

    return EnergySample{g[0], g[1], g[0] + g[1]};
}
