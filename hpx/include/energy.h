#pragma once

#include <cstdint>
#include <vector>

#include "body.h"
#include "bounding_box.h"
#include "hpx_collectives.h"
#include "linear_octree.h"
#include "policy.h"

// One synchronised (position/velocity) energy measurement, globally reduced.
// Units: kg * AU^2 / day^2 (consistent with G = 1.48812e-34 in main.cpp).
struct EnergySample {
    double kinetic;      // global KE  (>= 0)
    double potential;    // global PE  (< 0 for a bound system)
    double total;        // kinetic + potential
};

double computeLocalKineticEnergy(const std::vector<double>&   mass,
                                  const std::vector<Velocity>& vel);

// Barnes-Hut potential walk. Mirrors computeAccelerations() node-for-node
// (same acceptance criterion, same soft2, same depth-21 self-skip), so the
// energy estimate stays consistent with whatever approximation the force
// calculation is making.
// Returns 0.5 * sum_i mass[i] * phi_i over the LOCAL bodies only.
double computeLocalPotentialEnergy(const OctreeMap&              tree,
                                    const std::vector<uint64_t>&  key,
                                    const std::vector<Position>&  pos,
                                    const std::vector<double>&    mass,
                                    double                        theta,
                                    double                        G,
                                    double                        soft2,
                                    const BoundingBox&            global_bb);

// FCPolicy::LET_DirectSum counterpart: BH walk over the local tree plus a
// direct 1/r sum over the received remote pseudo-leaves.
double computeLocalPotentialEnergyWithRemoteDirectSum(
    const OctreeMap&                local_tree,
    const std::vector<NodeRecord>&  remote_nodes,
    const std::vector<uint64_t>&    key,
    const std::vector<Position>&    pos,
    const std::vector<double>&      mass,
    double                          theta,
    double                          G,
    double                          soft2,
    const BoundingBox&              global_bb);

// Exact O(N_total * N_local) direct-sum potential energy. Performs one
// all_gather of every particle (ctx.comm_energy_gather) - validation/small-N
// use only.
// COLLECTIVE: must be entered by every locality or the run hangs.
double computeLocalExactPotentialEnergy(const std::vector<Position>&  pos,
                                         const std::vector<double>&   mass,
                                         const std::vector<uint64_t>& ids,
                                         double                       G,
                                         double                       soft2,
                                         hpxc::Ctx&                   ctx);

// Single entry point used by main.cpp. Picks the PE variant matching `pol`
// (or the exact one if `exact_pe`), then performs exactly ONE all_reduce over
// ctx.comm_energy. Returns the identical global result on every locality.
//
// COLLECTIVE: never call this from a rank-dependent branch. The caller's
// predicate must depend only on replicated state (step, t, dt, CLI flags),
// otherwise ctx.gen_energy desynchronises and the program hangs forever.
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
                                  hpxc::Ctx&                      ctx);
