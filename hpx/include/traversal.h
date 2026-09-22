#pragma once

#include "linear_octree.h"
#include "body.h"
#include "bounding_box.h"




/**
 * @brief Computes gravitational accelerations for a set of bodies using a single, pre-built octree.
 * @param tree The complete octree containing all mass information (local and/or remote).
 * @param bodyKeys Morton keys of the local bodies for which to calculate forces.
 * @param positions Positions of the local bodies.
 * @param theta The Barnes-Hut opening angle criterion.
 * @param G The gravitational constant.
 * @param soft2 The softening factor squared (ε²) to avoid singularities.
 * @param global_bb The global bounding box of the entire simulation space.
 * @param[out] out The output vector where calculated accelerations will be stored.
 */
void computeAccelerations(
    const OctreeMap&                tree,
    const std::vector<uint64_t>&    key,       // one Morton key per body
    const std::vector<Position>&    pos,       // physical coords (e.g. AU)
    const std::vector<double>&      mass,      // needed for exact leaf direct-sums
    double                          theta,
    double                          G,
    double                          soft2,     // ε²
    const BoundingBox&              global_bb, // from rebalance_bodies
    std::vector<Acceleration>&      out,
    int                             rank=0);



void computeAccelerationsWithRemoteDirectSum(
const OctreeMap&                local_tree,
const std::vector<NodeRecord>&  remote_nodes,
const std::vector<uint64_t>&    key,
const std::vector<Position>&    pos,
const std::vector<double>&      mass,
double                          theta,
double                          G,
double                          soft2,
const BoundingBox&              global_bb,
std::vector<Acceleration>&      out,
int                             rank=0);


/**
 * @brief Computes accelerations by merging a remote tree of pseudo-leaves with the local tree.
 *
 * This function implements the full LET (Locally Essential Tree) force calculation.
 * It takes the local tree and merges the received remote pseudo-leaves into it.
 * It then performs a single, unified traversal on the combined tree to calculate the final forces.
 * The local_tree is modified in-place.
 *
 * @param[in,out] local_tree The local octree, which will be modified to include remote data.
 * @param remote_nodes The vector of pseudo-leaves received from other ranks.
 * @param bodyKeys Morton keys of the local bodies.
 * @param positions Positions of the local bodies.
 * @param theta The Barnes-Hut opening angle.
 * @param G The gravitational constant.
 * @param soft2 The softening factor squared (ε²).
 * @param global_bb The global bounding box of the simulation.
 * @param[out] out The output vector for the final calculated accelerations.
 */
void computeAccelerationsWithLET(
    OctreeMap&               local_tree,
    const std::vector<NodeRecord>& remote_nodes,
    const std::vector<uint64_t>&   bodyKey,   // local leaf keys
    const std::vector<Position>&   pos,
    const std::vector<double>&     mass,
    double                         theta,
    double                         G,
    double                         soft2,
    const BoundingBox&             global_bb,
    std::vector<Acceleration>&     out,
    int                            rank);