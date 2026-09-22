#pragma once
#include <vector>

#include "policy.h"
#include "linear_octree.h"
#include "bounding_box.h"
#include "hpx_collectives.h"


/**
 * @brief Exchanges the complete local octrees between all ranks (allgatherv-equivalent).
 * @param local_tree The calling rank's local octree.
 * @param[out] full_tree The output map containing the merged, global octree.
 */
void exchangeFullTrees(const OctreeMap &local_tree, OctreeMap &full_tree,
                          int rank, int size, hpxc::Ctx& ctx);




/**
 * @brief Generates and exchanges Locally Essential Trees (LETs) between all ranks.
 *
 * Each rank determines which parts of its local tree are essential for every other
 * rank, and then performs an MPI_Alltoallv to send and receive these pseudo-leaves.
 *
 * @param local_tree The local octree of the calling rank.
 * @param[out] remote_nodes A vector containing all pseudo-leaves received from other ranks.
 * @param[out] recv_counts A vector where recv_counts[i] is the number of nodes from rank i.
 * @param global_bb The global bounding box of the simulation.
 * @param theta The Barnes-Hut opening angle used to generate interaction lists.
 */
void exchangeEssentialTrees(
    const OctreeMap&                        local_tree,
    std::vector<NodeRecord>&                remote_nodes, // Output
    std::vector<int>&                       recv_counts, //output
    const BoundingBox&                      global_bb,
    double                                  theta,
    int                                     rank,
    int                                     size,
    const std::vector<std::vector<uint64_t>>& rank_domain_keys,
    int                                     max_traversal_depth,
    int                                     bucket_bits,
    int                                     leaf_capacity,   // >=2 marks LET records as leaves
    hpxc::Ctx&                              ctx);



/**
 * @brief Traverses a local tree to find the nodes essential for a remote set of domains.
 *
 * Creates a list of pseudo-leaves from the local 'tree' that are needed to compute
 * forces on bodies located in the 'remote_boxes'. A node is considered essential if
 * it fails the Barnes-Hut criterion for any part of the remote domain.
 *
 * @param tree The local octree to traverse.
 * @param remote_boxes A vector of bounding boxes defining the domains of the destination rank.
 * @return A vector of NodeRecords representing the interaction list (the LET).
 */
std::vector<NodeRecord> createInteractionListForRank(
        const OctreeMap&                      tree,
        const std::vector<BoundingBox>&       remote_boxes,   // all buckets of dest rank
        const BoundingBox&                    global_bb,
        double                                theta2,
        int                                   max_traversal_depth,
        bool                                  mark_records_as_leaves);


