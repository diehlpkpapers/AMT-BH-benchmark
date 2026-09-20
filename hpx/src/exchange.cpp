#include "exchange.h"
#include "morton_keys.h"
#include <numeric>
#include <stack>
#include <iostream>



void exchangeFullTrees(const OctreeMap &local_tree, OctreeMap &full_tree,
                          int rank, int size, hpxc::Ctx& ctx) {
    std::vector<NodeRecord> send_buf = serializeTreeToRecords(local_tree);

    std::vector<std::vector<NodeRecord>> per_site =
        hpx::collectives::all_gather(ctx.comm_fulltree, std::move(send_buf),
            hpxc::this_site_arg(rank), hpxc::generation_arg(ctx.gen_fulltree++)).get();

    // Start from a copy of the caller's own tree, not an empty one: it
    // already has this locality's real leaves, complete with body_begin/
    // body_count. A NodeRecord can't carry that (indices are locality-local),
    // so re-merging our own slice of per_site back in would clobber every
    // local leaf to body_count == 0 - collapsing it to a monopole that
    // includes each of its own bodies with no self-exclusion (the same
    // singularity mergeRecordsIntoTree's protected_internal guards against
    // for the LET path, but that guard can't help a tree rebuilt from
    // scratch, since nothing here is "already present" to protect). Merging
    // only every *other* site's records - exactly like the LET path never
    // receiving its own contribution back - keeps our own leaves exact.
    full_tree = local_tree;
    for (int site = 0; site < size; ++site) {
        if (site == rank) continue;
        mergeRecordsIntoTree(full_tree, per_site[static_cast<std::size_t>(site)]);
    }
}


void exchangeEssentialTrees(
    const OctreeMap&                        local_tree,
    std::vector<NodeRecord>&                remote_nodes,
    std::vector<int>&                       recv_counts,
    const BoundingBox&                      global_bb,
    double                                  theta,
    int                                     rank,
    int                                     size,
    const std::vector<std::vector<uint64_t>>& rank_domain_keys,
    int                                     max_traversal_depth,
    int                                     bucket_bits,
    int                                     leaf_capacity,
    hpxc::Ctx&                              ctx)
{

    int bucket_depth = bucket_bits/3;

    // build per rank bucket AABBs with correct anisotropic dimensions
    std::vector<std::vector<BoundingBox>> bucket_boxes(size);
    double dx   = global_bb.max.x - global_bb.min.x;
    double dy   = global_bb.max.y - global_bb.min.y;
    double dz   = global_bb.max.z - global_bb.min.z;

    double cellX = dx / (1 << bucket_depth);
    double cellY = dy / (1 << bucket_depth);
    double cellZ = dz / (1 << bucket_depth);

    for(int r=0; r<size; ++r){
      bucket_boxes[r].reserve(rank_domain_keys[r].size());
      for(auto pref: rank_domain_keys[r]){
        Position n = key_to_normalized_position(pref, bucket_depth);
        BoundingBox bb;
        bb.min.x = global_bb.min.x + n.x * dx;
        bb.min.y = global_bb.min.y + n.y * dy;
        bb.min.z = global_bb.min.z + n.z * dz;
        bb.max.x = bb.min.x + cellX;
        bb.max.y = bb.min.y + cellY;
        bb.max.z = bb.min.z + cellZ;
        bucket_boxes[r].push_back(bb);
      }
    }

    // Generate per destination send lists using the corrected interaction list function
    std::vector<std::vector<NodeRecord>> send_lists(size);
    double theta2 = theta*theta;
    // Every cell the local build stopped at is now a real leaf (see
    // buildSubtreeSerial), at any leaf_capacity >= 1, so every record this
    // locality accepts into a LET must be flagged: the receiver has no
    // children for it and would otherwise open it and silently drop its mass.
    const bool mark_records_as_leaves = leaf_capacity >= 1;
    for(int dest=0; dest<size; ++dest){
        if(dest==rank) continue;
        send_lists[dest] = createInteractionListForRank(local_tree, bucket_boxes[dest], global_bb, theta2, max_traversal_depth, mark_records_as_leaves);
    }

    // Exchange the LET data (variable-size per destination).
    auto recv_per_src = hpx::collectives::all_to_all(ctx.comm_let, std::move(send_lists),
        hpxc::this_site_arg(rank), hpxc::generation_arg(ctx.gen_let++)).get();

    recv_counts.resize(size);
    remote_nodes.clear();
    for (int i = 0; i < size; ++i) {
        recv_counts[i] = static_cast<int>(recv_per_src[i].size());
        remote_nodes.insert(remote_nodes.end(), recv_per_src[i].begin(), recv_per_src[i].end());
    }
}


std::vector<NodeRecord> createInteractionListForRank(
    const OctreeMap&                      tree,
    const std::vector<BoundingBox>&       remote_boxes,
    const BoundingBox&                    global_bb,
    double                                theta2,
    int                                   max_traversal_depth,
    bool                                  mark_records_as_leaves)
{
    if (tree.empty() || remote_boxes.empty())
        return {};

    robin_hood::unordered_set<OctreeKey, OctreeKeyHash> keep;

    //  256 seems safe for depth ≤ 21 (7*d+1 bound)
    std::array<OctreeKey, 256> stk;
    int top = 0;
    if (tree.count({0ULL, 0}))
        stk[top++] = {0ULL, 0}; // push root

    while (top > 0) {
        OctreeKey k = stk[--top];
        auto it = tree.find(k);
        if (it == tree.end()) continue;
        const OctreeNode& node = it->second;

        BoundingBox node_bb = getBoundingBoxForCell(k, global_bb);

        // Barnes Hut opening test
        double sx = node_bb.max.x - node_bb.min.x;
        double sy = node_bb.max.y - node_bb.min.y;
        double sz = node_bb.max.z - node_bb.min.z;
        double s  = std::max({sx, sy, sz});
        double s2 = s * s;                // largest side squared

        bool bhFails = false;
        for (const auto& bucket : remote_boxes) {
            double d2 = min_distance_sq(node_bb, bucket);
            if (s2 >= theta2 * d2) {      // Barnes–Hut criterion
                bhFails = true;
                break;
            }
        }

        // A leaf (leaf_capacity >= 2) is never descended into, even if it
        // fails the BH test - it has no children to descend to (the whole
        // point of stopping there), so without this check a failing leaf
        // would silently be dropped from the LET (neither kept nor sent).
        if (bhFails && k.depth < max_traversal_depth && !node.is_leaf) {
            // criterion failed  and we are still allowed to descend
            uint8_t cd = k.depth + 1;
            uint64_t stride = 1ULL << (63 - 3 * cd);
            // Only the octants the build actually created (OctreeNode::
            // child_mask), instead of blind-probing all 8 candidate keys.
            uint8_t m = node.child_mask;
            while (m) {
                const unsigned oc = static_cast<unsigned>(__builtin_ctz(m));
                m &= static_cast<uint8_t>(m - 1);
                uint64_t child_prefix = k.prefix | (static_cast<uint64_t>(oc) * stride);
                OctreeKey ch{child_prefix, cd};
                if (top < 256) {
                    stk[top++] = ch;
                }
            }
        } else {
            // Accept: either BH test passed, we hit max_traversal_depth, or
            // this is a leaf that cannot be descended into further.
            keep.insert(k);
        }
    }

    std::vector<NodeRecord> out;
    out.reserve(keep.size());
    for (const auto& k : keep) {
        const auto& n = tree.at(k);
        out.push_back({k.prefix, k.depth,
                        static_cast<uint8_t>(mark_records_as_leaves ? 1 : 0),
                        n.mass, n.comX, n.comY, n.comZ});
    }
    return out;
}

