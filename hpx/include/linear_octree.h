#pragma once

#include <vector>
#include <cstdint>
#include <unordered_map>
#include "robin_hood.h"
#include "sharded_map.h"
#include "body.h"

/**
 * @brief Defines a unique key for a node in the octree.
 *
 * A key is a combination of a Morton prefix and a depth. This uniquely identifies
 * any possible cell within the octree's spatial domain.
 */
struct OctreeKey {
    uint64_t prefix;
    uint8_t  depth;

    bool operator==(const OctreeKey& other) const noexcept {
        return prefix == other.prefix && depth == other.depth;
    }
};

// A 63-bit Morton code resolves 21 levels (21 * 3 = 63 bits), so depth 21 is
// the deepest a cell can ever be. The build stops well above it wherever a
// cell already holds <= leaf_capacity bodies; only bodies sharing an
// identical Morton code force a cell all the way down here.
inline constexpr uint8_t BH_MAX_DEPTH = 21;

// Custom hash function for OctreeKey to be used in std::unordered_map.
struct OctreeKeyHash {
    size_t operator()(const OctreeKey& k) const noexcept {
        // Pack depth into low bits so different depths never collide when prefixes match.
        // Then mix with a 64-bit finalizer (splitmix64 style).
        uint64_t x = k.prefix ^ (uint64_t{k.depth} * 0x9e3779b97f4a7c15ULL);
        // final avalanche
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<size_t>(x);
    }
};

/**
 * @brief Represents a single node in the Barnes-Hut octree.
 *
 * Stores the consolidated mass properties (total mass and center of mass)
 * for all particles contained within the node's spatial volume. It also
 * tracks if the node is a pseudo-leaf from a remote rank.
 */
struct OctreeNode {
    double mass{0.0};
    double comX{0.0};
    double comY{0.0};
    double comZ{0.0};
    bool is_pseudo{false};

    // True when the build stopped at this cell and did not subdivide it (or
    // it is a remote record that was marked as a leaf on
    // some locality - see mergeRecordsIntoTree/buildTreeFromPseudoLeaves,
    // which OR this in sticky). A node with is_leaf must never be opened by
    // any walk: its full mass is consumed right there. A merged tree can
    // still give such a node children (another locality subdivided the same
    // cell) - accepting the node's total mass as a monopole/direct-sum is
    // mass-conserving; descending into those children would lose this
    // locality's own leaf mass.
    bool is_leaf{false};

    // [body_begin, body_begin + body_count) into the Morton-sorted LOCAL
    // arrays passed to buildOctreeBottomUp (codes/local_pos/local_mass in
    // main.cpp). Valid only when is_leaf && body_count > 0; zero for nodes
    // that arrived over the wire (they never carry a local particle list).
    uint32_t body_begin{0};
    uint32_t body_count{0};

    // Bit `oc` is set iff the child cell {prefix | oc << (63-3*(depth+1)),
    // depth+1} exists in the same map. Filled in by the build (which already
    // knows exactly which octants it created - see buildSubtreeSerial /
    // buildSubtreeAsync in linear_octree.cpp) and kept up to date by
    // mergeRecordsIntoTree / buildTreeFromPseudoLeaves, the only other places
    // that ever add keys to a tree that is later walked.
    //
    // Lets every tree walk look up only the children that actually exist
    // instead of blind-probing all 8 candidate keys: measured on
    // scenario2_300149 at theta 1.05, 73.0M of the walk's 275.4M child probes
    // per step (26.5%) were guaranteed misses (an opened node has 5.88 of 8
    // octants on average). Not transmitted over the wire (NodeRecord is
    // unchanged); it is purely local structural information about one map.
    uint8_t child_mask{0};
};

/**
 * @brief One node of the flattened, directly-indexed copy of an OctreeMap.
 *
 * The walk-time replacement for a hashmap entry: a node's children live at
 * consecutive indices [children_offset, children_offset + child_count), so a
 * descent costs one array read instead of a hash + shard select + probe.
 * Carries everything the Barnes-Hut walks read off an OctreeNode plus the
 * key fields (prefix/depth) they used to get from the traversal stack.
 */
struct FlatNode {
    double   mass{0.0};
    double   comX{0.0}, comY{0.0}, comZ{0.0};
    uint64_t prefix{0};
    uint32_t children_offset{0};   // meaningful only when child_count > 0
    uint32_t body_begin{0};
    uint32_t body_count{0};
    uint8_t  child_count{0};       // 0 for every node a walk must not open
    uint8_t  depth{0};
    bool     is_leaf{false};
    bool     is_pseudo{false};
};

using FlatTree = std::vector<FlatNode>;


// Hash map partitioned into independent shards so the build can fill it
// from many tasks at once (sharded_map.h); only iteration order changes.
// 128, not 256: the adaptive-stop leaf_capacity==1 fix (see
// buildSubtreeSerial) cut this tree from ~3.26M to ~447K nodes, so 256
// shards now average far fewer nodes each - re-measured empirically and
// 128 scales better across 32-128 threads for this smaller tree.
using OctreeMap = ShardedMap<OctreeKey, OctreeNode, OctreeKeyHash, 128>;

/**
 * @brief Flattens an OctreeMap into a contiguous, directly-indexed FlatTree.
 *
 * Index 0 is the root ({0,0}); a node's children occupy consecutive slots.
 * Only the part of the tree a Barnes-Hut walk can actually reach is emitted:
 * a node is expanded only when the walks' own opening rule would ever open it
 * (!is_leaf && !is_pseudo && depth < BH_MAX_DEPTH). Children are emitted in
 * ascending octant order, exactly the order the old 8-way probe pushed them
 * in, so a walk over the FlatTree visits nodes in the identical sequence and
 * accumulates bit-identical sums.
 *
 * Built once per force-calculation call (same cadence as the tree itself):
 * a short serial breadth-first "backbone" from the root until there are
 * enough independent subtree roots to keep every worker busy, then one
 * private buffer per subtree filled in parallel, then a cheap copy/offset-fix
 * pass. Returns an empty vector if the tree has no root node.
 */
FlatTree flattenTree(const OctreeMap& tree);

// A plain data structure for sending/receiving tree nodes across localities.
struct NodeRecord {
    uint64_t prefix;
    uint8_t  depth;
    uint8_t  is_leaf;    // 0/1 - occupies bytes that were already padding, so
                         // sizeof/offsets below are unchanged (see static_assert)
    double   mass;
    double   comX, comY, comZ;
};

static_assert(sizeof(NodeRecord) == 48,
    "NodeRecord crosses HPX collectives as raw bytes (HPX_IS_BITWISE_SERIALIZABLE); "
    "its wire layout must not change size");

HPX_IS_BITWISE_SERIALIZABLE(NodeRecord);

/**
 * @brief Builds a linear octree top-down, stopping and marking a cell as a
 * leaf once it holds at most leaf_capacity bodies (or depth 21 is reached).
 *
 * Recurses via HPX tasks, joining children with hpx::dataflow instead of a
 * blocking wait, then scatters and merges per-subtree buffers into the
 * sharded OctreeMap in parallel (see linear_octree.cpp). Bit-identical to a
 * serial build; only iteration order is unspecified.
 *
 * @param mortonCodes A sorted (ascending) vector of Morton keys for the bodies.
 * @param positions The positions of the bodies, permuted to match mortonCodes.
 * @param masses The masses of the bodies, permuted to match mortonCodes.
 * @param leaf_capacity Maximum bodies per unsubdivided cell, >= 1. A cell is
 *   stopped, flagged is_leaf and given its body range for exact pairwise
 *   summation as soon as it holds <= leaf_capacity bodies. 1 (default) is the
 *   classic recursive Barnes-Hut rule: stop once a single body is left.
 *   Depth 21 is then only reached by bodies that share a Morton code.
 * @return An OctreeMap representing the completed octree.
 */
OctreeMap buildOctreeBottomUp(
    const std::vector<uint64_t>& mortonCodes,
    const std::vector<Position>& positions,
    const std::vector<double>&   masses,
    int                          leaf_capacity = 1);

/**
 * @brief Constructs an octree from a list of received pseudo-leaves (NodeRecord).
 *
 * This function is used to reconstruct a remote tree structure on the receiving rank.
 * The nodes corresponding to the input records are marked as 'is_pseudo=true' to prevent
 * the traversal algorithm from opening them.
 *
 * @param remote_nodes A vector of nodes received from other ranks.
 * @return An OctreeMap representing the remote interaction tree.
 */
OctreeMap buildTreeFromPseudoLeaves(const std::vector<NodeRecord>& remote_nodes);


/**
 * @brief Serializes an OctreeMap into a flat vector of NodeRecord structs for communication.
 */
std::vector<NodeRecord> serializeTreeToRecords(const OctreeMap& tree);

/**
 * @brief Merges a vector of NodeRecord structs into an existing OctreeMap.
 *
 * This function adds the mass and combines the center-of-mass data for each record
 * into the corresponding node in the destination tree.
 */
void mergeRecordsIntoTree(OctreeMap& tree, const std::vector<NodeRecord>& records);

void printTree(const char* title, const OctreeMap& tree);

void compareFlattened(const std::vector<NodeRecord> A,
                      const std::vector<NodeRecord> B,
                      double tol = 1e-9);

