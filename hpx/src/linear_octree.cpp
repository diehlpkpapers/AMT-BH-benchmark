#include "linear_octree.h"
#include <algorithm>
#include <array>
#include <numeric>
#include <vector>
#include <iostream>
#include <morton_keys.h>
#include <chrono>
#include <hpx/async.hpp>
#include <hpx/future.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <atomic>
#include <hpx/runtime_local/get_os_thread_count.hpp>
#include <hpx/execution/executors/dynamic_chunk_size.hpp>
#include <limits>

namespace {

// Mass and mass-weighted coordinate sums for one subtree.
struct Agg { double m = 0.0, wx = 0.0, wy = 0.0, wz = 0.0; };

inline void addAgg(Agg& a, const Agg& c) {
    a.m += c.m; a.wx += c.wx; a.wy += c.wy; a.wz += c.wz;
}

inline void finalizeCom(OctreeNode& n) {
    if (n.mass > 0) { n.comX /= n.mass; n.comY /= n.mass; n.comZ /= n.mass; }
}

// One (key, node) pair, in the serial build's insertion order (post-order).
struct NodeOut { OctreeKey key; OctreeNode node; };
using NodeBuf = std::vector<NodeOut>;

// Private, append-only subtree output (see buildSubtreeAsync). Depth-first
// children-then-own walk reproduces the serial build's insertion order.
struct Chunk { std::vector<Chunk> children; NodeBuf own; };
struct SubtreeResult { Agg agg; Chunk chunk; };

// The serial recursion, writing into a private post-order buffer instead of
// a shared OctreeMap. Must not hold a NodeOut&/OctreeNode& into `out` across
// recursive calls: push_back can reallocate it.
Agg buildSubtreeSerial(NodeBuf& out,
                        const std::vector<uint64_t>& codes,
                        const std::vector<Position>& pos,
                        const std::vector<double>&   mass,
                        uint64_t prefix, uint8_t depth,
                        std::size_t lo, std::size_t hi,
                        int leaf_capacity)
{
    const std::size_t n = hi - lo;
    // Adaptive stop: a cell holding at most leaf_capacity bodies cannot be
    // usefully split any further, so stop right there. leaf_capacity is
    // validated >= 1 (main.cpp), so at the default 1 this is the classic
    // recursive Barnes-Hut rule - stop as soon as one body is left - and the
    // tree only ever reaches BH_MAX_DEPTH where bodies share a Morton code.
    const bool stop = (depth == BH_MAX_DEPTH) ||
                       (n <= static_cast<std::size_t>(leaf_capacity));

    Agg a;
    uint8_t child_mask = 0;   // octants actually created below (Fix 1)

    if (stop) {
        // Leaf: ascending index order, matching the original code.
        for (std::size_t j = lo; j < hi; ++j) {
            double m = mass[j];
            a.m  += m;
            a.wx += pos[j].x * m;
            a.wy += pos[j].y * m;
            a.wz += pos[j].z * m;
        }
    } else {
        // Internal: Morton-sorted, so each octant's bodies form one
        // contiguous sub-range a left-to-right sweep can find.
        const uint8_t cdep = depth + 1;
        const uint64_t shift = 63 - 3 * cdep;
        std::size_t j = lo;
        for (int oc = 0; oc < 8; ++oc) {
            std::size_t k = j;
            while (k < hi && ((codes[k] >> shift) & 7ULL) == static_cast<uint64_t>(oc))
                ++k;
            if (k > j) {
                uint64_t child_prefix = prefix | (static_cast<uint64_t>(oc) << shift);
                Agg c = buildSubtreeSerial(out, codes, pos, mass, child_prefix, cdep, j, k, leaf_capacity);
                addAgg(a, c);
                child_mask |= static_cast<uint8_t>(1u << oc);
            }
            j = k;
        }
    }

    OctreeNode node;
    node.mass = a.m;
    node.comX = a.wx;
    node.comY = a.wy;
    node.comZ = a.wz;
    node.child_mask = child_mask;    // 0 when stop, by construction
    if (stop) {
        node.is_leaf = true;
        node.body_begin = static_cast<uint32_t>(lo);
        node.body_count = static_cast<uint32_t>(n);
    }
    finalizeCom(node);
    out.push_back({OctreeKey{prefix, depth}, node});
    return a;
}

// Fall through to the serial recursion below this many bodies. Measured on
// kamand1 (128 cores, 1-128 threads): 4096 beats both 1024 (11-28% slower)
// and 8192 (worse at 16/64/128 threads).
constexpr std::size_t kTreeBuildTaskCutoff = 4096;

// Upper bound on nodes one buildSubtreeSerial call appends, so the caller
// can reserve its buffer up front. stop => exactly 1 node; !stop => n is
// already <= kTreeBuildTaskCutoff, bounding every call regardless of depth.
//
// An internal cell holds >= leaf_capacity+1 bodies and cells at one depth are
// disjoint, so each depth carries at most n/(leaf_capacity+1) of them; every
// other node is a leaf, and leaves hold >= 1 body each, so there are at most
// min(n, 8*internal) of those. Holds for leaf_capacity == 1 too, now that the
// adaptive stop applies there as well.
std::size_t serialNodeUpperBound(std::size_t n, uint8_t depth,
                                  int leaf_capacity, bool stop)
{
    if (stop) return 1;

    const std::size_t levels = std::size_t{BH_MAX_DEPTH} + 1u - depth;
    const std::size_t internal =
        levels * std::max<std::size_t>(n / (static_cast<std::size_t>(leaf_capacity) + 1u), 1u);
    return internal + std::min(n, 8u * internal);
}

// Non-owning pointers, passed BY VALUE - a reference would dangle, since
// buildSubtreeAsync returns while spawned tasks are still pending. Raw
// pointer suffices: buildOctreeBottomUp's root .get() outlives the DAG.
struct Inputs {
    const std::vector<uint64_t>* codes;
    const std::vector<Position>* pos;
    const std::vector<double>*   mass;
    int leaf_capacity;
};
static_assert(std::is_trivially_copyable_v<Inputs>);

// Dataflow layer over buildSubtreeSerial: returns a future instead of
// blocking, so combining children runs as a task once ready. Root .get() in
// buildOctreeBottomUp is the only blocking wait in the whole build.
//
// `futs` is filled in ascending octant order, and dataflow preserves that
// order regardless of completion order, so results stay bit-identical to
// the serial sweep. Tasks spawn only during descent, never inside a
// combine, so the DAG is finite and always drains, even at --hpx:threads=1.
hpx::future<SubtreeResult> buildSubtreeAsync(Inputs in,
                                              uint64_t prefix, uint8_t depth,
                                              std::size_t lo, std::size_t hi)
{
    const std::vector<uint64_t>& codes = *in.codes;
    const std::size_t n = hi - lo;
    // Must stay identical to buildSubtreeSerial's stop rule above.
    const bool stop = (depth == BH_MAX_DEPTH) ||
                       (n <= static_cast<std::size_t>(in.leaf_capacity));

    if (stop || n <= kTreeBuildTaskCutoff) {
        SubtreeResult r;
        r.chunk.own.reserve(serialNodeUpperBound(n, depth, in.leaf_capacity, stop));
        r.agg = buildSubtreeSerial(r.chunk.own, codes, *in.pos, *in.mass,
                                    prefix, depth, lo, hi, in.leaf_capacity);
        return hpx::make_ready_future(std::move(r));  // no task, no continuation
    }

    // Same left-to-right octant sweep as the serial code.
    const uint8_t cdep = depth + 1;
    const uint64_t shift = 63 - 3 * cdep;
    struct Range { uint64_t prefix; std::size_t lo, hi; };
    std::array<Range, 8> rg;
    int nr = 0;
    uint8_t child_mask = 0;   // octants actually spawned below (Fix 1)
    std::size_t j = lo;
    for (int oc = 0; oc < 8; ++oc) {
        std::size_t k = j;
        while (k < hi && ((codes[k] >> shift) & 7ULL) == static_cast<uint64_t>(oc))
            ++k;
        if (k > j) {
            rg[static_cast<std::size_t>(nr++)] = {prefix | (static_cast<uint64_t>(oc) << shift), j, k};
            child_mask |= static_cast<uint8_t>(1u << oc);
        }
        j = k;
    }

    std::vector<hpx::future<SubtreeResult>> futs;
    futs.reserve(static_cast<std::size_t>(nr));
    for (int c = 0; c + 1 < nr; ++c) {
        const Range q = rg[static_cast<std::size_t>(c)];  // captured by value (small POD)
        // Unwraps future<future<...>> into one future ready once the
        // child's whole subtree DAG completes.
        futs.emplace_back(hpx::future<SubtreeResult>(
            hpx::async([in, q, cdep] {
                return buildSubtreeAsync(in, q.prefix, cdep, q.lo, q.hi);
            })));
    }
    // Last occupied octant: its DAG is built on this task (one spawn saved,
    // as before). Appended last, keeping `futs` in ascending octant order.
    const Range lastRg = rg[static_cast<std::size_t>(nr - 1)];
    futs.push_back(buildSubtreeAsync(in, lastRg.prefix, cdep, lastRg.lo, lastRg.hi));

    // Explicit launch::async: combine runs as a fresh task, never inline on
    // a child's stack.
    return hpx::dataflow(
        hpx::launch::async,
        [prefix, depth, child_mask](std::vector<hpx::future<SubtreeResult>> cf) -> SubtreeResult {
            Agg a;
            SubtreeResult r;
            r.chunk.children.reserve(cf.size());
            for (hpx::future<SubtreeResult>& cfut : cf) {
                // Ready by construction - never suspends.
                SubtreeResult cr = cfut.get();
                addAgg(a, cr.agg);
                r.chunk.children.push_back(std::move(cr.chunk));
            }
            OctreeNode node;  // internal node => is_leaf/body_* stay defaulted
            node.mass = a.m;
            node.comX = a.wx;
            node.comY = a.wy;
            node.comZ = a.wz;
            node.child_mask = child_mask;
            finalizeCom(node);
            r.chunk.own.push_back({OctreeKey{prefix, depth}, node});
            r.agg = a;
            return r;
        },
        std::move(futs));
}

// Collects every non-empty per-subtree buffer plus the total node count -
// walk order no longer matters, since the fill below is concurrent.
void collectBufs(Chunk& c, std::vector<NodeBuf*>& out, std::size_t& total) {
    for (Chunk& ch : c.children) collectBufs(ch, out, total);
    if (!c.own.empty()) { total += c.own.size(); out.push_back(&c.own); }
}

}    // namespace

OctreeMap buildOctreeBottomUp(const std::vector<uint64_t>& mortonCodes,
                               const std::vector<Position>& positions,
                               const std::vector<double>&   masses,
                               int                          leaf_capacity)
{
    const size_t N = mortonCodes.size();
    if (N == 0) return {};

    // The ONE blocking wait in the entire build - the caller immediately
    // feeds the result into the tree-exchange collectives, and once ready
    // the whole DAG has drained (proving Inputs' pointers stay valid).
    SubtreeResult root =
        buildSubtreeAsync(Inputs{&mortonCodes, &positions, &masses, leaf_capacity},
                           0ULL, 0, 0, N).get();

    std::vector<NodeBuf*> bufs;
    std::size_t total = 0;
    collectBufs(root.chunk, bufs, total);

    OctreeMap tree;

    // Scatter-then-merge fill, replacing a design that inserted directly
    // into the sharded `tree` under one spinlock per shard - measured 7.5x
    // slower at 128 threads than at 32 (256-way spinlock contention).
    // num_cores() and hpx::mutex were tried first and both failed.
    //
    // Makes it structurally impossible for two tasks to touch the same
    // shard at once:
    //  1. Scatter: each worker claims buffers from a shared atomic counter
    //     and appends every entry to its OWN private bucket per shard.
    //  2. Merge: one task per shard drains every worker's bucket for that
    //     shard into `tree` via insert_shard_unsynchronized() - safe since
    //     no other task touches that shard.
    //
    // Trade-off: each entry is copied twice instead of once, but that's far
    // cheaper than the lock contention it replaces.
    //
    // Capped at 32 regardless of thread count: with the adaptive-stop
    // leaf_capacity==1 fix, this tree now has ~447K nodes (was ~3.26M), so
    // there are only a few hundred buffers total to scatter - spawning up
    // to 128 workers to contend on one shared claim counter for that little
    // work measurably hurt at high thread counts; 32 was empirically best.
    const std::size_t scatter_workers = std::min({bufs.size(), hpx::get_os_thread_count(), std::size_t(32)});
    std::vector<std::vector<std::vector<NodeOut>>> scatter(
        scatter_workers, std::vector<std::vector<NodeOut>>(OctreeMap::shard_count));

    if (scatter_workers > 0) {
        std::atomic<std::size_t> next_buf{0};
        std::vector<hpx::future<void>> scatter_futs;
        scatter_futs.reserve(scatter_workers);
        for (std::size_t w = 0; w < scatter_workers; ++w) {
            scatter_futs.push_back(hpx::async([&bufs, &next_buf, &scatter, w]() {
                std::vector<std::vector<NodeOut>>& my_buckets = scatter[w];
                for (;;) {
                    std::size_t i = next_buf.fetch_add(1, std::memory_order_relaxed);
                    if (i >= bufs.size()) break;
                    NodeBuf& b = *bufs[i];
                    for (const NodeOut& e : b) {
                        my_buckets[OctreeMap::shard_of(e.key)].push_back(e);
                    }
                    NodeBuf().swap(b);
                }
            }));
        }
        hpx::when_all(scatter_futs).get();
    }

    hpx::experimental::for_loop(
        hpx::execution::par, std::size_t(0), OctreeMap::shard_count,
        [&tree, &scatter, scatter_workers](std::size_t s) {
            std::size_t shard_total = 0;
            for (std::size_t w = 0; w < scatter_workers; ++w) shard_total += scatter[w][s].size();
            tree.reserve_shard(s, shard_total);
            for (std::size_t w = 0; w < scatter_workers; ++w) {
                std::vector<NodeOut>& bucket = scatter[w][s];
                for (const NodeOut& e : bucket) tree.insert_shard_unsynchronized(s, e.key, e.node);
                std::vector<NodeOut>().swap(bucket);
            }
        });

    // A size mismatch means a duplicate key or a lost insert - a real bug,
    // worth catching unconditionally (the check is O(#shards)).
    if (tree.size() != total) {
        throw std::runtime_error("octree build: sharded fill produced " +
                                  std::to_string(tree.size()) + " of " +
                                  std::to_string(total) + " nodes");
    }
    return tree;                            // COM already finalised per node
}


// ---------------------------------------------------------------------------
// Fix 2: flatten the hashmap into a directly-indexed array for the walks.
// ---------------------------------------------------------------------------
namespace {

// Independent subtrees to aim for per worker before the serial backbone
// stops. Higher = better load balance across subtrees of wildly different
// sizes (a handful of them hold most of the nodes), at the cost of a longer
// serial backbone. 32 puts the backbone at a couple of percent of the tree
// at 128 threads and still leaves ~50 nodes in an average subtree.
constexpr std::size_t kFlattenSubtreesPerWorker = 32;

// A walk opens a node only if none of its acceptance rules fires first (see
// computeAccelerations / computeLocalPotentialEnergy): a leaf, a remote
// pseudo-node and a depth-BH_MAX_DEPTH cell are all consumed where they are,
// never descended into. Anything not expandable therefore gets child_count
// == 0, which also prunes its (unreachable) subtree out of the FlatTree.
inline bool flatExpandable(const OctreeNode& n, uint8_t depth) {
    return !n.is_leaf && !n.is_pseudo && depth < BH_MAX_DEPTH && n.child_mask != 0;
}

inline FlatNode makeFlat(uint64_t prefix, uint8_t depth, const OctreeNode& n) {
    FlatNode f;
    f.mass = n.mass;
    f.comX = n.comX;
    f.comY = n.comY;
    f.comZ = n.comZ;
    f.prefix = prefix;
    f.body_begin = n.body_begin;
    f.body_count = n.body_count;
    f.depth = depth;
    f.is_leaf = n.is_leaf;
    f.is_pseudo = n.is_pseudo;
    return f;                    // children_offset/child_count set on expansion
}

// A node already written to some buffer at `slot`, whose children have not
// been appended yet.
struct FrontierItem {
    uint64_t          prefix;
    const OctreeNode* node;
    uint32_t          slot;
    uint8_t           depth;
};

// Appends the existing children of (prefix, depth) to `out` in ascending
// octant order - exactly the order the old 8-way blind probe pushed them in,
// which is what keeps a FlatTree walk bit-identical to the hashmap walk - and
// queues the ones a walk could open. Returns how many were appended.
//
// Only child_mask's set bits are looked up: this is where Fix 1 pays off,
// dropping ~35% guaranteed-miss probes. The `if (!cn)` guard keeps the count
// honest even if a mask were ever to over-approximate.
uint32_t appendChildren(const OctreeMap& tree,
                         uint64_t prefix, uint8_t depth, const OctreeNode& node,
                         std::vector<FlatNode>& out,
                         std::vector<FrontierItem>& queue)
{
    const uint8_t  cdep   = static_cast<uint8_t>(depth + 1);
    const uint64_t stride = 1ULL << (63 - 3 * cdep);
    uint32_t cnt = 0;
    uint8_t  m   = node.child_mask;
    while (m) {
        const unsigned oc = static_cast<unsigned>(__builtin_ctz(m));
        m &= static_cast<uint8_t>(m - 1);
        const uint64_t cpref = prefix + stride * static_cast<uint64_t>(oc);
        const OctreeNode* cn = tree.lookup(OctreeKey{cpref, cdep});
        if (!cn) continue;
        const uint32_t slot = static_cast<uint32_t>(out.size());
        out.push_back(makeFlat(cpref, cdep, *cn));
        if (flatExpandable(*cn, cdep)) queue.push_back({cpref, cn, slot, cdep});
        ++cnt;
    }
    return cnt;
}

// Breadth-first expansion of everything reachable from `queue`, writing into
// `out` (whose indices stay buffer-local). Stops early - leaving the rest of
// the queue for someone else - once the pending frontier reaches `stop_at`.
void expandBfs(const OctreeMap& tree, std::vector<FlatNode>& out,
                std::vector<FrontierItem>& queue, std::size_t& head,
                std::size_t stop_at)
{
    while (head < queue.size() && (queue.size() - head) < stop_at) {
        const FrontierItem p = queue[head++];    // by value: queue may grow
        const uint32_t off = static_cast<uint32_t>(out.size());
        const uint32_t cnt = appendChildren(tree, p.prefix, p.depth, *p.node, out, queue);
        out[p.slot].children_offset = off;
        out[p.slot].child_count     = static_cast<uint8_t>(cnt);
    }
}

}    // namespace

FlatTree flattenTree(const OctreeMap& tree)
{
    FlatTree out;
    const OctreeNode* const root = tree.lookup(OctreeKey{0ULL, 0});
    if (!root) return out;                   // absent root => empty walk

    out.reserve(tree.size() + 1);
    out.push_back(makeFlat(0ULL, 0, *root));

    std::vector<FrontierItem> q;
    if (flatExpandable(*root, 0)) q.push_back({0ULL, root, 0, 0});

    // 1. Serial backbone. Breadth-first from the root, stopping as soon as
    //    the pending frontier holds enough independent subtrees to keep every
    //    worker busy. Cheap, because it stops far above the bulk of the tree.
    const std::size_t target = std::max<std::size_t>(
        1, hpx::get_os_thread_count() * kFlattenSubtreesPerWorker);
    std::size_t head = 0;
    expandBfs(tree, out, q, head, target);

    const std::size_t backbone = out.size();
    const std::size_t S        = q.size() - head;
    if (S == 0) return out;                  // whole tree fit in the backbone

    // 2. One private buffer per remaining subtree, filled fully in parallel.
    //    No shared mutable state at all here, so no synchronisation: buffer
    //    indices are buffer-local and get rebased in step 4. dynamic_chunk_size(1)
    //    because subtree sizes are wildly uneven - a static split would leave
    //    workers idle behind the few big ones.
    std::vector<std::vector<FlatNode>> bufs(S);
    std::vector<uint32_t>              root_children(S, 0);

    hpx::experimental::for_loop(
        hpx::execution::par.with(hpx::execution::experimental::dynamic_chunk_size(1)),
        std::size_t(0), S,
        [&tree, &bufs, &root_children, &q, head](std::size_t s) {
            const FrontierItem r = q[head + s];
            std::vector<FlatNode>&    buf = bufs[s];
            std::vector<FrontierItem> lq;
            // r itself lives in the backbone; its children open this buffer.
            root_children[s] =
                appendChildren(tree, r.prefix, r.depth, *r.node, buf, lq);
            std::size_t lhead = 0;
            expandBfs(tree, buf, lq, lhead,
                       std::numeric_limits<std::size_t>::max());
        });

    // 3. Serial stitch: where each buffer lands in the final array.
    std::vector<std::size_t> base(S + 1);
    base[0] = backbone;
    for (std::size_t s = 0; s < S; ++s) base[s + 1] = base[s] + bufs[s].size();
    if (base[S] > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("flattenTree: more nodes than a uint32_t offset holds");
    out.resize(base[S]);

    // 4. Parallel copy with the offsets rebased. Destination ranges are
    //    disjoint by construction, and each task patches exactly one distinct
    //    backbone slot (its own subtree root), so this needs no locking.
    hpx::experimental::for_loop(
        hpx::execution::par.with(hpx::execution::experimental::dynamic_chunk_size(1)),
        std::size_t(0), S,
        [&out, &bufs, &base, &root_children, &q, head](std::size_t s) {
            const std::size_t b = base[s];
            FlatNode& parent = out[q[head + s].slot];
            parent.children_offset = static_cast<uint32_t>(b);
            parent.child_count     = static_cast<uint8_t>(root_children[s]);

            std::vector<FlatNode>& buf = bufs[s];
            for (std::size_t i = 0; i < buf.size(); ++i) {
                FlatNode n = buf[i];
                if (n.child_count) n.children_offset += static_cast<uint32_t>(b);
                out[b + i] = n;
            }
            std::vector<FlatNode>().swap(buf);
        });

    return out;
}


// Compare two flattened octree vectors
void compareFlattened(const std::vector<NodeRecord> A,
                      const std::vector<NodeRecord> B,
                      double tol)
{
    // sort by (depth,prefix)
    auto cmp = [](auto const &a, auto const &b) {
        if (a.depth != b.depth) return a.depth < b.depth;
        return a.prefix < b.prefix;
    };
    std::vector<NodeRecord> vA = A, vB = B;
    std::sort(vA.begin(), vA.end(), cmp);
    std::sort(vB.begin(), vB.end(), cmp);

    // walk them in lock-step
    size_t i = 0, n = std::min(vA.size(), vB.size());
    bool allGood = true;
    for (; i < n; ++i) {
        auto &a = vA[i];
        auto &b = vB[i];
        if (a.prefix != b.prefix || a.depth != b.depth) {
            std::printf("Key mismatch at index %zu:\n", i);
            std::printf("  A: (%02u,0x%016llx)\n", a.depth, (unsigned long long)a.prefix);
            std::printf("  B: (%02u,0x%016llx)\n", b.depth, (unsigned long long)b.prefix);
            allGood = false;
            continue;
        }
        if (std::abs(a.mass - b.mass) > tol
         || std::abs(a.comX - b.comX) > tol
         || std::abs(a.comY - b.comY) > tol
         || std::abs(a.comZ - b.comZ) > tol)
        {
            std::printf("Value mismatch at (d=%u,0x%016llx):\n",
                        a.depth, (unsigned long long)a.prefix);
            std::printf("  A.mass=%.9e, B.mass=%.9e\n", a.mass, b.mass);
            std::printf("  A.COM=(%.9e,%.9e,%.9e)\n",
                        a.comX, a.comY, a.comZ);
            std::printf("  B.COM=(%.9e,%.9e,%.9e)\n",
                        b.comX, b.comY, b.comZ);
            allGood = false;
        }
    }

    // check for size mismatches
    if (vA.size() != vB.size()) {
        std::printf("Size mismatch: A has %zu nodes, B has %zu nodes\n",
                    vA.size(), vB.size());
        allGood = false;
    }

    if (allGood) {
        std::cout << "✅ All " << n << " nodes match exactly!\n";
    } else {
        std::cout << "❌ Discrepancies detected.\n";
    }
}



std::vector<NodeRecord> serializeTreeToRecords(const OctreeMap& tree) {
    std::vector<NodeRecord> out;
    out.reserve(tree.size());
    for (const auto& [key, node] : tree) {
        out.push_back({key.prefix, key.depth,
                        static_cast<uint8_t>(node.is_leaf ? 1 : 0),
                        node.mass, node.comX, node.comY, node.comZ});
    }
    return out;
}

void mergeRecordsIntoTree(OctreeMap& tree, const std::vector<NodeRecord>& records) {
    // Keys that already exist in `tree`, before this merge, as genuine
    // internal (non-leaf) nodes with their own real children - protect their
    // is_leaf==false. A sender's "is_leaf" only describes whether ITS OWN
    // contribution at that key can be opened further; it says nothing about
    // whether the RECEIVER independently has real children of its own there
    // (e.g. every locality's local tree has a node at the root key {0,0}).
    // Flipping such a node to is_leaf via sticky OR would stop the walk from
    // ever opening those real local children, and since the node has no
    // body_begin/body_count range of its own (it was never actually a leaf
    // locally), the pairwise self-exclusion never runs - so a body can end up
    // treated as its own external monopole, a near-singular self-interaction.
    robin_hood::unordered_set<OctreeKey, OctreeKeyHash> protected_internal;
    for (const auto& r : records) {
        OctreeKey key{r.prefix, r.depth};
        auto it = tree.find(key);
        if (it != tree.end() && !it->second.is_leaf) protected_internal.insert(key);
    }

    for (const auto& r : records) {
        OctreeKey key = {r.prefix, r.depth};
        OctreeNode& node = tree[key]; // Creates node if it doesn't exist or finds existing one

        // A node is a leaf (must never be opened) if it was a leaf on THIS
        // locality or on the sender's - sticky OR, checked before the
        // early-out below so a zero-mass placeholder still picks up the flag.
        // Skipped for keys protected above.
        if (!protected_internal.count(key)) {
            node.is_leaf = node.is_leaf || (r.is_leaf != 0);
        }

        // Store old mass and center of mass
        double m0 = node.mass;
        double comX0 = node.comX;
        double comY0 = node.comY;
        double comZ0 = node.comZ;

        // Get mass and CoM from the record to be merged
        double m1 = r.mass;
        double comX1 = r.comX;
        double comY1 = r.comY;
        double comZ1 = r.comZ;

        double total_mass = m0 + m1;
        if (total_mass == 0) continue;

        // Update center of mass by weighted average
        node.comX = (m0 * comX0 + m1 * comX1) / total_mass;
        node.comY = (m0 * comY0 + m1 * comY1) / total_mass;
        node.comZ = (m0 * comZ0 + m1 * comZ1) / total_mass;
        node.mass = total_mass;
    }

    // A record can introduce a key this locality's own build never created,
    // so its parent's child_mask (set at build time, see buildSubtreeSerial)
    // would not mention it. Record every merged key in its parent now -
    // otherwise the mask-driven descent in flattenTree()/the walks would skip
    // a node the old 8-way blind probe still found. A key whose parent is
    // absent stays unreachable from the root either way, exactly as before.
    for (const auto& r : records) {
        if (r.depth == 0 || r.depth > BH_MAX_DEPTH) continue;
        const uint64_t shift = 63ULL - 3ULL * static_cast<uint64_t>(r.depth);
        const OctreeKey pk{mortonPrefix(r.prefix, r.depth - 1),
                            static_cast<uint8_t>(r.depth - 1)};
        if (OctreeNode* p = tree.lookup(pk))
            p->child_mask |= static_cast<uint8_t>(1u << ((r.prefix >> shift) & 7ULL));
    }
}


void printTree(const char* title, const OctreeMap& tree)
{
    // collect pointers so we can sort without copying OctreeNode
    std::vector<std::pair<OctreeKey,const OctreeNode*>> v;
    v.reserve(tree.size());
    for (const auto& kv : tree) v.push_back({kv.first,&kv.second});

    // depth primary, prefix secondary
    std::sort(v.begin(), v.end(),
              [](auto a, auto b)
              {
                  return (a.first.depth <  b.first.depth) ||
                         (a.first.depth == b.first.depth &&
                          a.first.prefix < b.first.prefix);
              });

    std::cout << '\n' << title << "\n";
    std::cout << "prefix(hex)          depth   mass       COM(x,y,z)\n";
    std::cout << "-----------------------------------------------------------\n";

    for (auto& e : v)
    {
        const OctreeKey&  k = e.first;
        const OctreeNode& n = *e.second;
        std::printf("0x%016llx  %3u   %-6.1f  (%.2e %.2e %.2e)\n",
                    static_cast<unsigned long long>(k.prefix),
                    k.depth,
                    n.mass,
                    n.comX, n.comY, n.comZ);
    }
}




OctreeMap buildTreeFromPseudoLeaves(const std::vector<NodeRecord>& remote_nodes)
{
    OctreeMap tree;
    if (remote_nodes.empty()) {
        return tree;
    }

    tree.reserve(remote_nodes.size() * 4); // conservative reservation

    // Insert and accumulate all remote nodes
    // First, insert all nodes from the input list. This correctly handles the
    // case where multiple ranks send a node with the same key, by summing
    // their mass and mass-weighted CoM contributions
    for (const auto& r : remote_nodes) {
        double m = r.mass;
        if (!(m > 0.0) || !std::isfinite(m)) continue;

        uint8_t d = r.depth > BH_MAX_DEPTH ? BH_MAX_DEPTH : r.depth;
        OctreeKey k{mortonPrefix(r.prefix, d), d};

        OctreeNode& n = tree[k];
        n.mass += m;
        n.comX += r.comX * m;
        n.comY += r.comY * m;
        n.comZ += r.comZ * m;
        n.is_pseudo = true;
        n.is_leaf = n.is_leaf || (r.is_leaf != 0);    // sticky; ancestors created below stay false
    }

    // Create all ancestor nodes
    // Now, for every node we just inserted, we ensure its parents exist by
    // walking up to the root. We only need to create the keys; the mass
    // aggregation will happen in the next step.
    OctreeMap temp_ancestors;
    for (const auto& kv : tree) {
        const OctreeKey& k = kv.first;
        for (int d = k.depth - 1; d >= 0; --d) {
            OctreeKey parent_key{mortonPrefix(k.prefix, d), (uint8_t)d};
            // The [] operator creates the node if it doesn't exist.
            temp_ancestors[parent_key];
        }
    }
    tree.insert(temp_ancestors.begin(), temp_ancestors.end());


    // Aggregate mass bottom up
    // Now that all nodes (leaves and ancestors) exist in the tree, we can
    // iterate from the deepest level upwards and sum the children's contributions
    // into their parents.
    for (int d = BH_MAX_DEPTH; d > 0; --d) {
        for (const auto& kv : tree) {
            const OctreeKey& child_key = kv.first;
            if (child_key.depth != d) continue;

            // Find the parent and add this child's contribution.
            OctreeKey parent_key = {mortonPrefix(child_key.prefix, d - 1), (uint8_t)(d - 1)};
            auto it = tree.find(parent_key);
            if (it != tree.end()) {
                OctreeNode& parent_node = it->second;
                const OctreeNode& child_node = kv.second;

                parent_node.mass += child_node.mass;
                parent_node.comX += child_node.comX; // Already mass-weighted
                parent_node.comY += child_node.comY;
                parent_node.comZ += child_node.comZ;
            }
        }
    }


    // Record every node in its parent's child_mask (see OctreeNode). Both the
    // pseudo-leaves inserted above and the ancestors created for them are
    // covered; no key is added past this point, so no mask can go stale.
    for (auto& kv : tree) {
        const OctreeKey& k = kv.first;
        if (k.depth == 0) continue;
        const uint64_t shift = 63ULL - 3ULL * static_cast<uint64_t>(k.depth);
        const OctreeKey pk{mortonPrefix(k.prefix, k.depth - 1),
                            static_cast<uint8_t>(k.depth - 1)};
        if (OctreeNode* p = tree.lookup(pk))
            p->child_mask |= static_cast<uint8_t>(1u << ((k.prefix >> shift) & 7ULL));
    }

    // After all aggregation is complete, divide by mass to get the final COM
    for (auto& kv : tree) {
        OctreeNode& n = kv.second;
        if (n.mass > 1e-12) {
            n.comX /= n.mass;
            n.comY /= n.mass;
            n.comZ /= n.mass;
        }
    }

    return tree;
}