#pragma once

#include "domain.h"
#include "key.h"
#include "multipole.h"
#include "particle.h"
#include "types.h"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace bh {

// Build-time bookkeeping only - not touched by any task's own logic (only
// read through Tree's small topology accessors, below, which are static and
// safe for any task to consult - see is_leaf()/num_children()/
// for_each_child()). particle_begin/count are read exactly once, host-side,
// at the sequential seeding boundary (see run_bh_pass in bh_pass.h) to slice
// each leaf's own particles out of the tree's particle array before they
// flow into the graph as data.
struct Node {
  Key key;
  bool leaf = false;
  std::uint32_t particle_begin = 0, particle_count = 0;  // valid iff leaf
  // has_child[idx] iff the child at octant idx actually exists (all false
  // iff leaf); the child's own Key is reconstructed on demand via
  // key.child_at(idx) rather than stored, since it's cheap arithmetic and
  // this way avoids a per-node heap allocation entirely.
  std::array<bool, static_cast<std::size_t>(Key::num_children())> has_child{};

  bool is_leaf() const { return leaf; }
};

// Every leaf's own particles, deposited by the upward pass's seed_leaf task
// as each leaf is seeded (see upward_ttg.h) and read by the downward pass
// (see traversal_ttg.h) once it starts - shared between the two passes
// rather than owned by either, so neither pass header needs to depend on
// the other. seed_leaf instances for DIFFERENT leaves can run concurrently
// on different worker threads, and all insert into the same map, so
// inserts need the mutex; reads happen only after the whole upward pass
// (every seed_leaf invocation included) has completed, so they don't.
struct LeafParticleStore {
  std::mutex mutex;
  std::unordered_map<Key, std::vector<Particle>> data;
};

// Every INTERIOR node's finalized Multipole, deposited by the upward pass's
// combine_up task as each node's aggregator fires (see upward_ttg.h) and
// read by the downward pass's dispatch task (see traversal_ttg.h) whenever
// it needs to test a source node's multipole - the same "produced once by
// one task, consumed later by however many different (target, source) pairs
// happen to reach it" shape as LeafParticleStore, and for the same reason:
// combine_up instances for DIFFERENT nodes can run concurrently, all
// inserting into the same map, so inserts need the mutex; reads happen only
// after the whole upward pass has completed, so they don't.
struct MultipoleStore {
  std::mutex mutex;
  std::unordered_map<Key, Multipole> data;
};

// Sequential (non-TTG) octree build: recursively partitions particles into
// octants until a leaf holds at most `max_particles_per_leaf`. Nodes are
// identified by their Key (level + per-axis translation, see key.h) rather
// than by a separately-assigned id, so a node's parent/child ADDRESSES are
// computed directly from its own key (Key::parent()/Key::child_at()) - the
// sparse `nodes_` map from Key to Node is what the upward and downward TTG
// passes key their tasks by and read node geometry/structure from.
class Tree {
 public:
  Tree(std::vector<Particle> particles, unsigned max_particles_per_leaf);

  std::vector<Particle>& particles() { return particles_; }
  const std::vector<Particle>& particles() const { return particles_; }
  std::size_t num_particles() const { return particles_.size(); }

  const std::unordered_map<Key, Node>& nodes() const { return nodes_; }
  Node& node(const Key& key) { return nodes_.at(key); }
  const Node& node(const Key& key) const { return nodes_.at(key); }
  Key root() const { return root_; }
  const Domain& domain() const { return domain_; }

  // A node's (center, half-width) is a pure function of its Key and the
  // domain - see box_geometry() in domain.h. No storage or lookup needed.
  std::pair<Vec3, double> box_geometry(const Key& key) const { return bh::box_geometry(domain_, key); }

  // Whether `key` is a leaf - static topology, known from the tree build and
  // never touched again, unlike a node's particles/multipole (which are
  // genuinely produced during the passes - see LeafParticleStore and
  // MultipoleStore, above). Safe for any task to read directly, same as
  // num_children()/for_each_child(), below: small, replicable, read-only by
  // the time any task runs.
  bool is_leaf(const Key& key) const { return nodes_.at(key).leaf; }

  // Number of `key`'s actual children - read from the node's own stored
  // has_child bitmap (populated once at build time - see tree.cc), rather
  // than probing nodes_ for each of the (up to) 8 candidates
  // bh::children(key) (key.h) enumerates.
  std::size_t num_children(const Key& key) const {
    const auto& has_child = nodes_.at(key).has_child;
    return static_cast<std::size_t>(std::count(has_child.begin(), has_child.end(), true));
  }

  // Invokes f(child_key) for each of `key`'s actual children, reconstructing
  // each child's Key from its octant index via child_at() rather than
  // storing it.
  template <typename F>
  void for_each_child(const Key& key, F&& f) const {
    const auto& has_child = nodes_.at(key).has_child;
    for (int idx = 0; idx < Key::num_children(); ++idx) {
      if (has_child[static_cast<std::size_t>(idx)]) f(key.child_at(idx));
    }
  }

 private:
  void build_recursive(std::uint32_t begin, std::uint32_t end, const Key& key,
                        unsigned max_particles_per_leaf, int depth);

  std::vector<Particle> particles_;
  std::vector<Particle> scratch_;  // reused across all build_recursive calls - see tree.cc
  std::unordered_map<Key, Node> nodes_;
  Key root_;
  Domain domain_;
};

}  // namespace bh
