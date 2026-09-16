#pragma once

#include "particle.h"
#include "traversal_ttg.h"
#include "tree.h"
#include "types.h"
#include "upward_ttg.h"

#include <ttg.h>

#include <vector>

namespace bh {

// Wires make_upward_graph's completion signal (UpwardGraph::done_edge)
// directly into make_downward_graph's entry point (its seed_from_upward
// task), so the two passes run as one continuous graph with a single
// trailing ttg::fence() instead of the caller fencing after the upward pass
// and re-seeding the downward pass by hand.
struct BHPass {
  UpwardGraph up;
  DownwardGraph down;
};

inline BHPass make_bh_pass(Tree& tree, const ForceParams& fp, SinkFn on_result) {
  BHPass p;
  p.up = make_upward_graph(tree);
  p.down = make_downward_graph(tree, fp, std::move(on_result), p.up.done_edge, p.up.leaf_particles,
                                p.up.multipoles);
  return p;
}

// Makes the whole fused graph executable, seeds it, and fences. Must be
// called after ttg::execute().
//
// Slicing each leaf's own particles out of tree.particles() here - once per
// leaf, at the sequential host/build boundary - is the one remaining place
// that touches the tree's particle array directly; every task downstream of
// this only ever sees particle data that arrived over an edge (see
// UpwardGraph::seed and DownwardGraph in traversal_ttg.h).
inline void run_bh_pass(Tree& tree, BHPass& p) {
  ttg::make_graph_executable(p.up.seed_leaf.get(), p.up.combine_up.get(), p.down.seed_from_upward.get(),
                              p.down.dispatch.get(), p.down.combine_down.get(), p.down.sink.get());
  if (ttg::default_execution_context().rank() == 0) {
    const auto& particles = tree.particles();
    bool root_is_leaf = tree.is_leaf(tree.root());
    for (auto& [key, node] : tree.nodes()) {
      if (!node.is_leaf()) continue;
      std::vector<Particle> leaf_particles(particles.begin() + node.particle_begin,
                                            particles.begin() + node.particle_begin + node.particle_count);
      if (root_is_leaf) {
        // Degenerate single-leaf tree: seed_leaf would return immediately
        // without ever reaching combine_up (nothing to aggregate), so
        // there's no done_edge signal to chain the downward pass off of -
        // it has to be seeded directly instead. But unlike the normal path
        // (where done_edge firing is causally downstream of every
        // seed_leaf completing, guaranteeing leaf_particles is populated
        // first), nothing here would order seed_leaf's insert into
        // leaf_particles before dispatch/sink read it - so populate it
        // synchronously ourselves (safe: this is the only leaf, so nothing
        // else can be concurrently inserting) rather than going through
        // seed_leaf at all. The sole leaf is both target and source:
        // dispatch's leaf branch reduces exactly to "sum this leaf's own
        // particles against each other."
        p.up.leaf_particles->data[key] = std::move(leaf_particles);
        p.down.seed_direct(key);
      } else {
        p.up.seed(key, std::move(leaf_particles));
      }
    }
    // In the non-degenerate case, combine_up(root) will fire once every
    // leaf seeded above has been processed, and its done_edge signal fans
    // out to all of them via seed_from_upward - see make_downward_graph.
  }
  ttg::fence();
}

}  // namespace bh
