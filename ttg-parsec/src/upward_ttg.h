#pragma once

#include "multipole.h"
#include "particle.h"
#include "tree.h"
#include "types.h"

#include <ttg.h>

#include <functional>
#include <memory>
#include <vector>

namespace bh {

// The bottom-up multipole-accumulation pass, expressed as two TTs over the
// tree's node Keys:
//   seed_leaf   - one task per leaf; receives that leaf's own particles
//                 directly as data (rather than reaching into a shared
//                 particle array via an index range) and computes that
//                 leaf's contribution as a RawMoment about its PARENT's box
//                 center (skipping any intermediate "leaf's own multipole"
//                 step - see multipole.h), forwarding it into raw_edge. It
//                 also deposits its particles into leaf_particles, below,
//                 for the downward pass to read when it needs a leaf's data
//                 for a reason other than being the fixed target of a
//                 traversal - see make_downward_graph in traversal_ttg.h.
//   combine_up  - one task per INTERIOR node; an aggregator (target =
//                 that node's actual child count, a pure function of the
//                 node table) fires once every child's RawMoment has
//                 arrived, finalizes this node's Multipole, deposits it into
//                 multipoles, below (for the downward pass to read whenever
//                 it needs a source node's multipole), and - unless this is
//                 the root - re-expresses it as a RawMoment about its OWN
//                 parent's box center and sends it up again via the very
//                 same raw_edge (self-loop, c.f. TTG's fibonacci.cc).
//                 On the ROOT specifically, it instead signals done_edge:
//                 by construction of the aggregator dependency chain, every
//                 other node's multipole is necessarily already finalized
//                 by the time the root's fires, so a downward pass wired to
//                 done_edge (see make_downward_graph in traversal_ttg.h)
//                 can start the instant this fires - no caller-side
//                 ttg::fence() between the two passes required.
//
// target_fn below is passed to ttg::make_aggregator as a named local
// variable, which requires the TTG fix decaying its TargetFn template
// parameter (see aggregator.h) - without that fix, AggregatorFactory ends
// up storing a *reference* to this function-local closure, which dangles
// the moment make_upward_graph returns.
struct UpwardGraph {
  ttg::Edge<Key, std::vector<Particle>> seed_edge;
  ttg::Edge<Key, RawMoment> raw_edge;
  ttg::Edge<Key, void> done_edge;  // fires with the root's Key once the whole pass is complete
  std::unique_ptr<ttg::TTBase> seed_leaf;
  std::unique_ptr<ttg::TTBase> combine_up;
  // Bound to seed_leaf's own (concretely-typed) input terminal at
  // construction time, since TTBase erases the type .invoke()/.in<>() need.
  std::function<void(Key, std::vector<Particle>)> seed;
  // Heap-allocated (shared_ptr-held, not a plain member) so that
  // moving/copying UpwardGraph itself (e.g. into BHPass) never invalidates
  // the references seed_leaf/combine_up's own closures captured to them.
  std::shared_ptr<LeafParticleStore> leaf_particles = std::make_shared<LeafParticleStore>();
  std::shared_ptr<MultipoleStore> multipoles = std::make_shared<MultipoleStore>();
};

inline UpwardGraph make_upward_graph(Tree& tree) {
  UpwardGraph g;
  Key root = tree.root();
  // shared_ptr copies - stable regardless of g's later moves
  auto leaf_particles = g.leaf_particles;
  auto multipoles = g.multipoles;

  auto seed_leaf_tt = ttg::make_tt(
      [&tree, leaf_particles](const Key& leaf_key, const std::vector<Particle>& particles) {
        {
          std::lock_guard<std::mutex> lock(leaf_particles->mutex);
          leaf_particles->data[leaf_key] = particles;
        }
        if (leaf_key == tree.root()) return;  // degenerate single-leaf tree: nothing to combine
        Key parent_key = leaf_key.parent();
        const Vec3& ref = tree.box_geometry(parent_key).first;
        RawMoment rm;
        for (const auto& p : particles) rm += raw_moment_of_particle(p.mass, p.pos, ref);
        ttg::send<0>(parent_key, rm);
      },
      ttg::edges(g.seed_edge), ttg::edges(g.raw_edge), "seed_leaf");

  auto target_fn = [&tree](const Key& n) -> std::size_t { return tree.num_children(n); };
  auto agg_edge = ttg::make_aggregator(g.raw_edge, target_fn);

  auto combine_up_tt = ttg::make_tt(
      [&tree, root, multipoles](const Key& key, const ttg::Aggregator<RawMoment>& agg) {
        RawMoment sum;
        for (auto&& v : agg) sum += v;
        Multipole multipole = finalize_raw_moment(sum, tree.box_geometry(key).first);
        {
          std::lock_guard<std::mutex> lock(multipoles->mutex);
          multipoles->data[key] = multipole;
        }
        if (key != root) {
          Key parent_key = key.parent();
          const Vec3& parent_ref = tree.box_geometry(parent_key).first;
          RawMoment shifted = raw_moment_from_multipole(multipole, parent_ref);
          ttg::send<0>(parent_key, shifted);
        } else {
          ttg::sendk<1>(root);
        }
      },
      ttg::edges(agg_edge), ttg::edges(g.raw_edge, g.done_edge), "combine_up");

  auto* seed_terminal = seed_leaf_tt->template in<0>();
  g.seed = [seed_terminal](Key key, std::vector<Particle> particles) {
    seed_terminal->send(key, std::move(particles));
  };

  g.seed_leaf = std::move(seed_leaf_tt);
  g.combine_up = std::move(combine_up_tt);
  return g;
}

}  // namespace bh
