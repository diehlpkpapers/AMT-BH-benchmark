#pragma once

#include "key.h"
#include "multipole.h"
#include "particle.h"
#include "tree.h"
#include "types.h"

#include <ttg.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <unordered_map>
#include <vector>

namespace bh {

// Key for the downward pass: a fixed target LEAF (whose particles all need
// forces) paired with the source node currently being tested against it. A
// dedicated struct (rather than std::pair<Key,Key>) so operator<< and
// std::hash can be added via ordinary ADL/specialization - std::pair cannot
// legally gain a library-added operator<<, which TTG's trace-mode
// key-printing otherwise needs at compile time.
//
// Batching by leaf rather than by individual particle (as an earlier version
// of this pass did, keyed by (ParticleId, Key)) means every particle in a
// leaf shares the exact same traversal decisions and task count, cutting the
// number of dispatch/combine_down task instances by roughly
// max_particles_per_leaf - see mac_accept_box (multipole.h) for the
// leaf-wide MAC test that makes this safe.
struct NN {
  Key target;  // the leaf whose particles are being evaluated (fixed for the whole descent)
  Key source;  // the node currently being tested against target

  bool operator==(const NN& o) const { return target == o.target && source == o.source; }

  std::size_t hash() const {
    std::size_t h = static_cast<std::size_t>(target.hash());
    std::size_t s = static_cast<std::size_t>(source.hash());
    return h ^ (s + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
  }

  template <typename Archive>
  void serialize(Archive& ar) { ar & target & source; }
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) { ar & target & source; }
};

inline std::ostream& operator<<(std::ostream& os, const NN& k) {
  return os << "(" << k.target << "," << k.source << ")";
}

}  // namespace bh

namespace std {
template <>
struct hash<bh::NN> {
  std::size_t operator()(const bh::NN& k) const noexcept { return k.hash(); }
};
}  // namespace std

namespace bh {

// The MAC-driven downward traversal / force-evaluation pass, expressed as
// four TTs:
//   seed_from_upward - one task, fired once by the upward pass's done_edge
//                 (see make_upward_graph in upward_ttg.h) once every node's
//                 multipole is finalized; fans out one seed per LEAF - each
//                 carrying that leaf's own particles (from leaf_particles,
//                 populated by the upward pass's seed_leaf) - into
//                 descend_edge. This is the only place that translates
//                 "upward pass complete" into the downward pass's actual
//                 entry points - upward_ttg.h never needs to know about NN.
//   dispatch     - one task per (target leaf, source node) pair actually
//                 visited. Its input carries the TARGET leaf's own particles
//                 as data (attached once at the seed and forwarded along
//                 through every recursive step, since target never changes
//                 for a given descent) - so unlike an earlier version of
//                 this pass, it never reaches into a shared particle array
//                 for the target side. If source is a leaf, direct-sums
//                 every target particle against every source particle (read
//                 from leaf_particles - source, unlike target, is a
//                 different leaf each time, produced once by seed_leaf and
//                 consumed here by however many different targets happen to
//                 reach it, so it has to come from that shared, graph-
//                 populated map rather than being forwarded through descent);
//                 self-interaction is excluded by particle id, which only
//                 collides when target == source. Otherwise, reads source's
//                 multipole from multipoles (deposited by the upward pass's
//                 combine_up - the same "produced once, consumed by however
//                 many different targets reach it" shape as leaf_particles)
//                 and tests the whole target leaf against it via
//                 mac_accept_box; if it accepts, evaluates that multipole
//                 against every target particle; if not, recurses into
//                 every ACTUAL child of source (a small, statically-known
//                 set from the node table) via the self-looping
//                 descend_edge, forwarding the same target particles along.
//   combine_down - one task per (target leaf, interior source) that
//                 dispatch actually recursed into. An aggregator (target =
//                 that source's actual child count - known the instant
//                 dispatch fires, no unbounded/streaming count ever needed)
//                 sums the children's per-particle partial accelerations
//                 elementwise and forwards the combined vector up to the
//                 parent's slot (self-loop) or, at the root, to the sink.
//   sink         - one task per LEAF; invokes the caller-supplied callback
//                 once per particle in that leaf (particle data again read
//                 from leaf_particles), the instant the whole leaf's
//                 traversal is done. Different leaves finish at very
//                 different times (some resolve right at the root, others
//                 recurse many levels down many branches), so this lets
//                 particles in an already-finished leaf get integrated
//                 while other leaves are still deep in traversal - see
//                 integrate_ttg.h for that usage.
// A single-node result (leaf-leaf or MAC-accept right at the root) bypasses
// combine_down entirely and reaches the sink directly from dispatch.
struct DownwardGraph {
  std::unique_ptr<ttg::TTBase> seed_from_upward;
  std::unique_ptr<ttg::TTBase> dispatch;
  std::unique_ptr<ttg::TTBase> combine_down;
  std::unique_ptr<ttg::TTBase> sink;
  // Seeds NN{leaf, leaf} directly, bypassing seed_from_upward - only needed
  // for the degenerate single-leaf tree, where the upward pass never
  // aggregates anything so its done_edge never fires. Relies on
  // leaf_particles already holding that leaf's entry, populated
  // synchronously by run_bh_pass for this case (see bh_pass.h) rather than
  // via seed_leaf, since nothing here would otherwise order seed_leaf's
  // insert before this reads it.
  std::function<void(Key)> seed_direct;
};

struct ForceParams {
  double theta = 0.5;
  double eps2 = 0.0;
  double G = 1.0;
  bool use_quadrupole = true;
};

using SinkFn = std::function<void(const Particle&, const Vec3&)>;

inline DownwardGraph make_downward_graph(Tree& tree, const ForceParams& fp, SinkFn on_result,
                                          ttg::Edge<Key, void>& upward_done_edge,
                                          std::shared_ptr<LeafParticleStore> leaf_particles,
                                          std::shared_ptr<MultipoleStore> multipoles) {
  DownwardGraph g;
  Key root = tree.root();

  ttg::Edge<NN, std::vector<Particle>> descend_edge;
  ttg::Edge<NN, std::vector<Vec3>> contribute_edge;
  ttg::Edge<Key, std::vector<Vec3>> final_edge;

  auto dispatch_tt = ttg::make_tt(
      [&tree, fp, root, leaf_particles, multipoles](const NN& key,
                                                     const std::vector<Particle>& target_particles) {
        std::vector<Vec3> contrib(target_particles.size(), Vec3{});
        bool resolved = false;

        if (tree.is_leaf(key.source)) {
          const auto& source_particles = leaf_particles->data.at(key.source);
          for (std::size_t ti = 0; ti < target_particles.size(); ++ti) {
            const Particle& tp = target_particles[ti];
            Vec3 a{};
            for (const auto& sp : source_particles) {
              if (sp.id == tp.id) continue;  // self-interaction (only possible when target == source)
              a += accel_from_source(tp.pos, sp.mass, sp.pos, Sym3{}, fp.eps2, false, fp.G);
            }
            contrib[ti] = a;
          }
          resolved = true;
        } else {
          const Multipole& source_multipole = multipoles->data.at(key.source);
          double source_half_width = tree.box_geometry(key.source).second;
          auto [target_center, target_half_width] = tree.box_geometry(key.target);
          double edge_length = 2.0 * source_half_width;
          double target_radius = target_half_width * std::sqrt(3.0);
          if (mac_accept_box(target_center, target_radius, source_multipole.com, edge_length, fp.theta)) {
            for (std::size_t ti = 0; ti < target_particles.size(); ++ti) {
              contrib[ti] = accel_from_source(target_particles[ti].pos, source_multipole.mass,
                                               source_multipole.com, source_multipole.iab, fp.eps2,
                                               fp.use_quadrupole, fp.G);
            }
            resolved = true;
          }
        }

        if (resolved) {
          if (key.source == root) {
            ttg::send<2>(key.target, std::move(contrib));
          } else {
            ttg::send<1>(NN{key.target, key.source.parent()}, std::move(contrib));
          }
        } else {
          tree.for_each_child(key.source, [&](const Key& child) {
            ttg::send<0>(NN{key.target, child}, target_particles);
          });
        }
      },
      ttg::edges(descend_edge), ttg::edges(descend_edge, contribute_edge, final_edge), "dispatch");

  auto target_fn = [&tree](const NN& key) -> std::size_t { return tree.num_children(key.source); };
  auto agg_edge = ttg::make_aggregator(contribute_edge, target_fn);

  auto combine_down_tt = ttg::make_tt(
      [root](const NN& key, const ttg::Aggregator<std::vector<Vec3>>& agg) {
        std::vector<Vec3> sum;
        for (auto&& v : agg) {
          if (sum.empty()) sum.assign(v.size(), Vec3{});
          for (std::size_t i = 0; i < sum.size(); ++i) sum[i] += v[i];
        }
        if (key.source == root) {
          ttg::send<1>(key.target, sum);
        } else {
          ttg::send<0>(NN{key.target, key.source.parent()}, sum);
        }
      },
      ttg::edges(agg_edge), ttg::edges(contribute_edge, final_edge), "combine_down");

  auto sink_tt = ttg::make_tt(
      [leaf_particles, on_result = std::move(on_result)](const Key& leaf_key,
                                                          const std::vector<Vec3>& accel) {
        const auto& particles = leaf_particles->data.at(leaf_key);
        for (std::size_t i = 0; i < particles.size(); ++i) on_result(particles[i], accel[i]);
      },
      ttg::edges(final_edge), ttg::edges(), "sink");

  auto seed_from_upward_tt = ttg::make_tt(
      [leaf_particles, root](const Key&) {
        for (auto& [leaf_key, particles] : leaf_particles->data) {
          ttg::send<0>(NN{leaf_key, root}, particles);
        }
      },
      ttg::edges(upward_done_edge), ttg::edges(descend_edge), "seed_from_upward");

  auto* descend_terminal = dispatch_tt->template in<0>();
  g.seed_direct = [descend_terminal, leaf_particles](Key leaf) {
    descend_terminal->send(NN{leaf, leaf}, leaf_particles->data.at(leaf));
  };

  g.seed_from_upward = std::move(seed_from_upward_tt);
  g.dispatch = std::move(dispatch_tt);
  g.combine_down = std::move(combine_down_tt);
  g.sink = std::move(sink_tt);
  return g;
}

}  // namespace bh
