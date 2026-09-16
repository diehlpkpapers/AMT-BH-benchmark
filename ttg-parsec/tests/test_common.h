#pragma once

#include "../src/bh_pass.h"
#include "../src/tree.h"

#include <ttg.h>

#include <utility>
#include <vector>

namespace bh_test {

// Runs the fused upward+downward pass once and returns {particles, matching
// acceleration vector}, both indexed by each particle's own id (the sink
// delivers particles by value as they flowed through the graph, not by an
// index into tree.particles() - see traversal_ttg.h) - mirrors main.cc's
// compute_accel helper, but as a free function here since main.cc's is
// file-local.
inline std::pair<std::vector<bh::Particle>, std::vector<bh::Vec3>> run_bh(
    std::vector<bh::Particle> particles, const bh::ForceParams& fp, unsigned max_leaf) {
  bh::Tree tree(std::move(particles), max_leaf);

  std::vector<bh::Particle> out_particles(tree.num_particles());
  std::vector<bh::Vec3> accel(tree.num_particles());
  auto pass = bh::make_bh_pass(tree, fp, [&](const bh::Particle& p, const bh::Vec3& a) {
    out_particles[p.id] = p;
    accel[p.id] = a;
  });
  bh::run_bh_pass(tree, pass);

  return {std::move(out_particles), std::move(accel)};
}

}  // namespace bh_test
