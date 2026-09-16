#pragma once

#include "bh_pass.h"
#include "integrate.h"
#include "tree.h"
#include "types.h"

#include <chrono>
#include <vector>

namespace bh {

// Wall-clock breakdown of one step_particles() call, in milliseconds.
// Populated only when a non-null StepTiming* is passed in. The upward and
// downward passes no longer have a separate timing split: they're wired
// together via an edge (see bh_pass.h) rather than the caller fencing
// between them, so there's no synchronization point left to measure a
// boundary at - traversal_ms covers both.
struct StepTiming {
  double tree_build_ms = 0.0;
  double traversal_ms = 0.0;
};

// Builds a fresh octree from `particles`, runs the fused upward
// (multipole) + downward (MAC-driven force) pass, and integrates each
// particle's velocity (by `kick_dt`) and, if `drift_dt != 0`, its position
// (by `drift_dt`) - AS PART OF the downward pass's per-leaf Sink callback,
// the instant that leaf's particles' accelerations are known, rather than
// materializing every particle's acceleration into a vector and looping
// over it in a separate host-side pass afterward. A leaf whose traversal
// finishes early starts integrating immediately, overlapping with leaves
// still deep in traversal.
//
// The sink callback receives each particle by value (as it flowed through
// the graph - see DownwardGraph in traversal_ttg.h) rather than an index
// into a shared array, so results are collected into a fresh vector here
// (keyed by the particle's own, tree-build-independent id) instead of
// mutating tree.particles() in place.
//
// kick_dt/drift_dt are independent (not both "the timestep") because the
// standard optimized leapfrog sequence - one half-kick+drift to prime,
// N-1 full-kick+drift steps, one closing half-kick with no drift - reuses
// each force evaluation for both the trailing half-kick of the previous
// step and the leading half-kick of the next (see main.cc).
inline std::vector<Particle> step_particles(std::vector<Particle> particles, const ForceParams& fp,
                                             unsigned max_leaf, double kick_dt, double drift_dt,
                                             StepTiming* timing = nullptr) {
  using clock = std::chrono::steady_clock;
  auto elapsed_ms = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  auto t0 = clock::now();
  Tree tree(std::move(particles), max_leaf);
  auto t1 = clock::now();

  std::vector<Particle> updated(tree.num_particles());
  auto pass = make_bh_pass(tree, fp, [&updated, kick_dt, drift_dt](const Particle& p, const Vec3& a) {
    Particle np = p;
    kick_one(np, a, kick_dt);
    if (drift_dt != 0.0) drift_one(np, drift_dt);
    updated[np.id] = np;
  });
  run_bh_pass(tree, pass);
  auto t2 = clock::now();

  if (timing != nullptr) {
    timing->tree_build_ms = elapsed_ms(t0, t1);
    timing->traversal_ms = elapsed_ms(t1, t2);
  }

  return updated;
}

}  // namespace bh
