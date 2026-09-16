#pragma once

#include "multipole.h"
#include "particle.h"
#include "types.h"

#include <vector>

namespace bh {

// O(N^2) reference: reuses accel_from_source with an all-zero second moment,
// which makes its quadrupole term vanish automatically, so this is exactly
// the plain pairwise-gravity kernel (whitepaper Eq. 3). Matches the
// whitepaper's observation that theta=0 degenerates Barnes-Hut into this.
inline std::vector<Vec3> brute_force_accel(const std::vector<Particle>& particles, double eps2,
                                            double G) {
  std::vector<Vec3> accel(particles.size());
  for (std::size_t i = 0; i < particles.size(); ++i) {
    Vec3 a{};
    for (std::size_t j = 0; j < particles.size(); ++j) {
      if (i == j) continue;
      a += accel_from_source(particles[i].pos, particles[j].mass, particles[j].pos, Sym3{}, eps2,
                              false, G);
    }
    accel[i] = a;
  }
  return accel;
}

}  // namespace bh
