#pragma once

#include "particle.h"
#include "types.h"

#include <vector>

namespace bh {

// Per-particle kick/drift, matching the whitepaper's Eqs. 6-9. main.cc
// fuses these directly into the downward pass's per-particle Sink callback
// (see integrate_ttg.h) rather than looping over a materialized vector
// after the fact; they're also used standalone by tests that materialize
// an accel vector via traversal_ttg.h's vector-based overload.
inline void kick_one(Particle& p, const Vec3& accel, double dt) { p.vel += dt * accel; }
inline void drift_one(Particle& p, double dt) { p.pos += dt * p.vel; }

inline void kick(std::vector<Particle>& particles, const std::vector<Vec3>& accel, double half_dt) {
  for (std::size_t i = 0; i < particles.size(); ++i) kick_one(particles[i], accel[i], half_dt);
}

inline void drift(std::vector<Particle>& particles, double dt) {
  for (auto& p : particles) drift_one(p, dt);
}

// Eq. 11.
inline double kinetic_energy(const std::vector<Particle>& particles) {
  double e = 0.0;
  for (const auto& p : particles) e += 0.5 * p.mass * norm2(p.vel);
  return e;
}

// Eq. 12 (O(N^2); intended for periodic verification, not every timestep
// of a large run).
inline double potential_energy(const std::vector<Particle>& particles, double G) {
  double e = 0.0;
  for (std::size_t i = 0; i < particles.size(); ++i) {
    for (std::size_t j = i + 1; j < particles.size(); ++j) {
      e -= G * particles[i].mass * particles[j].mass / norm(particles[i].pos - particles[j].pos);
    }
  }
  return e;
}

}  // namespace bh
