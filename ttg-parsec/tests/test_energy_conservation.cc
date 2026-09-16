#include "../src/integrate.h"
#include "test_common.h"

#include <ttg.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace bh;

int main(int argc, char** argv) {
  ttg::initialize(argc, argv, -1);
  ttg::execute();
  int rc = 0;

  const std::size_t n = 1000;
  const int steps = 30;
  const double dt = 1e-3;
  ForceParams fp;
  fp.theta = 0.5;
  fp.eps2 = 1e-4;
  fp.G = 1.0;
  fp.use_quadrupole = true;
  const unsigned max_leaf = 8;

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> pos_dist(-5.0, 5.0);
  std::uniform_real_distribution<double> mass_dist(0.5, 1.5);
  std::vector<Particle> particles(n);
  for (std::size_t i = 0; i < n; ++i) {
    particles[i].id = static_cast<ParticleId>(i);
    particles[i].mass = mass_dist(rng);
    particles[i].pos = {pos_dist(rng), pos_dist(rng), pos_dist(rng)};
    // no initial velocity: a cold, randomly-placed cloud
  }

  auto [p0, accel0] = bh_test::run_bh(particles, fp, max_leaf);
  particles = p0;
  auto accel = accel0;

  double e0 = kinetic_energy(particles) + potential_energy(particles, fp.G);
  double max_drift = 0.0;

  for (int step = 0; step < steps; ++step) {
    kick(particles, accel, dt / 2.0);
    drift(particles, dt);
    auto [p1, accel1] = bh_test::run_bh(particles, fp, max_leaf);
    particles = p1;
    accel = accel1;
    kick(particles, accel, dt / 2.0);

    double et = kinetic_energy(particles) + potential_energy(particles, fp.G);
    double drift_frac = std::abs((et - e0) / e0);
    max_drift = std::max(max_drift, drift_frac);
  }

  std::cout << "initial E_T=" << e0 << ", max relative drift over " << steps
            << " steps: " << max_drift << "\n";
  if (max_drift > 0.05) {
    std::cerr << "FAIL: energy drift exceeds 5% over " << steps << " steps\n";
    rc = 1;
  } else {
    std::cout << "PASS: energy conserved within tolerance\n";
  }

  ttg::fence();
  ttg::finalize();
  return rc;
}
