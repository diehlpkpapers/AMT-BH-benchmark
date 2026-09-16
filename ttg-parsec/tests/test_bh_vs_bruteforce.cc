#include "../src/brute_force.h"
#include "test_common.h"

#include <ttg.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace bh;

namespace {

std::vector<Particle> random_particles(std::size_t n, unsigned seed, double spread,
                                        Vec3 center = {}) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> pos_dist(-spread, spread);
  std::uniform_real_distribution<double> mass_dist(0.5, 1.5);
  std::vector<Particle> particles(n);
  for (std::size_t i = 0; i < n; ++i) {
    particles[i].id = static_cast<ParticleId>(i);
    particles[i].mass = mass_dist(rng);
    particles[i].pos = {center.x + pos_dist(rng), center.y + pos_dist(rng), center.z + pos_dist(rng)};
  }
  return particles;
}

// Global-normalized error metric: sqrt(sum|a_test-a_ref|^2) / sqrt(sum|a_ref|^2).
// Avoids the blow-up a per-particle relative error would have for particles
// whose true net force happens to be near zero.
double relative_error(const std::vector<Vec3>& test, const std::vector<Vec3>& ref) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    num += norm2(test[i] - ref[i]);
    den += norm2(ref[i]);
  }
  return std::sqrt(num / den);
}

}  // namespace

int main(int argc, char** argv) {
  ttg::initialize(argc, argv, -1);
  ttg::execute();
  int rc = 0;

  {
    // theta=0 must degenerate Barnes-Hut into brute force (whitepaper's own
    // stated property of Eq. 10), so the tree code should reproduce the
    // O(N^2) reference to floating-point summation-order tolerance.
    auto particles = random_particles(200, /*seed=*/1, /*spread=*/10.0);
    ForceParams fp;
    fp.theta = 0.0;
    fp.eps2 = 1e-6;
    fp.G = 1.0;
    fp.use_quadrupole = true;

    auto [bh_particles, bh_accel] = bh_test::run_bh(particles, fp, /*max_leaf=*/4);
    auto ref_accel = brute_force_accel(bh_particles, fp.eps2, fp.G);

    double err = relative_error(bh_accel, ref_accel);
    std::cout << "theta=0 relative error: " << err << "\n";
    if (err > 1e-8) {
      std::cerr << "FAIL: theta=0 should reproduce brute force to floating-point tolerance\n";
      rc = 1;
    } else {
      std::cout << "PASS: theta=0 matches brute force\n";
    }
  }

  {
    // Two tight, well-separated clusters at moderate theta: each cluster's
    // internal spread gives it a nontrivial quadrupole moment, so adding
    // the quadrupole term should measurably reduce the force error versus
    // a monopole-only approximation - the concrete check that the
    // parallel-axis-shift/quadrupole formula's sign and factor are correct.
    auto cluster_a = random_particles(100, /*seed=*/2, /*spread=*/1.0, Vec3{-50.0, 0.0, 0.0});
    auto cluster_b = random_particles(100, /*seed=*/3, /*spread=*/1.0, Vec3{50.0, 0.0, 0.0});
    std::vector<Particle> particles = cluster_a;
    particles.insert(particles.end(), cluster_b.begin(), cluster_b.end());
    for (std::size_t i = 0; i < particles.size(); ++i) particles[i].id = static_cast<ParticleId>(i);

    ForceParams fp;
    fp.theta = 0.6;
    fp.eps2 = 1e-6;
    fp.G = 1.0;

    fp.use_quadrupole = false;
    auto [mono_particles, mono_accel] = bh_test::run_bh(particles, fp, /*max_leaf=*/4);
    auto ref_accel_mono = brute_force_accel(mono_particles, fp.eps2, fp.G);
    double mono_err = relative_error(mono_accel, ref_accel_mono);

    fp.use_quadrupole = true;
    auto [quad_particles, quad_accel] = bh_test::run_bh(particles, fp, /*max_leaf=*/4);
    auto ref_accel_quad = brute_force_accel(quad_particles, fp.eps2, fp.G);
    double quad_err = relative_error(quad_accel, ref_accel_quad);

    std::cout << "monopole-only relative error: " << mono_err << "\n";
    std::cout << "monopole+quadrupole relative error: " << quad_err << "\n";
    if (!(quad_err < mono_err)) {
      std::cerr << "FAIL: quadrupole should reduce force error versus monopole-only\n";
      rc = 1;
    } else {
      std::cout << "PASS: quadrupole term reduces force error versus monopole-only\n";
    }
  }

  ttg::fence();
  ttg::finalize();
  return rc;
}
