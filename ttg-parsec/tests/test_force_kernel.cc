#include "../src/multipole.h"

#include <cmath>
#include <iostream>

// Analytic two-body check: no tree, no softening - accel_from_source with
// an all-zero second moment should reduce exactly to G*m/d^2 pointing from
// target toward source.
int main() {
  using namespace bh;

  Vec3 target{0.0, 0.0, 0.0};
  Vec3 source_com{4.0, 0.0, 0.0};
  double source_mass = 3.0;
  double G = 2.0;

  Vec3 a = accel_from_source(target, source_mass, source_com, Sym3{}, 0.0, false, G);
  double expected_mag = G * source_mass / (4.0 * 4.0);

  double err = std::abs(a.x - expected_mag) + std::abs(a.y) + std::abs(a.z);
  if (err > 1e-12) {
    std::cerr << "FAIL: expected a=(" << expected_mag << ",0,0), got (" << a.x << "," << a.y << ","
               << a.z << ")\n";
    return 1;
  }

  // Softening should reduce the magnitude relative to the unsoftened case.
  Vec3 a_soft = accel_from_source(target, source_mass, source_com, Sym3{}, 4.0, false, G);
  if (!(norm(a_soft) < norm(a))) {
    std::cerr << "FAIL: softened force should be weaker than unsoftened\n";
    return 1;
  }

  // All-zero second moment must make the quadrupole term vanish exactly,
  // regardless of use_quadrupole - this is what lets brute_force.h and
  // leaf-leaf direct sums reuse this same kernel.
  Vec3 a_quad_off = accel_from_source(target, source_mass, source_com, Sym3{}, 0.0, false, G);
  Vec3 a_quad_on = accel_from_source(target, source_mass, source_com, Sym3{}, 0.0, true, G);
  if (std::abs(a_quad_off.x - a_quad_on.x) > 1e-12 || std::abs(a_quad_off.y - a_quad_on.y) > 1e-12 ||
      std::abs(a_quad_off.z - a_quad_on.z) > 1e-12) {
    std::cerr << "FAIL: zero second moment should give zero quadrupole contribution\n";
    return 1;
  }

  std::cout << "PASS: force kernel matches analytic two-body result\n";
  return 0;
}
