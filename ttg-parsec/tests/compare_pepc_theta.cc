// Standalone comparison probe: runs pepc-ttg's real TTG-based upward+downward
// passes (not brute force) at a given theta, for direct comparison against
// tools/parse_pepc_vtu.py's output on the same shared IC at matching PEPC
// params. Monopole-only by default to match PEPC's gravity backend, which
// doesn't implement quadrupole moments.
//
// Wired into the build via CMakeLists.txt (add_ttg_executable), since it
// needs the real TTG/PaRSEC runtime, unlike ../tools/compare_pepc.cc's
// plain brute-force probe.
#include "../src/io.h"
#include "test_common.h"

#include <ttg.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  ttg::initialize(argc, argv, -1);
  ttg::execute();

  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <csv> <theta> <eps2> <max_leaf> [--quadrupole]\n", argv[0]);
    ttg::finalize();
    return 1;
  }
  auto particles = bh::read_csv(argv[1]);
  bh::ForceParams fp;
  fp.theta = std::atof(argv[2]);
  fp.eps2 = std::atof(argv[3]);
  fp.G = 1.0;
  fp.use_quadrupole = (argc > 5 && std::string(argv[5]) == "--quadrupole");
  unsigned max_leaf = static_cast<unsigned>(std::atoi(argv[4]));

  auto [out_particles, accel] = bh_test::run_bh(particles, fp, max_leaf);

  std::vector<std::size_t> order(out_particles.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return out_particles[a].id < out_particles[b].id; });

  for (std::size_t i : order) {
    std::printf("label=%3u mass=%.6f accel=(%+.9e, %+.9e, %+.9e)\n", out_particles[i].id,
                out_particles[i].mass, accel[i].x, accel[i].y, accel[i].z);
  }

  ttg::fence();
  ttg::finalize();
  return 0;
}
