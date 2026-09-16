// Standalone comparison probe (not part of the CMake build): computes
// brute-force accelerations for a shared IC using pepc-ttg's own force
// kernel, matching PEPC's params (G=1, eps2, theta=0), for direct
// comparison against tools/parse_pepc_vtu.py's output on the same IC.
#include "../src/brute_force.h"
#include "../src/io.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <csv> <eps2>\n", argv[0]);
    return 1;
  }
  auto particles = bh::read_csv(argv[1]);
  double eps2 = std::atof(argv[2]);
  auto accel = bh::brute_force_accel(particles, eps2, /*G=*/1.0);
  for (std::size_t i = 0; i < particles.size(); ++i) {
    std::printf("label=%3u mass=%.6f accel=(%+.9e, %+.9e, %+.9e)\n", particles[i].id,
                particles[i].mass, accel[i].x, accel[i].y, accel[i].z);
  }
  return 0;
}
