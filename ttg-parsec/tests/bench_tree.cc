#include "../src/tree.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace bh;

namespace {

struct Options {
  std::size_t n = 10000;
  int reps = 50;
  unsigned max_leaf = 8;
};

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--n") {
      opt.n = std::stoul(next());
    } else if (a == "--reps") {
      opt.reps = std::stoi(next());
    } else if (a == "--max-leaf") {
      opt.max_leaf = static_cast<unsigned>(std::stoul(next()));
    } else {
      throw std::runtime_error("unknown argument: " + a);
    }
  }
  return opt;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  try {
    opt = parse_args(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\nusage: %s [--n N] [--reps R] [--max-leaf L]\n", e.what(),
                 argv[0]);
    return 1;
  }

  std::mt19937 rng(123);
  std::uniform_real_distribution<double> pos_dist(-10.0, 10.0);
  std::vector<Particle> base(opt.n);
  for (std::size_t i = 0; i < opt.n; ++i) {
    base[i].id = static_cast<ParticleId>(i);
    base[i].mass = 1.0;
    base[i].pos = {pos_dist(rng), pos_dist(rng), pos_dist(rng)};
  }

  auto t0 = std::chrono::steady_clock::now();
  std::size_t node_count = 0;
  for (int r = 0; r < opt.reps; ++r) {
    Tree tree(base, opt.max_leaf);
    node_count += tree.nodes().size();
  }
  auto t1 = std::chrono::steady_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::printf("n=%zu reps=%d max_leaf=%u total=%.2fms per_build=%.4fms avg_nodes=%.1f\n", opt.n,
              opt.reps, opt.max_leaf, ms, ms / opt.reps, double(node_count) / opt.reps);
  return 0;
}
