#pragma once
// Shared code between barneshut.cpp (sequential reference) and
// barneshut_ityr.cpp (parallel version). Physics, tree building, I/O, config
// and timing live here so both programs produce the same numbers.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std;
using Clock = chrono::steady_clock;
using TimePoint = chrono::time_point<Clock>;

constexpr int NLEVELS = 20;
constexpr uint64_t COORD_MAX = (1ull << 21) - 1;

struct Body {
  long long id;
  double mass, x, y, z, vx, vy, vz, ax = 0, ay = 0, az = 0, pot = 0;
};

// Octree node. The fields the walk touches on every visit (mass, center,
// child count, leaf flag) come first; child[] is last. Children are stored
// packed in child[0..nchild-1] in increasing octant order, and child indices
// are relative to the node's own index, so a subtree can be relocated with a
// plain copy.
struct Node {
  double mass = 0, cx = 0, cy = 0, cz = 0;
  int first, count;
  int nchild;
  bool leaf;
  int child[8];
};

// Full option set; the sequential binary parses but ignores the parallel-only
// knobs (--chunk, --build-cutoff).
struct Config {
  string input;
  string prefix = "snapshot";
  int steps = 100, output_every = 10, energy_every = 1, leaf = 8;
  size_t chunk = 1024, build_cutoff = 32768;
  double dt = 0.01, theta = 0.5, eps2 = 1e-6, G = 1.0;
  bool output = true, timers = true, brute_force = false;
  // PEPC namelist drop-in mode (parse_namelist sets these).
  bool binary_input = false;
  size_t tnp = 0;
};

inline Config cfg;

using KeyPair = std::pair<uint64_t, int>;

// Position-only projection of a body (the walk reads exactly these fields from
// leaf bodies). The parallel version replicates this per node instead of the
// full Body.
struct CompactBody {
  double mass, x, y, z;
};

// Walk constants; copied into task frames by value in the parallel version.
// s2[l] = (box_side / 2^l)^2, the squared cell size at level l, precomputed so
// the MAC test is a load + compare.
struct WalkParams {
  double box_side, xmin, ymin, zmin, theta2, G, eps2;
  double s2[NLEVELS + 1];
};

inline WalkParams make_walk_params(double box_side, double xmin, double ymin, double zmin,
                                   double theta2, double G, double eps2) {
  WalkParams wp{box_side, xmin, ymin, zmin, theta2, G, eps2, {}};
  for (int l = 0; l <= NLEVELS; ++l) {
    double s = box_side / (double)(1 << l);
    wp.s2[l] = s * s;
  }
  return wp;
}

double ms_since(TimePoint t) {
  return chrono::duration<double, milli>(Clock::now() - t).count();
}

struct Timer {
  double ms = 0;
  void add(TimePoint t) {
    if (cfg.timers) ms += ms_since(t);
  }
};

inline Timer t_input, t_build, t_forces, t_integrate, t_output, t_collective;

// Category view of t_collective (same waits, bucketed per callsite group) so the
// master's collective-wait time can be attributed without touching every phase.
inline Timer tc_dist, tc_bar, tc_energy, tc_ptr, tc_misc;

// Wall time of a blocking MPI collective (Allreduce/Bcast/Barrier): this is
// the time the rank is stuck waiting in the collective, i.e. the "does nothing
// but wait" cost. Accumulated into t_collective.
template <typename Fn>
auto collective_timed(Fn fn) {
  TimePoint t = Clock::now();
  if constexpr (std::is_void_v<std::invoke_result_t<Fn>>) {
    fn();
    if (cfg.timers) t_collective.ms += ms_since(t);
  } else {
    auto r = fn();
    if (cfg.timers) t_collective.ms += ms_since(t);
    return r;
  }
}

// Same, but also buckets the wait into a category timer (tc_*).
template <typename Fn>
auto collective_timed_cat(Timer& cat, Fn fn) {
  TimePoint t = Clock::now();
  if constexpr (std::is_void_v<std::invoke_result_t<Fn>>) {
    fn();
    if (cfg.timers) { double d = ms_since(t); t_collective.ms += d; cat.ms += d; }
  } else {
    auto r = fn();
    if (cfg.timers) { double d = ms_since(t); t_collective.ms += d; cat.ms += d; }
    return r;
  }
}

// ---------------- geometry / tree helpers ----------------

inline uint64_t coord_to_int(double v, double lo, double inv) {
  long long i = (long long)((v - lo) * inv * (double)COORD_MAX);
  if (i < 0) i = 0;
  if (i > (long long)COORD_MAX) i = (long long)COORD_MAX;
  return (uint64_t)i;
}

// Spread the low 21 bits of v so that bit j lands at bit 3j (the standard
// magic-number Morton dilation).
inline uint64_t spread_bits_3(uint64_t v) {
  v &= 0x1fffffull;
  v = (v | (v << 32)) & 0x1f00000000ffffull;
  v = (v | (v << 16)) & 0x1f0000ff0000ffull;
  v = (v | (v << 8))  & 0x100f00f00f00f00full;
  v = (v | (v << 4))  & 0x10c30c30c30c30c3ull;
  v = (v | (v << 2))  & 0x1249249249249249ull;
  return v;
}

inline uint64_t make_key(double x, double y, double z, double xmin, double ymin, double zmin,
                  double inv) {
  uint64_t ix = coord_to_int(x, xmin, inv);
  uint64_t iy = coord_to_int(y, ymin, inv);
  uint64_t iz = coord_to_int(z, zmin, inv);
  return spread_bits_3(ix) | (spread_bits_3(iy) << 1) | (spread_bits_3(iz) << 2) |
         (1ull << 63);
}

inline int octant(uint64_t key, int shift) { return (int)((key >> shift) & 7); }

// End of the run of equal octant digits starting at `from`, within the sorted
// range [from,to). The keys are Morton-sorted so the digit is non-decreasing;
// a binary search finds the boundary, with a linear scan for short runs.
inline size_t octant_run_end(const KeyPair* keys, size_t from, size_t to, int shift, int c) {
  if (to - from <= 64) {
    size_t j = from;
    while (j < to && octant(keys[j].first, shift) == c) ++j;
    return j;
  }
  return (size_t)(std::upper_bound(keys + from, keys + to, c,
                                   [shift](int cv, const KeyPair& kp) {
                                     return cv < octant(kp.first, shift);
                                   }) -
                  keys);
}

// Overload for the raw-key arrays the parallel version keeps.
inline size_t octant_run_end(const uint64_t* keys, size_t from, size_t to, int shift, int c) {
  if (to - from <= 64) {
    size_t j = from;
    while (j < to && octant(keys[j], shift) == c) ++j;
    return j;
  }
  return (size_t)(std::upper_bound(keys + from, keys + to, c,
                                   [shift](int cv, uint64_t k) {
                                     return cv < octant(k, shift);
                                   }) -
                  keys);
}

inline void add_to_com(Node& n, double mass, double x, double y, double z) {
  n.mass += mass;
  n.cx += mass * x;
  n.cy += mass * y;
  n.cz += mass * z;
}

inline void finalize_com(Node& n) {
  if (n.mass > 0) {
    n.cx /= n.mass;
    n.cy /= n.mass;
    n.cz /= n.mass;
  }
}

// ---------------- force kernel ----------------

inline void interact(double ox, double oy, double oz, double om, double mi, double x, double y,
              double z, const WalkParams& wp, double& ax, double& ay, double& az,
              double& pot) {
  double dx = ox - x, dy = oy - y, dz = oz - z;
  double r2 = dx * dx + dy * dy + dz * dz + wp.eps2;
  // ir = 1/sqrt(r2), inv = ir^3 = 1/r2^(3/2): one reciprocal sqrt per pair.
  double ir = 1.0 / sqrt(r2);
  double inv = ir * ir * ir;
  ax += wp.G * om * dx * inv;
  ay += wp.G * om * dy * inv;
  az += wp.G * om * dz * inv;
  pot -= wp.G * mi * om * ir;
}

// `level` is carried down the recursion instead of being stored per node (see Node).
inline void walk(int idx, int level, const Body& b, const Node* nodes, const Body* bodies,
          const WalkParams& wp, double& ax, double& ay, double& az, double& pot) {
  const Node& n = nodes[idx];
  if (n.leaf) {
    for (int i = n.first; i < n.first + n.count; ++i) {
      const Body& o = bodies[i];
      if (&o == &b) continue;
      interact(o.x, o.y, o.z, o.mass, b.mass, b.x, b.y, b.z, wp, ax, ay, az, pot);
    }
    return;
  }
  double dx = n.cx - b.x, dy = n.cy - b.y, dz = n.cz - b.z;
  double d2 = dx * dx + dy * dy + dz * dz;
  if (wp.s2[level] < wp.theta2 * d2) {
    interact(n.cx, n.cy, n.cz, n.mass, b.mass, b.x, b.y, b.z, wp, ax, ay, az, pot);
    return;
  }
  for (int k = 0; k < n.nchild; ++k)
    walk(idx + n.child[k], level + 1, b, nodes, bodies, wp, ax, ay, az, pot);
}

// Same traversal over the position-only projection (see CompactBody).
inline void walk_compact(int idx, int level, const CompactBody& b, const Node* nodes,
                 const CompactBody* bodies, const WalkParams& wp, double& ax, double& ay,
                 double& az, double& pot) {
  const Node& n = nodes[idx];
  if (n.leaf) {
    for (int i = n.first; i < n.first + n.count; ++i) {
      const CompactBody& o = bodies[i];
      if (&o == &b) continue;
      interact(o.x, o.y, o.z, o.mass, b.mass, b.x, b.y, b.z, wp, ax, ay, az, pot);
    }
    return;
  }
  double dx = n.cx - b.x, dy = n.cy - b.y, dz = n.cz - b.z;
  double d2 = dx * dx + dy * dy + dz * dz;
  if (wp.s2[level] < wp.theta2 * d2) {
    interact(n.cx, n.cy, n.cz, n.mass, b.mass, b.x, b.y, b.z, wp, ax, ay, az, pot);
    return;
  }
  for (int k = 0; k < n.nchild; ++k)
    walk_compact(idx + n.child[k], level + 1, b, nodes, bodies, wp, ax, ay, az, pot);
}

// Axis-aligned bounds of the bodies mapped to the unit cube for key computation.
struct Bounds { double xmin, ymin, zmin, inv, box_side; };

inline Bounds compute_bounds(const Body* b, size_t n) {
  double xmin = b[0].x, xmax = b[0].x;
  double ymin = b[0].y, ymax = b[0].y;
  double zmin = b[0].z, zmax = b[0].z;
  for (size_t i = 1; i < n; ++i) {
    xmin = min(xmin, b[i].x);
    xmax = max(xmax, b[i].x);
    ymin = min(ymin, b[i].y);
    ymax = max(ymax, b[i].y);
    zmin = min(zmin, b[i].z);
    zmax = max(zmax, b[i].z);
  }
  double box_side = max({xmax - xmin, ymax - ymin, zmax - zmin});
  if (box_side <= 0) box_side = 1.0;
  return {xmin, ymin, zmin, 1.0 / box_side, box_side};
}

// ---------------- tree building ----------------

// Octree for the sorted key range [lo,hi) at `level`, preorder (root at index
// 0). Nodes are appended to vec; child indices are relative to the owning
// node's position (see Node) and packed in increasing octant order.
inline void build_rec(const KeyPair* keys, const Body* bodies, size_t lo, size_t hi,
               int level, int leaf, vector<Node>& vec) {
  size_t own = vec.size();
  vec.push_back(Node{});
  Node& n = vec[own];
  n.first = (int)lo;
  n.count = (int)(hi - lo);
  n.nchild = 0;
  if (hi - lo <= (size_t)leaf || level >= NLEVELS) {
    n.leaf = true;
    for (size_t i = lo; i < hi; ++i)
      add_to_com(n, bodies[i].mass, bodies[i].x, bodies[i].y, bodies[i].z);
    finalize_com(n);
    return;
  }
  n.leaf = false;
  int shift = 3 * (NLEVELS - level);
  size_t i = lo;
  while (i < hi) {
    int c = octant(keys[i].first, shift);
    size_t j = octant_run_end(keys, i + 1, hi, shift, c);
    int slot = vec[own].nchild++;
    vec[own].child[slot] = (int)(vec.size() - own);
    build_rec(keys, bodies, i, j, level + 1, leaf, vec);
    i = j;
  }
  Node& m = vec[own];
  for (int k = 0; k < m.nchild; ++k) {
    const Node& ch = vec[own + m.child[k]];
    add_to_com(m, ch.mass, ch.cx, ch.cy, ch.cz);
  }
  finalize_com(m);
}

// ---------------- energy / verification ----------------

// Deterministic blocked summation. A plain left-to-right serial sum cannot be
// reproduced by a parallel decomposition (the association differs at every
// chunk boundary), so both binaries define the sum as:
//   * contiguous blocks of ENERGY_BLOCK elements, each summed serially in
//     index order (reproducible on any rank);
//   * block sums combined in a fixed binary tree over the block list
//     (tree_combine: t(lo,hi) = t(lo,mid) + t(mid,hi), mid = lo + (hi-lo)/2).
// The sequential and parallel programs therefore produce identical results.
constexpr size_t ENERGY_BLOCK = 16384;

inline double tree_combine(const double* s, size_t lo, size_t hi) {
  if (hi - lo == 1) return s[lo];
  size_t mid = lo + (hi - lo) / 2;
  return tree_combine(s, lo, mid) + tree_combine(s, mid, hi);
}

inline double blocked_sum(const double* v, size_t n) {
  size_t nblocks = (n + ENERGY_BLOCK - 1) / ENERGY_BLOCK;
  if (nblocks == 1) {   // fast path: plain serial sum (identical either way)
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += v[i];
    return s;
  }
  vector<double> s(nblocks);
  for (size_t k = 0; k < nblocks; ++k) {
    size_t a = k * ENERGY_BLOCK, b = std::min(a + ENERGY_BLOCK, n);
    double acc = 0;
    for (size_t i = a; i < b; ++i) acc += v[i];
    s[k] = acc;
  }
  return tree_combine(s.data(), 0, nblocks);
}

// Kinetic energy summed with blocked_sum. Templated so both the full Body and
// the Body64 variants use the same definition.
template <typename B>
inline double kinetic_energy(const B* b, size_t n) {
  vector<double> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = 0.5 * b[i].mass * (b[i].vx * b[i].vx + b[i].vy * b[i].vy + b[i].vz * b[i].vz);
  return blocked_sum(v.data(), n);
}

// Compare the walk results already stored in b[].ax/ay/az/pot against direct O(N^2);
// ep_walk is the walk's total potential (0.5 * sum of per-body pot). Prints + returns 0/1.
inline int brute_force_check(const Body* b, size_t n, double ep_walk) {
  vector<double> ax(n), ay(n), az(n);
  for (size_t i = 0; i < n; ++i) {
    ax[i] = b[i].ax;
    ay[i] = b[i].ay;
    az[i] = b[i].az;
  }
  vector<double> bx(n), by(n), bz(n);
  double ep_bf = 0;
  for (size_t i = 0; i < n; ++i) {
    const Body& bi = b[i];
    for (size_t j = i + 1; j < n; ++j) {
      const Body& bj = b[j];
      double dx = bj.x - bi.x, dy = bj.y - bi.y, dz = bj.z - bi.z;
      double r2 = dx * dx + dy * dy + dz * dz + cfg.eps2;
      double inv = cfg.G / (r2 * sqrt(r2));
      bx[i] += inv * bj.mass * dx;
      by[i] += inv * bj.mass * dy;
      bz[i] += inv * bj.mass * dz;
      bx[j] -= inv * bi.mass * dx;
      by[j] -= inv * bi.mass * dy;
      bz[j] -= inv * bi.mass * dz;
      ep_bf -= cfg.G * bi.mass * bj.mass / sqrt(r2);
    }
  }
  double max_abs = 0;
  for (size_t i = 0; i < n; ++i)
    max_abs = max({max_abs, fabs(bx[i]), fabs(by[i]), fabs(bz[i])});
  double max_rel = 0, ep_rel = 0;
  for (size_t i = 0; i < n; ++i) {
    double da = max({fabs(ax[i] - bx[i]), fabs(ay[i] - by[i]), fabs(az[i] - bz[i])});
    max_rel = max(max_rel, da / max(max_abs, 1e-300));
  }
  ep_rel = fabs(ep_walk - ep_bf) / max(fabs(ep_bf), 1e-300);
  printf("brute-force check: N=%d  max|a|=%.6e  max_rel_acc=%.3e  rel_EP=%.3e\n", (int)n,
         max_abs, max_rel, ep_rel);
  return max_rel < 1e-10 && ep_rel < 1e-10 ? 0 : 1;
}

// ---------------- input / output / options ----------------

inline string strim(const string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

template <typename B>
inline bool read_bodies_t(const string& path, vector<B>& bodies) {
  ifstream in(path);
  if (!in) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    return false;
  }
  string line;
  while (getline(in, line)) {
    if (line.empty() || line[0] == 'i' || line[0] == 'I') continue;
    replace(line.begin(), line.end(), ',', ' ');
    istringstream ss(line);
    long long id;
    double m, x, y, z, vx, vy, vz;
    if (!(ss >> id >> m >> x >> y >> z >> vx >> vy >> vz)) continue;
    bodies.push_back({id, m, x, y, z, vx, vy, vz});
  }
  if (bodies.empty()) {
    fprintf(stderr, "no bodies in %s\n", path.c_str());
    return false;
  }
  return true;
}

// PEPC binary input: no header, 7 doubles per particle (x,y,z,vx,vy,vz,mass).
template <typename B>
inline bool read_pepcbin_t(const string& path, vector<B>& bodies) {
  ifstream in(path, ios::binary);
  if (!in) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    return false;
  }
  double d[7];
  long long id = 0;
  while (cfg.tnp == 0 || (size_t)id < cfg.tnp) {
    in.read((char*)d, sizeof(d));
    if (!in) break;
    bodies.push_back({id++, d[6], d[0], d[1], d[2], d[3], d[4], d[5]});
  }
  if (bodies.empty()) {
    fprintf(stderr, "no bodies in %s\n", path.c_str());
    return false;
  }
  if (cfg.tnp > 0 && (size_t)id != cfg.tnp)
    fprintf(stderr, "warning: %s has %lld particles, tnp=%zu\n", path.c_str(), id, cfg.tnp);
  return true;
}

template <typename B>
inline bool read_input_t(const string& path, vector<B>& bodies) {
  return cfg.binary_input ? read_pepcbin_t(path, bodies) : read_bodies_t(path, bodies);
}

// The sequential version keeps the full Body (it stores acc into it); the
// parallel version uses the 64-byte Body64.
inline bool read_input(const string& path, vector<Body>& bodies) {
  return read_input_t(path, bodies);
}

// Legacy ASCII VTK snapshot (17-digit precision — default 6 breaks comparisons).
inline void write_vtk(const Body* b, size_t n, int step) {
  char path[512];
  snprintf(path, sizeof(path), "%s_%05d.vtk", cfg.prefix.c_str(), step);
  ofstream f(path);
  f.precision(17);
  f << "# vtk DataFile Version 3.0\nbarneshut snapshot " << step
    << "\nASCII\nDATASET POLYDATA\nPOINTS " << n << " double\n";
  for (size_t i = 0; i < n; ++i) f << b[i].x << " " << b[i].y << " " << b[i].z << "\n";
  f << "POINT_DATA " << n << "\nSCALARS id long\nLOOKUP_TABLE default\n";
  for (size_t i = 0; i < n; ++i) f << b[i].id << "\n";
  f << "SCALARS mass double 1\nLOOKUP_TABLE default\n";
  for (size_t i = 0; i < n; ++i) f << b[i].mass << "\n";
  f << "SCALARS potential double 1\nLOOKUP_TABLE default\n";
  for (size_t i = 0; i < n; ++i) f << b[i].pot << "\n";
  f << "VECTORS velocity double\n";
  for (size_t i = 0; i < n; ++i) f << b[i].vx << " " << b[i].vy << " " << b[i].vz << "\n";
  f << "VECTORS acceleration double\n";
  for (size_t i = 0; i < n; ++i) f << b[i].ax << " " << b[i].ay << " " << b[i].az << "\n";
}

inline void report_energy(int step, double ek, double ep, double et0) {
  double et = ek + ep;
  printf("%8d %18.8e %18.8e %18.8e %+12.3e\n", step, ek, ep, et,
         (et - et0) / max(fabs(et0), 1e-300));
}

inline void usage() {
  printf("Usage: barneshut --input FILE [options]\n"
         "  PARAMS_FILE        PEPC-style namelist params file (drop-in; overrides below)\n"
         "  --input FILE       CSV input (id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z)\n"
         "  --steps N          integration steps (default 100)\n"
         "  --dt T             timestep (default 0.01)\n"
         "  --theta T          Barnes-Hut opening angle (default 0.5; 0 = exact)\n"
         "  --eps2 T           softening epsilon^2 (default 1e-6)\n"
         "  --leaf N           max bodies per leaf node (default 8)\n"
         "  --G T              gravitational constant (default 1.0)\n"
         "  --chunk N          bodies per force leaf task (default 1024)\n"
         "  --build-cutoff N   tree task cutoff + per-node slice granularity (default 32768)\n"
         "  --output-prefix P  VTK snapshot prefix (default snapshot)\n"
         "  --output-every N   write VTK every N steps (default 10)\n"
         "  --energy-every N   print energies every N steps (default 1)\n"
         "  --no-output        disable VTK output\n"
         "  --no-timers        disable per-phase timing\n"
         "  --brute-force      verify forces against O(N^2) and exit\n"
         "  -h, --help         show this help\n");
}

enum class Args { ok, error, help };

// Apply one key = value from a PEPC-style namelist group to the config.
inline void apply_namelist(const string& group, const string& key, const string& val) {
  auto num = [](const string& v) {
    string s = v;
    for (char& c : s)
      if (c == 'd' || c == 'D') c = 'e';
    return atof(s.c_str());
  };
  auto truth = [](const string& v) {
    return v.find("true") != string::npos || v.find("TRUE") != string::npos ||
           v == "T" || v == "t";
  };
  if (group == "pepcmini") {
    if (key == "tnp") cfg.tnp = (size_t)atoll(val.c_str());
    else if (key == "inputfile") {
      cfg.input = val;
      if (cfg.input.size() >= 2 &&
          (cfg.input.front() == '\'' || cfg.input.front() == '"'))
        cfg.input = cfg.input.substr(1, cfg.input.size() - 2);
      cfg.binary_input = true;
    } else if (key == "nt") cfg.steps = atoi(val.c_str());
    else if (key == "dt") cfg.dt = num(val);
    else if (key == "particle_output") {
      cfg.output = truth(val);
      if (cfg.output) cfg.output_every = 1;  // PEPC writes particle output every step
    } else if (key == "diag_interval") cfg.energy_every = atoi(val.c_str());
    // domain_output / particle_filter / particle_probe / particle_test: ignored
  } else if (group == "calc_force_coulomb") {
    if (key == "force_law" && atoi(val.c_str()) != 3)
      fprintf(stderr, "warning: force_law=%s unsupported (only 3 = gravity)\n", val.c_str());
    else if (key == "mac_select" && atoi(val.c_str()) != 0)
      fprintf(stderr, "warning: mac_select=%s unsupported (only 0 = standard MAC)\n", val.c_str());
    else if (key == "theta2") cfg.theta = sqrt(max(num(val), 0.0));
    else if (key == "eps2") cfg.eps2 = num(val);
  } else if (group == "libpepc") {
    // debug_level / np_mult / num_threads / weighted / curve_type: accepted, ignored
  } else if (group == "walk_para_pthreads") {
    if (key == "max_particles_per_thread") cfg.chunk = (size_t)atoll(val.c_str());
  }
}

// Parse a PEPC-style Fortran namelist params file into cfg (drop-in compatibility).
inline bool parse_namelist(const string& path) {
  ifstream in(path);
  if (!in) {
    fprintf(stderr, "cannot open params file %s\n", path.c_str());
    return false;
  }
  string group;
  string line;
  while (getline(in, line)) {
    size_t bang = line.find('!');
    if (bang != string::npos) line = line.substr(0, bang);
    line = strim(line);
    if (line.empty()) continue;
    if (line[0] == '&') {
      group = strim(line.substr(1));
      continue;
    }
    if (line[0] == '/') {
      group.clear();
      continue;
    }
    if (group.empty()) continue;
    size_t eq = line.find('=');
    if (eq == string::npos) continue;
    string key = strim(line.substr(0, eq));
    string val = strim(line.substr(eq + 1));
    while (!val.empty() && (val.back() == ',' || val.back() == ';')) val.pop_back();
    val = strim(val);
    apply_namelist(group, key, val);
  }
  return true;
}

// Parse the common CLI. A leading non-option argument is a PEPC-style namelist params
// file (drop-in: `barneshut_ityr pepc.params`), further flags override it.
// Prints usage/errors; exits on missing option values.
inline Args parse_args(int argc, char** argv) {
  int start = 1;
  if (argc > 1 && argv[1][0] != '-') {
    if (!parse_namelist(argv[1])) return Args::error;
    start = 2;
  }
  for (int i = start; i < argc; ++i) {
    string a = argv[i];
    auto value = [&]() -> const char* {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", a.c_str());
        exit(1);
      }
      return argv[++i];
    };
    if (a == "--input") {
      cfg.input = value();
      cfg.binary_input = false;
    } else if (a == "--steps") cfg.steps = atoi(value());
    else if (a == "--dt") cfg.dt = atof(value());
    else if (a == "--theta") cfg.theta = atof(value());
    else if (a == "--eps2") cfg.eps2 = atof(value());
    else if (a == "--leaf") cfg.leaf = atoi(value());
    else if (a == "--G") cfg.G = atof(value());
    else if (a == "--chunk") cfg.chunk = (size_t)atoll(value());
    else if (a == "--build-cutoff") cfg.build_cutoff = (size_t)atoll(value());
    else if (a == "--output-prefix") cfg.prefix = value();
    else if (a == "--output-every") cfg.output_every = atoi(value());
    else if (a == "--energy-every") cfg.energy_every = atoi(value());
    else if (a == "--no-output") cfg.output = false;
    else if (a == "--no-timers") cfg.timers = false;
    else if (a == "--brute-force") cfg.brute_force = true;
    else if (a == "-h" || a == "--help") {
      usage();
      return Args::help;
    } else {
      fprintf(stderr, "unknown option: %s\n", a.c_str());
      usage();
      return Args::error;
    }
  }
  if (cfg.input.empty()) {
    usage();
    return Args::error;
  }
  return Args::ok;
}
