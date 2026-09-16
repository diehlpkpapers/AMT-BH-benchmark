#pragma once

// SIMD batched force walk for the parallel Barnes-Hut binaries.
// Uses ityr/common/simd.hpp (portable AVX2/AVX-512/NEON/SVE wrappers) to
// process K = B::lanes() consecutive query bodies through one shared
// depth-first tree traversal. Two invariants keep the SIMD result equal to
// the scalar walk_compact:
//   * per-body node-visit order is unchanged (each body interacts at exactly
//     the nodes the scalar walk visits, in the same DFS order), and
//   * the interaction math mirrors interact() op-for-op with no FMA
//     contraction (-ffp-contract=off, see CMakeLists.txt).
//
// The kernel is a template over a simd backend; the dispatch below picks, once
// per process, the best ISA the hardware supports (ityr::common::simd::detect()).

#include <cstdint>

#include "barneshut_common.hpp"
#include "ityr/common/simd.hpp"

namespace simd = ityr::common::simd;

// One interaction of the K packed query bodies (bx,by,bz,bm = mass lanes)
// with a single source (ox,oy,oz,om). Accumulates only into lanes set in
// `mask`; other lanes keep their exact previous value (no 0.0 add, preserves
// signed zeros). Bit-identical to interact() per lane.
template <typename B>
inline void interactN(double ox, double oy, double oz, double om,
                      typename B::vec bx, typename B::vec by, typename B::vec bz,
                      typename B::vec bm, const WalkParams& wp, typename B::vec& ax,
                      typename B::vec& ay, typename B::vec& az, typename B::vec& pot,
                      uint32_t mask) {
  if (mask == 0) return;
  typename B::vec dx = B::sub(B::set1(ox), bx);
  typename B::vec dy = B::sub(B::set1(oy), by);
  typename B::vec dz = B::sub(B::set1(oz), bz);
  typename B::vec r2 = B::add(B::add(B::add(B::mul(dx, dx), B::mul(dy, dy)),
                                     B::mul(dz, dz)),
                              B::set1(wp.eps2));
  typename B::vec ir = B::div(B::set1(1.0), B::sqrt_(r2));
  typename B::vec inv = B::mul(B::mul(ir, ir), ir);
  typename B::vec gom = B::set1(wp.G * om);
  typename B::vec cax = B::mul(B::mul(gom, dx), inv);
  typename B::vec cay = B::mul(B::mul(gom, dy), inv);
  typename B::vec caz = B::mul(B::mul(gom, dz), inv);
  typename B::vec pterm = B::mul(B::mul(B::mul(B::set1(wp.G), bm), B::set1(om)), ir);
  ax = B::blend(mask, B::add(ax, cax), ax);
  ay = B::blend(mask, B::add(ay, cay), ay);
  az = B::blend(mask, B::add(az, caz), az);
  pot = B::blend(mask, B::sub(pot, pterm), pot);
}

// Shared depth-first walk for the K query bodies at compact indices
// [base, base+K). `active` is the K-bit mask of bodies still exploring this
// subtree. K is B::lanes(); the mask stays uint32_t (K <= 32).
// `level` is the traversal depth and indexes wp.s2[] (see Node in
// barneshut_common.hpp).
//
// Iterative explicit-stack DFS so the accumulators and query-body vectors stay
// in registers instead of being passed through every recursive call. Children
// are pushed in reverse, so the pop order equals the recursive DFS order
// (children are visited in increasing octant order). The stack is bounded by
// 8 + 7*(NLEVELS-1) frames: each internal node has <= 8 children, and the
// build forces a leaf at level >= NLEVELS.
template <typename B>
inline void walk_compactN(int idx, int level, size_t base,
                          typename B::vec bx, typename B::vec by, typename B::vec bz,
                          typename B::vec bm, const Node* nodes, const CompactBody* bodies,
                          const WalkParams& wp, typename B::vec& ax, typename B::vec& ay,
                          typename B::vec& az, typename B::vec& pot, uint32_t active) {
  if (active == 0) return;
  const size_t K = B::lanes();
  struct Frame { int idx, level; uint32_t active; };
  Frame st[NLEVELS * 7 + 8];
  int sp = 0;
  st[sp++] = {idx, level, active};
  while (sp > 0) {
    const Frame f = st[--sp];
    const Node& n = nodes[f.idx];
    if (n.leaf) {
      for (int i = n.first; i < n.first + n.count; ++i) {
        // lane k is the query body at compact index base+k: skip when it is o.
        uint32_t self = (i >= (int)base && i < (int)(base + K)) ? (1u << (i - base)) : 0;
        uint32_t m = f.active & ~self;
        if (m == 0) continue;
        const CompactBody& o = bodies[i];
        interactN<B>(o.x, o.y, o.z, o.mass, bx, by, bz, bm, wp, ax, ay, az, pot, m);
      }
      continue;
    }
    typename B::vec dx = B::sub(B::set1(n.cx), bx);
    typename B::vec dy = B::sub(B::set1(n.cy), by);
    typename B::vec dz = B::sub(B::set1(n.cz), bz);
    typename B::vec d2 = B::add(B::add(B::mul(dx, dx), B::mul(dy, dy)), B::mul(dz, dz));
    // accept lane iff s2 < theta2*d2 (identical values to the scalar test)
    uint32_t acc = B::lt_mask(B::set1(wp.s2[f.level]), B::mul(B::set1(wp.theta2), d2)) & f.active;
    if (acc) interactN<B>(n.cx, n.cy, n.cz, n.mass, bx, by, bz, bm, wp, ax, ay, az, pot, acc);
    uint32_t remain = f.active & ~acc;
    if (remain == 0) continue;
    for (int k = n.nchild - 1; k >= 0; --k)
      st[sp++] = {f.idx + n.child[k], f.level + 1, remain};
  }
}

// ---- runtime dispatch (chosen once) ----

// Pack K bodies (compact indices base..base+K-1) into lanes, walk the shared
// traversal, unpack. out_*[K] receive this batch's ForceResult fields. The
// pack buffers are deliberately not zero-initialised (only the first K entries
// are read, and GCC cannot always eliminate the memset, e.g. for SVE where
// lanes() is not constexpr); the unpack is one vector store per component.
template <typename B>
inline void forceN_dispatch(size_t base, const CompactBody* bodies, const Node* tree,
                            const WalkParams& wp, double* out_ax, double* out_ay,
                            double* out_az, double* out_pot) {
  const size_t K = B::lanes();
  double x[64], y[64], z[64], m[64];
  for (size_t k = 0; k < K; ++k) {
    x[k] = bodies[base + k].x;
    y[k] = bodies[base + k].y;
    z[k] = bodies[base + k].z;
    m[k] = bodies[base + k].mass;
  }
  typename B::vec bx = B::load(x), by = B::load(y), bz = B::load(z), bm = B::load(m);
  typename B::vec ax = B::set1(0.0), ay = B::set1(0.0), az = B::set1(0.0), pot = B::set1(0.0);
  uint32_t active = (K >= 32) ? 0xFFFFFFFFu : ((1u << K) - 1);
  walk_compactN<B>(0, 0, base, bx, by, bz, bm, tree, bodies, wp, ax, ay, az, pot, active);
  B::store(out_ax, ax);
  B::store(out_ay, ay);
  B::store(out_az, az);
  B::store(out_pot, pot);
}

// fn processes K = lanes() consecutive bodies into out_* (arrays sized >= 64).
using forceN_fn = void (*)(size_t base, const CompactBody* bodies, const Node* tree,
                           const WalkParams& wp, double* out_ax, double* out_ay,
                           double* out_az, double* out_pot);

// Pick, once per process, the best ISA the current hardware supports among the
// compiled-in backends (AVX-512 > AVX2 on x86, SVE > NEON on ARM64, scalar).
inline forceN_fn forceN_impl() {
  simd::isa best = simd::detect();
  switch (best) {
#if defined(ITYR_SIMD_HAS_AVX512F)
    case simd::isa::avx512f:
      return &forceN_dispatch<simd::avx512f_backend>;
#endif
#if defined(ITYR_SIMD_HAS_AVX2)
    case simd::isa::avx2:
      return &forceN_dispatch<simd::avx2_backend>;
#endif
#if defined(ITYR_SIMD_HAS_SVE)
    case simd::isa::sve: return &forceN_dispatch<simd::sve_backend>;
#endif
#if defined(ITYR_SIMD_HAS_NEON)
    case simd::isa::neon:
      return &forceN_dispatch<simd::neon_backend>;
#endif
    default:
      return &forceN_dispatch<simd::scalar_backend>;
  }
}

// The one entry point used by force_range. Also lets the caller know which ISA
// was selected and its lane count (for the batch stride / tail).
struct forceN_desc {
  forceN_fn fn;
  size_t lanes;
  const char* isa_name;
};

inline forceN_desc forceN() {
  static const forceN_desc d = [] {
    forceN_fn fn = forceN_impl();
    simd::isa best = simd::detect();
    size_t lanes = 4;
    switch (best) {
#if defined(ITYR_SIMD_HAS_AVX512F)
      case simd::isa::avx512f: lanes = simd::avx512f_backend::lanes(); break;
#endif
#if defined(ITYR_SIMD_HAS_AVX2)
      case simd::isa::avx2: lanes = simd::avx2_backend::lanes(); break;
#endif
#if defined(ITYR_SIMD_HAS_SVE)
      case simd::isa::sve: lanes = simd::sve_backend::lanes(); break;
#endif
#if defined(ITYR_SIMD_HAS_NEON)
      case simd::isa::neon: lanes = simd::neon_backend::lanes(); break;
#endif
      default: break;
    }
    return forceN_desc{fn, lanes, simd::isa_name(best)};
  }();
  return d;
}
inline const char* forceN_isa_name() {
  return simd::isa_name(simd::detect());
}
