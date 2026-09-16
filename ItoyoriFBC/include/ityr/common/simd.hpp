#pragma once

// Portable SIMD layer for the ItoyoriFBC runtime.
//
// Design: the walk kernel is written once as a template over a *backend*; each
// backend implements an N-lane double vector (N = lanes()) with
// IEEE-correctly-rounded ops (add/sub/mul/div/sqrt are bit-identical to the
// scalar C++ operators, so any backend gives byte-identical FP results — only
// the width/ISA differs). Backends are selected at COMPILE time (what this TU
// can emit):
//   scalar (always), AVX2 (x86_64 -mavx2), AVX-512F (x86_64 -mavx512f),
//   NEON (aarch64, baseline), SVE (aarch64 -march=armv8-a+sve).
// At RUNTIME, `detect()` picks the best ISA the CURRENT hardware supports among
// the compiled-in backends, and the kernel dispatches there once. So the SAME
// binary runs on any x86_64 or ARM64 machine and uses whatever SIMD level the
// hardware actually allows — no need to know the cluster's CPU at build time.
//
// NO external libraries, NO OpenMP — only compiler intrinsics guarded by ISA
// macros.

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(__AVX2__)
#define ITYR_SIMD_HAS_AVX2 1
#endif
#if defined(__AVX512F__)
#define ITYR_SIMD_HAS_AVX512F 1
#endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define ITYR_SIMD_HAS_NEON 1
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define ITYR_SIMD_HAS_SVE 1
#endif
#endif

namespace ityr::common::simd {

// Best SIMD ISA this TU can emit AND the current hardware supports.
enum class isa { scalar, avx2, avx512f, neon, sve };

inline const char* isa_name(isa i) {
  switch (i) {
    case isa::avx512f: return "AVX-512F";
    case isa::avx2:    return "AVX2";
    case isa::sve:     return "SVE";
    case isa::neon:    return "NEON";
    default:           return "scalar";
  }
}

inline isa detect() {
#if defined(__aarch64__)
#if defined(ITYR_SIMD_HAS_SVE)
  // SVE is mandatory only on machines compiled for it; when this TU targets
  // SVE the hardware supports it (homogeneous clusters).
  return isa::sve;
#else
  // NEON is mandatory in ARMv8-A; this TU is compiled for it.
  return isa::neon;
#endif
#elif defined(__x86_64__)
#if defined(ITYR_SIMD_HAS_AVX512F)
  if (__builtin_cpu_supports("avx512f")) return isa::avx512f;
#endif
#if defined(ITYR_SIMD_HAS_AVX2)
  if (__builtin_cpu_supports("avx2")) return isa::avx2;
#endif
  return isa::scalar;
#else
  return isa::scalar;
#endif
}

// ---------------- backends ----------------

// Each backend: `using vec` = N-lane double vector; `lanes()` = N;
// static ops set1/set4/load/add/sub/mul/div/sqrt_/extract/lt_mask/blend.
// All ops are IEEE-identical to the scalar C++ operators, so byte-identity
// holds across every backend (only N differs).

struct scalar_backend {
  struct vec {
    double d[4];
  };
  static constexpr size_t lanes() { return 4; }
  static inline vec set1(double s)              { return {{s, s, s, s}}; }
  static inline vec set4(double a, double b, double c, double d) {
    return {{a, b, c, d}};
  }
  static inline vec load(const double* p)       { return {{p[0], p[1], p[2], p[3]}}; }
  static inline vec add(vec a, vec b) {
    return {{a.d[0] + b.d[0], a.d[1] + b.d[1], a.d[2] + b.d[2], a.d[3] + b.d[3]}};
  }
  static inline vec sub(vec a, vec b) {
    return {{a.d[0] - b.d[0], a.d[1] - b.d[1], a.d[2] - b.d[2], a.d[3] - b.d[3]}};
  }
  static inline vec mul(vec a, vec b) {
    return {{a.d[0] * b.d[0], a.d[1] * b.d[1], a.d[2] * b.d[2], a.d[3] * b.d[3]}};
  }
  static inline vec div(vec a, vec b) {
    return {{a.d[0] / b.d[0], a.d[1] / b.d[1], a.d[2] / b.d[2], a.d[3] / b.d[3]}};
  }
  static inline vec sqrt_(vec a) {
    return {{sqrt(a.d[0]), sqrt(a.d[1]), sqrt(a.d[2]), sqrt(a.d[3])}};
  }
  static inline void store(double* p, vec a) {
    p[0] = a.d[0];
    p[1] = a.d[1];
    p[2] = a.d[2];
    p[3] = a.d[3];
  }
  static inline double extract(vec a, int lane) { return a.d[lane]; }
  static inline uint32_t lt_mask(vec a, vec b) {
    return (uint32_t)(a.d[0] < b.d[0]) | ((uint32_t)(a.d[1] < b.d[1]) << 1) |
           ((uint32_t)(a.d[2] < b.d[2]) << 2) | ((uint32_t)(a.d[3] < b.d[3]) << 3);
  }
  static inline vec blend(uint32_t mask, vec a, vec b) {
    return {{(mask & 1) ? a.d[0] : b.d[0], (mask & 2) ? a.d[1] : b.d[1],
             (mask & 4) ? a.d[2] : b.d[2], (mask & 8) ? a.d[3] : b.d[3]}};
  }
};

#if defined(ITYR_SIMD_HAS_AVX2)

struct avx2_backend {
  using vec = __m256d;
  static constexpr size_t lanes() { return 4; }
  static inline vec set1(double s)              { return _mm256_set1_pd(s); }
  static inline vec set4(double a, double b, double c, double d) {
    return _mm256_set_pd(d, c, b, a);
  }
  static inline vec load(const double* p)       { return _mm256_loadu_pd(p); }
  static inline vec add(vec a, vec b)           { return _mm256_add_pd(a, b); }
  static inline vec sub(vec a, vec b)           { return _mm256_sub_pd(a, b); }
  static inline vec mul(vec a, vec b)           { return _mm256_mul_pd(a, b); }
  static inline vec div(vec a, vec b)           { return _mm256_div_pd(a, b); }
  static inline vec sqrt_(vec a)                { return _mm256_sqrt_pd(a); }
  static inline void store(double* p, vec a)    { _mm256_storeu_pd(p, a); }
  static inline double extract(vec a, int lane) {
    double t[4];
    _mm256_storeu_pd(t, a);
    return t[lane];
  }
  static inline uint32_t lt_mask(vec a, vec b) {
    return (uint32_t)_mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_LT_OQ));
  }
  static inline vec blend(uint32_t mask, vec a, vec b) {
    __m256d m = _mm256_castsi256_pd(_mm256_set_epi64x(
        (mask & 8) ? -1LL : 0LL, (mask & 4) ? -1LL : 0LL,
        (mask & 2) ? -1LL : 0LL, (mask & 1) ? -1LL : 0LL));
    return _mm256_blendv_pd(b, a, m);
  }
};

#endif

#if defined(ITYR_SIMD_HAS_AVX512F)

struct avx512f_backend {
  using vec = __m512d;
  static constexpr size_t lanes() { return 8; }
  static inline vec set1(double s)              { return _mm512_set1_pd(s); }
  static inline vec set4(double a, double b, double c, double d) {
    // _mm512_set_pd takes e7..e0, so the operands go in the LOW four lanes to
    // match every other backend (extract(set4(a,..),0) == a).
    return _mm512_set_pd(0.0, 0.0, 0.0, 0.0, d, c, b, a);
  }
  static inline vec load(const double* p)       { return _mm512_loadu_pd(p); }
  static inline vec add(vec a, vec b)           { return _mm512_add_pd(a, b); }
  static inline vec sub(vec a, vec b)           { return _mm512_sub_pd(a, b); }
  static inline vec mul(vec a, vec b)           { return _mm512_mul_pd(a, b); }
  static inline vec div(vec a, vec b)           { return _mm512_div_pd(a, b); }
  static inline vec sqrt_(vec a)                { return _mm512_sqrt_pd(a); }
  static inline void store(double* p, vec a)    { _mm512_storeu_pd(p, a); }
  static inline double extract(vec a, int lane) {
    double t[8];
    _mm512_storeu_pd(t, a);
    return t[lane];
  }
  static inline uint32_t lt_mask(vec a, vec b) {
    return (uint32_t)_mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
  }
  static inline vec blend(uint32_t mask, vec a, vec b) {
    // Backend contract: "a where the mask bit is set, b elsewhere".
    // _mm512_mask_blend_pd(k, x, y) returns y where k is set, so swap the operands.
    return _mm512_mask_blend_pd((__mmask8)mask, b, a);
  }
};

#endif

#if defined(ITYR_SIMD_HAS_NEON)

// NEON float64x2 is 128-bit (2 doubles); a 4-wide vector is two of them.
struct neon_backend {
  struct vec {
    float64x2_t lo, hi;
  };
  static constexpr size_t lanes() { return 4; }
  static inline vec set1(double s) {
    float64x2_t t = vdupq_n_f64(s);
    return {t, t};
  }
  static inline vec set4(double a, double b, double c, double d) {
    return {vsetq_lane_f64(a, vdupq_n_f64(b), 0),
            vsetq_lane_f64(c, vdupq_n_f64(d), 0)};
  }
  static inline vec load(const double* p) {
    return {vld1q_f64(p), vld1q_f64(p + 2)};
  }
  static inline vec add(vec a, vec b) { return {vaddq_f64(a.lo, b.lo), vaddq_f64(a.hi, b.hi)}; }
  static inline vec sub(vec a, vec b) { return {vsubq_f64(a.lo, b.lo), vsubq_f64(a.hi, b.hi)}; }
  static inline vec mul(vec a, vec b) { return {vmulq_f64(a.lo, b.lo), vmulq_f64(a.hi, b.hi)}; }
  static inline vec div(vec a, vec b) { return {vdivq_f64(a.lo, b.lo), vdivq_f64(a.hi, b.hi)}; }
  static inline vec sqrt_(vec a)      { return {vsqrtq_f64(a.lo), vsqrtq_f64(a.hi)}; }
  static inline void store(double* p, vec a) {
    vst1q_f64(p, a.lo);
    vst1q_f64(p + 2, a.hi);
  }
  static inline double extract(vec a, int lane) {
    double t[4];
    vst1q_f64(t, a.lo);
    vst1q_f64(t + 2, a.hi);
    return t[lane];
  }
  static inline uint32_t lt_mask(vec a, vec b) {
    // vcltq_f64 already yields uint64x2_t.
    uint64x2_t ml = vcltq_f64(a.lo, b.lo);
    uint64x2_t mh = vcltq_f64(a.hi, b.hi);
    return (uint32_t)(vgetq_lane_u64(ml, 0) != 0) |
           ((uint32_t)(vgetq_lane_u64(ml, 1) != 0) << 1) |
           ((uint32_t)(vgetq_lane_u64(mh, 0) != 0) << 2) |
           ((uint32_t)(vgetq_lane_u64(mh, 1) != 0) << 3);
  }
  static inline vec blend(uint32_t mask, vec a, vec b) {
    uint64x2_t ml = vsetq_lane_u64(mask & 1 ? ~0ull : 0ull, vdupq_n_u64(mask & 2 ? ~0ull : 0ull), 0);
    uint64x2_t mh = vsetq_lane_u64(mask & 4 ? ~0ull : 0ull, vdupq_n_u64(mask & 8 ? ~0ull : 0ull), 0);
    return {vbslq_f64(ml, a.lo, b.lo), vbslq_f64(mh, a.hi, b.hi)};
  }
};

#endif

#if defined(ITYR_SIMD_HAS_SVE)

// SVE vector length is runtime (svcntd()); this TU is compiled for SVE, so the
// hardware supports at least that length (homogeneous clusters). Predicates
// convert to/from integer masks via svpmov (no predicate store/load needed).
struct sve_backend {
  using vec = svfloat64_t;
  static size_t lanes() { return (size_t)svcntd(); }
  static inline vec set1(double s)              { return svdup_f64(s); }
  static inline vec set4(double a, double b, double c, double d) {
    double t[64] = {};
    t[0] = a; t[1] = b; t[2] = c; t[3] = d;
    return svld1_f64(svptrue_b64(), t);
  }
  static inline vec load(const double* p)       { return svld1_f64(svptrue_b64(), p); }
  static inline vec add(vec a, vec b)           { return svadd_f64_x(svptrue_b64(), a, b); }
  static inline vec sub(vec a, vec b)           { return svsub_f64_x(svptrue_b64(), a, b); }
  static inline vec mul(vec a, vec b)           { return svmul_f64_x(svptrue_b64(), a, b); }
  static inline vec div(vec a, vec b)           { return svdiv_f64_x(svptrue_b64(), a, b); }
  static inline vec sqrt_(vec a)                { return svsqrt_f64_x(svptrue_b64(), a); }
  static inline void store(double* p, vec a)    { svst1_f64(svptrue_b64(), p, a); }
  static inline double extract(vec a, int lane) {
    double t[64];
    svst1_f64(svptrue_b64(), t, a);
    return t[lane];
  }
  static inline uint32_t lt_mask(vec a, vec b) {
    svbool_t p = svcmplt_f64(svptrue_b64(), a, b);
    svuint64_t m = svpmov_u64_z(p);
    uint64_t t[64];
    svst1_u64(svptrue_b64(), t, m);
    uint32_t mask = 0;
    size_t n = lanes();
    for (size_t k = 0; k < n && k < 32; ++k) mask |= (uint32_t)t[k] << k;
    return mask;
  }
  static inline vec blend(uint32_t mask, vec a, vec b) {
    uint64_t bits[64] = {};
    size_t n = lanes();
    for (size_t k = 0; k < n && k < 32; ++k) bits[k] = (mask >> k) & 1;
    svuint64_t m = svld1_u64(svptrue_b64(), bits);
    svbool_t p = svpmov_u64(m);
    return svsel_f64(p, a, b);
  }
};

#endif

}
