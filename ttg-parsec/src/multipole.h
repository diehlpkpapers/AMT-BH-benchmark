#pragma once

#include "types.h"

#include <array>
#include <cmath>

namespace bh {

// Symmetric-tensor component order used throughout: xx, yy, zz, xy, xz, yz.
using Sym3 = std::array<double, 6>;

inline Sym3 outer_sym(const Vec3& d, double scale) {
  return {scale * d.x * d.x, scale * d.y * d.y, scale * d.z * d.z,
          scale * d.x * d.y, scale * d.x * d.z, scale * d.y * d.z};
}

inline Sym3& sym_add(Sym3& a, const Sym3& b) {
  for (int i = 0; i < 6; ++i) a[i] += b[i];
  return a;
}

inline double sym_trace(const Sym3& a) { return a[0] + a[1] + a[2]; }

// Raw (non-traceless) second moment about a fixed geometric reference point:
//   mass = Sum m_i
//   m1   = Sum m_i (r_i - ref)
//   m2   = Sum m_i (r_i - ref)_a (r_i - ref)_b
// Raw moments about a *common* reference add with a plain component-wise
// sum - no cross terms - which is what makes the upward TTG pass's combine
// step a trivial reduction (see upward_ttg.h).
struct RawMoment {
  double mass = 0.0;
  Vec3 m1;
  Sym3 m2{};

  RawMoment& operator+=(const RawMoment& o) {
    mass += o.mass;
    m1 += o.m1;
    sym_add(m2, o.m2);
    return *this;
  }

  template <typename Archive>
  void serialize(Archive& ar) { ar & mass & m1 & m2; }
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) { ar & mass & m1 & m2; }
};

inline RawMoment raw_moment_of_particle(double mass, const Vec3& pos, const Vec3& ref) {
  RawMoment mo;
  mo.mass = mass;
  Vec3 d = pos - ref;
  mo.m1 = mass * d;
  mo.m2 = outer_sym(d, mass);
  return mo;
}

// A node's finalized multipole: total mass, true center of mass, and the
// raw (non-traceless) second moment *about that center of mass*. Keeping
// the non-traceless tensor (rather than the traceless quadrupole used in
// the force kernel) is what makes shifting this multipole to a new
// reference point (raw_moment_from_multipole, below) an exact, invertible
// parallel-axis-theorem shift with no information loss - the traceless
// quadrupole alone would lose the trace and couldn't be shifted correctly.
struct Multipole {
  double mass = 0.0;
  Vec3 com;
  Sym3 iab{};  // Sum m_i (r_i - com)_a (r_i - com)_b
};

// Finalize a RawMoment (accumulated about `reference`) into a Multipole
// (mass, true center of mass, second moment about that center of mass).
inline Multipole finalize_raw_moment(const RawMoment& rm, const Vec3& reference) {
  Multipole mp;
  mp.mass = rm.mass;
  Vec3 delta = (1.0 / rm.mass) * rm.m1;  // com - reference
  mp.com = reference + delta;
  // Parallel-axis theorem: I(reference) = I(com) + mass*outer(delta,delta)
  // => I(com) = I(reference) - mass*outer(delta,delta)
  Sym3 shift = outer_sym(delta, rm.mass);
  mp.iab = rm.m2;
  for (int i = 0; i < 6; ++i) mp.iab[i] -= shift[i];
  return mp;
}

// Re-express an already-finalized Multipole as a RawMoment about a
// different reference point (e.g. this node's parent's box center), so it
// can be summed alongside sibling contributions in the parent's aggregator.
inline RawMoment raw_moment_from_multipole(const Multipole& mp, const Vec3& new_ref) {
  RawMoment rm;
  rm.mass = mp.mass;
  Vec3 delta = mp.com - new_ref;
  rm.m1 = mp.mass * delta;
  // Parallel-axis theorem: I(new_ref) = I(com) + mass*outer(delta,delta)
  Sym3 shift = outer_sym(delta, mp.mass);
  rm.m2 = mp.iab;
  sym_add(rm.m2, shift);
  return rm;
}

// Barnes-Hut multipole acceptance criterion (whitepaper Eq. 10):
// octant_edge_length / |d| < theta, tested against the source node's
// center of mass (not its geometric box center).
inline bool mac_accept(const Vec3& target_pos, const Vec3& source_com, double edge_length,
                        double theta) {
  double d = norm(target_pos - source_com);
  return edge_length < theta * d;
}

// Conservative variant of mac_accept safe for an entire leaf's particles at
// once rather than a single point: accepts only if the source would be
// accepted (per mac_accept, above) for every point within target_radius of
// target_center - i.e. tested against the worst case (nearest possible)
// distance from the source to the target box, so it can only recurse more
// than the exact per-particle test would, never approximate something that
// test would have rejected.
inline bool mac_accept_box(const Vec3& target_center, double target_radius, const Vec3& source_com,
                            double edge_length, double theta) {
  double d = norm(target_center - source_com);
  return edge_length < theta * (d - target_radius);
}

// Softened gravitational acceleration on a target from a source multipole
// (monopole, optionally + quadrupole), following the whitepaper's Eq. 3
// convention: d_vec = source_com - target_pos, so the returned acceleration
// already points from target toward source (attractive, no extra sign flip
// needed by callers). Passing an all-zero `iab` (as for a direct
// particle-particle interaction, where the "source" is a single point mass)
// makes the quadrupole term vanish automatically, so this same kernel is
// reused for leaf-leaf direct summation and brute_force.h's O(N^2)
// reference.
inline Vec3 accel_from_source(const Vec3& target_pos, double source_mass, const Vec3& source_com,
                               const Sym3& iab, double eps2, bool use_quadrupole, double G) {
  Vec3 d = source_com - target_pos;
  double d2 = norm2(d);
  double ds2 = d2 + eps2;
  double ds = std::sqrt(ds2);
  double inv_ds3 = 1.0 / (ds2 * ds);

  Vec3 a = (source_mass * inv_ds3) * d;
  if (!use_quadrupole) return G * a;

  // Traceless quadrupole tensor Q_ab = 3*I_ab - trace(I)*delta_ab, derived
  // on the fly from the stored raw second moment (see multipole.h's
  // comment on Multipole for why the non-traceless form is what's stored).
  double trace = sym_trace(iab);
  Sym3 q = iab;
  for (int i = 0; i < 3; ++i) q[i] = 3.0 * q[i] - trace;  // diagonal: xx,yy,zz
  for (int i = 3; i < 6; ++i) q[i] = 3.0 * q[i];          // off-diagonal: xy,xz,yz

  // Q . d  (Q symmetric, stored as xx,yy,zz,xy,xz,yz)
  Vec3 qd{q[0] * d.x + q[3] * d.y + q[4] * d.z, q[3] * d.x + q[1] * d.y + q[5] * d.z,
          q[4] * d.x + q[5] * d.y + q[2] * d.z};
  double dqd = dot(d, qd);

  double inv_ds5 = inv_ds3 / ds2;
  double inv_ds7 = inv_ds5 / ds2;
  a += (-1.0) * inv_ds5 * qd;
  a += (2.5 * dqd * inv_ds7) * d;
  return G * a;
}

}  // namespace bh
