#pragma once

#include <cmath>
#include <cstdint>

namespace bh {

struct Vec3 {
  double x = 0.0, y = 0.0, z = 0.0;

  Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
  Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
  Vec3& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }

  // Explicit serialization: MADNESS's archive framework (which TTG's
  // PaRSEC backend uses for its serialization layer regardless of
  // execution backend) doesn't treat this as trivially serializable by
  // default, so any type flowing through a TTG Edge needs one of these.
  template <typename Archive>
  void serialize(Archive& ar) { ar & x & y & z; }
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) { ar & x & y & z; }
};

inline Vec3 operator+(Vec3 a, const Vec3& b) { a += b; return a; }
inline Vec3 operator-(Vec3 a, const Vec3& b) { a -= b; return a; }
inline Vec3 operator*(Vec3 a, double s) { a *= s; return a; }
inline Vec3 operator*(double s, Vec3 a) { a *= s; return a; }
inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double norm2(const Vec3& a) { return dot(a, a); }
inline double norm(const Vec3& a) { return std::sqrt(norm2(a)); }

using ParticleId = std::uint32_t;

}  // namespace bh
