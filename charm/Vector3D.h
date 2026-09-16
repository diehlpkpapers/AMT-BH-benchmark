/** @file Vector3D.h
 * A three-dimensional Cartesian vector and the operations the Barnes-Hut
 * benchmark needs from it.
 *
 * This is a self-contained replacement for the Vector3D.h shipped in the
 * N-BodyShop "utility/structures" library, written so that this benchmark has
 * no dependencies beyond Charm++ itself. The API is kept source-compatible
 * with the subset of the original that the application uses.
 */

#ifndef VECTOR3D_H
#define VECTOR3D_H

#include <cmath>
#include <iostream>

template <typename T = double>
class Vector3D {
public:
  T x, y, z;

  Vector3D(T a = 0) : x(a), y(a), z(a) { }
  Vector3D(T a, T b, T c) : x(a), y(b), z(c) { }

  template <typename T2>
  Vector3D(const Vector3D<T2> &v)
    : x(static_cast<T>(v.x)), y(static_cast<T>(v.y)), z(static_cast<T>(v.z)) { }

  template <typename T2>
  Vector3D<T> &operator=(const Vector3D<T2> &v) {
    x = static_cast<T>(v.x);
    y = static_cast<T>(v.y);
    z = static_cast<T>(v.z);
    return *this;
  }

  inline T lengthSquared() const { return x * x + y * y + z * z; }
  inline T length() const { return std::sqrt(lengthSquared()); }

  inline Vector3D<T> &normalize() {
    const T len = length();
    if (len > T(0)) {
      x /= len; y /= len; z /= len;
    }
    return *this;
  }

  inline T &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
  inline const T &operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }

  inline bool operator==(const Vector3D<T> &v) const {
    return x == v.x && y == v.y && z == v.z;
  }
  inline bool operator!=(const Vector3D<T> &v) const { return !(*this == v); }

  inline Vector3D<T> operator+(const Vector3D<T> &v) const {
    return Vector3D<T>(x + v.x, y + v.y, z + v.z);
  }
  inline Vector3D<T> operator-(const Vector3D<T> &v) const {
    return Vector3D<T>(x - v.x, y - v.y, z - v.z);
  }
  inline Vector3D<T> operator-() const { return Vector3D<T>(-x, -y, -z); }

  inline Vector3D<T> &operator+=(const Vector3D<T> &v) {
    x += v.x; y += v.y; z += v.z;
    return *this;
  }
  inline Vector3D<T> &operator-=(const Vector3D<T> &v) {
    x -= v.x; y -= v.y; z -= v.z;
    return *this;
  }

  inline Vector3D<T> operator*(const T &s) const {
    return Vector3D<T>(x * s, y * s, z * s);
  }
  inline Vector3D<T> &operator*=(const T &s) {
    x *= s; y *= s; z *= s;
    return *this;
  }
  inline Vector3D<T> operator/(const T &s) const {
    return Vector3D<T>(x / s, y / s, z / s);
  }
  inline Vector3D<T> &operator/=(const T &s) {
    x /= s; y /= s; z /= s;
    return *this;
  }

  /// Component-wise product.
  inline Vector3D<T> operator*(const Vector3D<T> &v) const {
    return Vector3D<T>(x * v.x, y * v.y, z * v.z);
  }
  /// Component-wise quotient.
  inline Vector3D<T> operator/(const Vector3D<T> &v) const {
    return Vector3D<T>(x / v.x, y / v.y, z / v.z);
  }
};

/// Scalar on the left: s * v. T2 is accepted so that literals of a different
/// floating-point type than T do not force the caller to cast.
template <typename T, typename T2>
inline Vector3D<T> operator*(const T2 &s, const Vector3D<T> &v) {
  return Vector3D<T>(static_cast<T>(s) * v.x,
                     static_cast<T>(s) * v.y,
                     static_cast<T>(s) * v.z);
}

template <typename T>
inline T dot(const Vector3D<T> &a, const Vector3D<T> &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename T>
inline Vector3D<T> cross(const Vector3D<T> &a, const Vector3D<T> &b) {
  return Vector3D<T>(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const Vector3D<T> &v) {
  os << v.x << ' ' << v.y << ' ' << v.z;
  return os;
}

#ifdef __CHARMC__
#include "pup.h"

template <typename T>
inline void operator|(PUP::er &p, Vector3D<T> &v) {
  p | v.x;
  p | v.y;
  p | v.z;
}
#endif // __CHARMC__

#endif // VECTOR3D_H
