/** @file OrientedBox.h
 * An axis-aligned bounding box in three dimensions.
 *
 * This is a self-contained replacement for the OrientedBox.h shipped in the
 * N-BodyShop "utility/structures" library, written so that this benchmark has
 * no dependencies beyond Charm++ itself.
 *
 * NOTE: unlike the original, this class does NOT derive from an abstract
 * Shape<T> base. That base contributed a vtable pointer, and OrientedBox is
 * embedded in BoundingBox, which is shipped between PEs as raw bytes by the
 * BoundingBoxGrow reduction. Sending a vptr across address spaces is not
 * meaningful, so the base class is deliberately dropped here.
 */

#ifndef ORIENTEDBOX_H
#define ORIENTEDBOX_H

#include <cmath>
#include <iostream>

#include "Vector3D.h"

template <typename T = double>
class OrientedBox {
public:
  /// The corner with the minimum x, y and z values.
  Vector3D<T> lesser_corner;
  /// The corner with the maximum x, y and z values.
  Vector3D<T> greater_corner;

  OrientedBox() { reset(); }

  OrientedBox(const Vector3D<T> &lesser, const Vector3D<T> &greater)
    : lesser_corner(lesser), greater_corner(greater) { }

  /// Return the box to the empty state: any grow() will set both corners.
  void reset() {
    lesser_corner = Vector3D<T>(HUGE_VAL, HUGE_VAL, HUGE_VAL);
    greater_corner = Vector3D<T>(-HUGE_VAL, -HUGE_VAL, -HUGE_VAL);
  }

  /// Has anything been added to this box yet?
  bool initialized() const {
    return lesser_corner.x <= greater_corner.x
        && lesser_corner.y <= greater_corner.y
        && lesser_corner.z <= greater_corner.z;
  }

  bool contains(const Vector3D<T> &point) const {
    return point.x >= lesser_corner.x && point.x <= greater_corner.x
        && point.y >= lesser_corner.y && point.y <= greater_corner.y
        && point.z >= lesser_corner.z && point.z <= greater_corner.z;
  }

  /// Enlarge the box so that it contains \c point.
  void grow(const Vector3D<T> &point) {
    if (point.x < lesser_corner.x) lesser_corner.x = point.x;
    if (point.y < lesser_corner.y) lesser_corner.y = point.y;
    if (point.z < lesser_corner.z) lesser_corner.z = point.z;

    if (point.x > greater_corner.x) greater_corner.x = point.x;
    if (point.y > greater_corner.y) greater_corner.y = point.y;
    if (point.z > greater_corner.z) greater_corner.z = point.z;
  }

  /// Enlarge the box so that it contains \c box.
  void grow(const OrientedBox<T> &box) {
    if (box.lesser_corner.x < lesser_corner.x) lesser_corner.x = box.lesser_corner.x;
    if (box.lesser_corner.y < lesser_corner.y) lesser_corner.y = box.lesser_corner.y;
    if (box.lesser_corner.z < lesser_corner.z) lesser_corner.z = box.lesser_corner.z;

    if (box.greater_corner.x > greater_corner.x) greater_corner.x = box.greater_corner.x;
    if (box.greater_corner.y > greater_corner.y) greater_corner.y = box.greater_corner.y;
    if (box.greater_corner.z > greater_corner.z) greater_corner.z = box.greater_corner.z;
  }

  T volume() const {
    return (greater_corner.x - lesser_corner.x)
         * (greater_corner.y - lesser_corner.y)
         * (greater_corner.z - lesser_corner.z);
  }

  Vector3D<T> center() const { return (lesser_corner + greater_corner) / T(2); }
  Vector3D<T> size() const { return greater_corner - lesser_corner; }

  OrientedBox<T> &shift(const Vector3D<T> &v) {
    lesser_corner += v;
    greater_corner += v;
    return *this;
  }
};

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const OrientedBox<T> &b) {
  os << '{' << b.lesser_corner << ',' << b.greater_corner << '}';
  return os;
}

#ifdef __CHARMC__
#include "pup.h"

template <typename T>
inline void operator|(PUP::er &p, OrientedBox<T> &b) {
  p | b.lesser_corner;
  p | b.greater_corner;
}
#endif // __CHARMC__

#endif // ORIENTEDBOX_H
