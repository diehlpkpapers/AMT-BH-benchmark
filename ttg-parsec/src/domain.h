#pragma once

#include "key.h"
#include "types.h"

#include <utility>

namespace bh {

// The simulation domain is a cube; per-level edge length is derived by
// halving.
struct Domain {
  Vec3 center;
  double half_width = 0.0;
};

// A node's box geometry (center, half-width) is fully determined by its Key
// (level + per-axis translation) and the domain's root box - no per-node
// storage or lookup is ever needed for it, so tasks that only need geometry
// never have to touch the tree at all, just this (tiny, trivially
// replicable) Domain plus whatever Key they already have.
inline std::pair<Vec3, double> box_geometry(const Domain& domain, const Key& key) {
  double cell_width = 2.0 * domain.half_width / static_cast<double>(1u << key.level());
  double half_width = 0.5 * cell_width;
  const auto& t = key.translation();
  Vec3 center{domain.center.x - domain.half_width + cell_width * (t[0] + 0.5),
              domain.center.y - domain.half_width + cell_width * (t[1] + 0.5),
              domain.center.z - domain.half_width + cell_width * (t[2] + 0.5)};
  return {center, half_width};
}

}  // namespace bh
