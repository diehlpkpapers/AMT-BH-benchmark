#pragma once

#include "types.h"

namespace bh {

struct Particle {
  ParticleId id = 0;
  double mass = 0.0;
  Vec3 pos;
  Vec3 vel;

  // Explicit serialization: MADNESS's archive framework (which TTG's
  // PaRSEC backend uses for its serialization layer regardless of
  // execution backend) doesn't treat this as trivially serializable by
  // default, so any type flowing through a TTG Edge needs one of these -
  // see Vec3 (types.h) for the same pattern. Particles now flow through
  // edges directly (see traversal_ttg.h/upward_ttg.h) rather than being
  // looked up via a shared array, so this is required, not just tidy.
  template <typename Archive>
  void serialize(Archive& ar) { ar & id & mass & pos & vel; }
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) { ar & id & mass & pos & vel; }
};

}  // namespace bh
