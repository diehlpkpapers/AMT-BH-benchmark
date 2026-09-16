#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>
#include <tuple>
#include <utility>

namespace bh {

using Dimension = int;
using Level = std::uint32_t;       // level of the tree (0..MAX_LEVEL)
using Translation = std::int32_t;  // translation along one dimension (0..2**level-1)
using HashValue = std::uint64_t;

constexpr Level MAX_LEVEL = 31;      // max level of the tree (2**MAX_LEVEL translations per dimension)
constexpr Dimension NDIM = 3;        // 3D octree only
// max translation (0..2**MAX_LEVEL-1); computed via unsigned shift since
// 1<<MAX_LEVEL overflows signed int (MAX_LEVEL==31) before the cast down.
constexpr Translation MAX_TRANSLATION = static_cast<Translation>((1u << MAX_LEVEL) - 1);

/// Extracts the n'th bit as 0 or 1
inline int get_bit(int bits, Dimension n) { return ((bits >> n) & 0x1); }

/// Extracts the low bit as 0 or 1
inline Translation low_bit(Translation l) { return l & Translation(1); }

namespace detail {
inline std::uint64_t rot(std::uint64_t x) {
  constexpr std::uint64_t n = 27;
  return (x << n) | (x >> (64 - n));
}
}  // namespace detail

/// Multiplicative hash with rotation to mix bits (std::hash does not) - empirically good enough for keys
inline HashValue mulhash(HashValue hash, Translation data) {
  constexpr HashValue m = 11400714819323198393ul;
  return detail::rot(hash) ^ (static_cast<HashValue>(data) * m);
}

/// Identifies a node of the (3D) octree by its level and per-axis translation.
/// Parent and children are computed directly from a key's own level/translation
/// (parent(), child_at(), ...) - no separate id/lookup table needed to navigate
/// between a node and its relatives; Tree (tree.h) keys its sparse node table by
/// Key rather than by any separately-assigned id.
class Key {
 private:
  Level n = 0;
  std::array<Translation, NDIM> l{};

  /// Refreshes the hash value. Note that the default std::hash does not mix enough.
  HashValue rehash() const {
    HashValue hashvalue = mulhash(HashValue(0), n);
    for (Dimension d = 0; d < NDIM; d++) hashvalue = mulhash(hashvalue, l[d]);
    return hashvalue;
  }

 public:
  enum class Direction { Left = -1, Right = 1 };

  static constexpr int num_children() { return (1 << NDIM); }

  constexpr Key() = default;
  constexpr Key(const Key&) = default;
  constexpr Key(Key&&) = default;

  /// Construct from level and translation
  constexpr Key(Level n, const std::array<Translation, NDIM>& l) : n(n), l(l) {}

  /// Construct from level with translation=0
  constexpr explicit Key(Level n) : n(n) {}

  Key& operator=(const Key&) = default;
  Key& operator=(Key&&) = default;

  bool operator<(const Key& other) const {
    auto compare = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return std::tie(n, l[Is]...) < std::tie(other.n, other.l[Is]...);
    };
    return compare(std::make_index_sequence<NDIM>{});
  }

  bool operator==(const Key& other) const {
    auto compare = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return n == other.n && ((l[Is] == other.l[Is]) && ...);
    };
    return compare(std::make_index_sequence<NDIM>{});
  }

  bool operator!=(const Key& other) const { return !(*this == other); }

  /// Hash to unsigned value
  HashValue hash() const { return rehash(); }

  /// Level (n = 0, 1, 2, ...)
  Level level() const { return n; }

  /// Translation (each element 0, 1, ..., 2**level-1)
  const std::array<Translation, NDIM>& translation() const { return l; }

  /// Parent key
  ///
  /// Default is the immediate parent (generation=1). To get the
  /// grandparent use generation=2, and similarly for great-grandparents.
  ///
  /// !! If there is no such parent it quietly returns the closest match
  /// (which may be self if this is the top of the tree).
  Key parent(Level generation = 1) const {
    generation = std::min(generation, n);
    std::array<Translation, NDIM> pl;
    for (Dimension i = 0; i < NDIM; i++) pl[i] = (l[i] >> generation);
    return Key(n - generation, pl);
  }

  /// First child in lexical ordering of KeyChildren iteration
  Key first_child() const {
    assert(n < MAX_LEVEL);
    std::array<Translation, NDIM> cl = this->l;
    for (auto& x : cl) x = 2 * x;
    return Key(n + 1, cl);
  }

  /// Last child in lexical ordering of KeyChildren iteration
  Key last_child() const {
    assert(n < MAX_LEVEL);
    std::array<Translation, NDIM> cl = this->l;
    for (auto& x : cl) x = 2 * x + 1;
    return Key(n + 1, cl);
  }

  /// Used by iterator to increment child translation
  void next_child(int& bits) {
    int oldbits = bits++;
    for (Dimension d = 0; d < NDIM; ++d) {
      l[d] += get_bit(bits, d) - get_bit(oldbits, d);
    }
  }

  /// Map translation to child index in parent which is formed from binary code (bits)
  int childindex() const {
    int b = low_bit(l[NDIM - 1]);
    for (Dimension d = NDIM - 1; d > 0; d--) b = (b << 1) | low_bit(l[d - 1]);
    return b;
  }

  /// Return the Key of the child at position idx \in [0, 1<<NDIM)
  Key child_at(int idx) const {
    assert(n < MAX_LEVEL);
    assert(idx < num_children());
    std::array<Translation, NDIM> cl = this->l;
    for (Dimension d = 0; d < NDIM; ++d) cl[d] = 2 * cl[d] + ((idx & (1 << d)) ? 1 : 0);
    return Key(n + 1, cl);
  }

  bool is_left_child(Dimension axis) const {
    assert(n < MAX_LEVEL);
    return (this->l[axis] % 2) == 0;
  }

  bool is_right_child(Dimension axis) const {
    assert(n < MAX_LEVEL);
    return (this->l[axis] % 2) == 1;
  }

  Key neighbor(const Key& disp) const {
    std::array<Translation, NDIM> nl = this->l;
    for (Dimension d = 0; d < NDIM; ++d) {
      nl[d] += disp.l[d];
      if (nl[d] < 0 || nl[d] >= static_cast<Translation>(1ul << n)) return invalid();
    }
    return Key(n, nl);
  }

  Key neighbor(Dimension axis, int disp) const {
    if ((is_right_boundary(axis) && disp > 0) || (is_left_boundary(axis) && disp < 0)) return invalid();
    std::array<Translation, NDIM> nl = this->l;
    nl[axis] += disp;
    return Key(n, nl);
  }

  Key operator+(const std::array<int, NDIM>& disp) const {
    std::array<Translation, NDIM> nl = this->l;
    for (Dimension d = 0; d < NDIM; ++d) nl[d] += disp[d];
    return Key(n, nl);
  }

  bool is_left_boundary(Dimension axis) const { return (l[axis] == 0); }

  bool is_right_boundary(Dimension axis) const {
    return (l[axis] == static_cast<Translation>((1ul << n) - 1));
  }

  bool is_boundary(Dimension axis) const { return is_left_boundary(axis) || is_right_boundary(axis); }

  bool is_boundary(Dimension axis, Direction dir) const {
    return (dir == Direction::Left) ? is_left_boundary(axis) : is_right_boundary(axis);
  }

  /// Invalid-key sentinel, same "wrap to max value" convention as NodeId's
  /// invalid_node (types.h) - relies on Level being unsigned.
  static constexpr Key invalid() { return Key(static_cast<Level>(-1)); }

  constexpr bool is_invalid() const { return n == static_cast<Level>(-1); }

  bool is_valid() const {
    if (is_invalid()) return false;
    bool valid = true;
    for (Dimension d = 0; valid && d < NDIM; ++d) {
      valid = valid && (l[d] >= 0) && (l[d] < static_cast<Translation>(1ul << n));
    }
    return valid;
  }

  Key step(Dimension axis, int width) const {
    std::array<Translation, NDIM> sl = translation();
    sl[axis] += width;
    return Key(level(), sl);
  }

  bool is_ancestor_of(const Key& other) const {
    if (n >= other.n) return false;
    Level shift = other.n - n;
    for (Dimension d = 0; d < NDIM; ++d) {
      if (l[d] != (other.l[d] >> shift)) return false;
    }
    return true;
  }

  bool is_descendant_of(const Key& other) const { return other.is_ancestor_of(*this); }

  Key operator-(const Key& other) const {
    assert(n == other.n);
    std::array<Translation, NDIM> dl = this->l;
    for (Dimension d = 0; d < NDIM; ++d) dl[d] -= other.l[d];
    return Key(n, dl);
  }

  std::uint64_t distsq() const {
    std::uint64_t dist = 0;
    for (std::size_t d = 0; d < NDIM; ++d) dist += static_cast<std::uint64_t>(l[d]) * l[d];
    return dist;
  }

  // Explicit serialization: MADNESS's archive framework (which TTG's
  // PaRSEC backend uses for its serialization layer regardless of
  // execution backend) doesn't treat this as trivially serializable by
  // default, so any type flowing through a TTG Edge needs one of these -
  // see Vec3 (types.h) for the same pattern.
  template <typename Archive>
  void serialize(Archive& ar) { ar & n & l[0] & l[1] & l[2]; }
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) { ar & n & l[0] & l[1] & l[2]; }
};

/// Range object used to iterate over children of a key
class KeyChildren {
  struct iterator {
    Key value;
    int bits;
    iterator(const Key& value, int bits) : value(value), bits(bits) {}
    operator const Key&() const { return value; }
    const Key& operator*() const { return value; }
    iterator& operator++() {
      value.next_child(bits);
      return *this;
    }
    bool operator!=(const iterator& other) { return bits != other.bits; }

    /// Provides the index of the child (0, 1, ..., Key::num_children-1) while iterating
    int index() const { return bits; }
  };
  iterator start, finish;

 public:
  explicit KeyChildren(const Key& key) : start(key.first_child(), 0), finish(key.last_child(), (1 << NDIM)) {}
  iterator begin() const { return start; }
  iterator end() const { return finish; }
};

/// Returns range object for iterating over children of a key
inline KeyChildren children(const Key& key) { return KeyChildren(key); }

inline std::ostream& operator<<(std::ostream& s, const Key& key) {
  s << "Key(" << key.level() << ",[";
  const auto& t = key.translation();
  for (Dimension d = 0; d < NDIM; ++d) s << (d ? "," : "") << t[d];
  s << "])";
  return s;
}

}  // namespace bh

namespace std {
/// Ensures Key satisfies std::hash protocol
template <>
struct hash<bh::Key> {
  std::size_t operator()(const bh::Key& s) const noexcept { return static_cast<std::size_t>(s.hash()); }
};
}  // namespace std
