#pragma once
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "robin_hood.h"

// A hash map partitioned into a fixed, power-of-two number of independent
// robin_hood tables ("shards"), selected by the same hash the map uses for
// its keys.
//
// Why: a single robin_hood table can't be filled concurrently even across
// disjoint keys (rehashes touch arbitrary slots), but each shard here is
// independent, so the fill parallelises one task per shard with no lock
// (see insert_shard_unsynchronized() below).
//
// The shard array is heap-allocated: an inline std::array at Shards=256
// would be ~14 kB, too big for a local on HPX's small default stacks.
template <class Key, class T, class Hash, std::size_t Shards = 256>
class ShardedMap {
    static_assert(Shards > 0 && (Shards & (Shards - 1)) == 0,
                  "Shards must be a power of two");

public:
    // ---- one-hash lookup support ------------------------------------------
    // A key bundled with its already-computed shard hash, handed to
    // robin_hood's transparent-lookup overloads so it isn't hashed again.
    struct PreHashed { Key k; std::size_t h; };

    // The Key overload just forwards to the user's Hash, so plain find(Key)
    // below still hashes exactly once.
    struct ShardHash {
        using is_transparent = void;
        std::size_t operator()(const Key& k)       const noexcept { return Hash{}(k); }
        std::size_t operator()(const PreHashed& p) const noexcept { return p.h; }
    };

    // Member-wise Key::operator== only, never memcmp - a padded key type
    // compares exactly as std::equal_to<Key> did before (see shardOf below).
    struct ShardEqual {
        using is_transparent = void;
        bool operator()(const Key& a, const Key& b)       const { return a == b; }
        bool operator()(const PreHashed& a, const Key& b) const { return a.k == b; }
        bool operator()(const Key& a, const PreHashed& b) const { return a == b.k; }
    };

    using Map = robin_hood::unordered_flat_map<Key, T, ShardHash, ShardEqual>;
    static_assert(Map::is_transparent,
        "shard tables must expose robin_hood's transparent-lookup overloads; "
        "without them every find()/lookup() would silently fall back to "
        "hashing twice, silently losing the point of PreHashed");

    using key_type    = Key;
    using mapped_type = T;
    using value_type  = typename Map::value_type;
    using size_type   = std::size_t;

    ShardedMap() : shards_(Shards) {}
    ShardedMap(const ShardedMap& o) : shards_(o.shards_) {}
    ShardedMap(ShardedMap&& o) noexcept : shards_(std::move(o.shards_)) {
        o.shards_.resize(Shards);   // a moved-from map stays a valid empty map
    }
    ShardedMap& operator=(const ShardedMap& o) {
        if (this != &o) shards_ = o.shards_;
        return *this;
    }
    ShardedMap& operator=(ShardedMap&& o) noexcept {
        if (this != &o) { shards_ = std::move(o.shards_); o.shards_.resize(Shards); }
        return *this;
    }

    template <bool IsConst>
    class Iterator {
        using ShardPtr = std::conditional_t<IsConst, const Map*, Map*>;
        using Inner    = std::conditional_t<IsConst, typename Map::const_iterator,
                                                      typename Map::iterator>;
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = typename Map::value_type;

        Iterator() = default;
        Iterator(ShardPtr s, std::size_t si, Inner it) : s_(s), si_(si), it_(it) { skip(); }

        // implicit iterator -> const_iterator conversion
        template <bool O, class = std::enable_if_t<IsConst && !O>>
        Iterator(const Iterator<O>& other) : s_(other.s_), si_(other.si_), it_(other.it_) {}

        decltype(auto) operator*()  const { return *it_; }
        decltype(auto) operator->() const { return it_.operator->(); }
        Iterator& operator++() { ++it_; skip(); return *this; }

        // At the sentinel (si_ == Shards) the inner iterators are default
        // constructed and must NOT be compared - robin_hood's default-
        // constructed Iter never compares equal to anything, including
        // itself's end().
        template <bool O>
        bool operator==(const Iterator<O>& o) const {
            return si_ == o.si_ && (si_ == Shards || it_ == o.it_);
        }
        template <bool O>
        bool operator!=(const Iterator<O>& o) const { return !(*this == o); }

        template <bool> friend class Iterator;

    private:
        void skip() {                       // advance past exhausted/empty shards
            while (si_ < Shards && it_ == s_[si_].end()) {
                if (++si_ < Shards) it_ = s_[si_].begin();
                else                it_ = Inner{};
            }
        }
        ShardPtr    s_  = nullptr;
        std::size_t si_ = Shards;
        Inner       it_{};
    };
    using iterator       = Iterator<false>;
    using const_iterator = Iterator<true>;

    iterator       begin()       { return iterator(shards_.data(), 0, shards_[0].begin()); }
    iterator       end()         { return iterator(shards_.data(), Shards, {}); }
    const_iterator begin() const { return const_iterator(shards_.data(), 0, shards_[0].begin()); }
    const_iterator end()   const { return const_iterator(shards_.data(), Shards, {}); }

    // ---- read path (hot: traversal.cpp / energy.cpp inner walks) ----
    // Bare-pointer fast path, avoiding the composite Iterator's overhead.
    const T* lookup(const Key& k) const {
        const std::size_t h  = Hash{}(k);
        const Map&        m  = shards_[h & (Shards - 1)];
        const auto        it = m.find(PreHashed{k, h});
        return it == m.end() ? nullptr : &it->second;
    }
    T* lookup(const Key& k) {
        const std::size_t h  = Hash{}(k);
        Map&              m  = shards_[h & (Shards - 1)];
        const auto        it = m.find(PreHashed{k, h});
        return it == m.end() ? nullptr : &it->second;
    }

    iterator find(const Key& k) {
        const std::size_t h = Hash{}(k);
        const std::size_t s = h & (Shards - 1);
        auto it = shards_[s].find(PreHashed{k, h});
        return it == shards_[s].end() ? end() : iterator(shards_.data(), s, it);
    }
    const_iterator find(const Key& k) const {
        const std::size_t h = Hash{}(k);
        const std::size_t s = h & (Shards - 1);
        auto it = shards_[s].find(PreHashed{k, h});
        return it == shards_[s].end() ? end() : const_iterator(shards_.data(), s, it);
    }
    std::size_t count(const Key& k) const { return lookup(k) ? 1u : 0u; }
    const T& at(const Key& k) const {
        const T* p = lookup(k);
        if (!p) throw std::out_of_range("ShardedMap::at: key not found");
        return *p;
    }
    T& at(const Key& k) {
        T* p = lookup(k);
        if (!p) throw std::out_of_range("ShardedMap::at: key not found");
        return *p;
    }
    // Hashes twice (shardOf(), then again inside robin_hood) - fine, these
    // are fill/merge paths, not the hot walk.
    T& operator[](const Key& k) { return shards_[shardOf(k)][k]; }

    bool        empty() const { for (const Map& m : shards_) if (!m.empty()) return false; return true; }
    std::size_t size()  const { std::size_t n = 0; for (const Map& m : shards_) n += m.size(); return n; }

    static constexpr std::size_t shard_count = Shards;
    static std::size_t per_shard_reserve(std::size_t n) { return (n + Shards - 1) / Shards; }

    // Exactly ceil(n/Shards), not n/Shards plus a margin: robin_hood rounds
    // a reserve up to a power-of-two capacity, so a margin that crosses one
    // would double this shard's memory.
    void reserve(std::size_t n) {
        const std::size_t per = per_shard_reserve(n);
        for (Map& m : shards_) m.reserve(per);
    }
    // Exposed so a caller can parallelise the Shards allocations itself.
    void reserve_shard(std::size_t s, std::size_t n) { shards_[s].reserve(n); }

    template <class It>
    void insert(It first, It last) {   // no-overwrite, like std/robin_hood
        for (; first != last; ++first) shards_[shardOf(first->first)].insert(*first);
    }

    // ---- concurrent fill (the point of this class) ----
    // UNSYNCHRONIZED insert into shard `s` - safe only when no other thread
    // touches this shard concurrently (see buildOctreeBottomUp's
    // scatter-then-merge fill, which guarantees exactly that).
    void insert_shard_unsynchronized(std::size_t s, const Key& k, const T& v) {
        shards_[s].insert_or_assign(k, v);
    }

    // Lets a caller partitioning its own data compute the destination shard.
    static std::size_t shard_of(const Key& k) { return shardOf(k); }

private:
    // Must call Hash{}(k), never hash the key's raw bytes: OctreeKey has
    // indeterminate padding, and an earlier raw-byte version let identical
    // keys land in different shards, corrupting the tree.
    static std::size_t shardOf(const Key& k) { return Hash{}(k) & (Shards - 1); }

    std::vector<Map> shards_;
};
