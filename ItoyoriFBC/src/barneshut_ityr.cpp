#define ITYR_COPY_FUTURE_DATA 1

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

#include "ityr/ityr.hpp"
#include "barneshut_common.hpp"
#include "barneshut_simd.hpp"

using namespace std;
using Clock = chrono::steady_clock;
using TimePoint = chrono::time_point<Clock>;

// Body without the accel fields: the parallel path keeps walk results in the
// per-node accel buffer, not in the body (only --brute-force materializes a
// full copy). id + 7 doubles is exactly one cache line. The shared readers are
// templated over the body struct.
struct Body64 {
  long long id;
  double mass, x, y, z, vx, vy, vz;
};

// Wall-clock phase timers. A task that suspends in wait()/get() may migrate to
// another process, so a timer interval spanning a suspension would mix clock
// bases between ranks. Every phase boundary here is a barrier-synchronized
// SPMD point, so each rank accumulates only local steady-clock deltas between
// boundaries and the reported value is the MPI_MAX across ranks (the critical
// path). Master-only sub-phases without suspension points use add_local.
// Barriers are skipped under --no-timers.
struct PhaseTimer {
  double ms = 0;
  TimePoint last{};
  void start() {
    if (cfg.timers) {
      MPI_Barrier(MPI_COMM_WORLD);
      last = Clock::now();
    }
  }
  void stop() {
    if (cfg.timers) ms += ms_since(last);
  }
  void add_local(TimePoint from) {
    if (cfg.timers) ms += ms_since(from);
  }
};
inline PhaseTimer w_input, w_build, w_forces, w_integrate, w_output, w_total;

#ifdef ITYR_PROF
// Profiler instrumentation, compiled only into barneshut_ityr_prof.cpp (the
// same source with ITYR_PROF defined).
inline PhaseTimer t_pkeys, t_pdist, t_ptree, t_pcompute, t_pflush, t_pcombine, t_penergy,
    t_pkloop, t_pkflush;
inline PhaseTimer t_pagather, t_pmerge_all, t_palltoall, t_palltoallv;
inline PhaseTimer t_psearch, t_pnmerge;
inline PhaseTimer t_pk1, t_pk2, t_pk3, t_pk4;
inline PhaseTimer t_splan, t_sfold, t_sasm;
inline PhaseTimer t_pcompact;
#endif

// Parallel Barnes-Hut on ItoyoriFBC (mirrors barneshut.cpp).
// Dataflow per step: kick+drift -> slice keys -> redistribute -> tree -> forces -> kick.
// Particles are distributed: each node owns a fixed slice of the global sorted
// order and only ever holds those bodies; the master transiently holds the
// gathered (key, body) pairs during the build. The tree and the compact bodies
// are derived structures, so they are replicated per node.
// Futures are the only channel. wait()/get() suspend until the future is ready
// and may migrate the task to another rank (get() on an already-ready future
// is a fast path that never suspends), so pointers obtained from get() are
// short-lived borrows valid only on the rank that obtained them. Capture only
// initialized values; the consumer of a buffer deallocates what get() returned.

struct ForceResult {
  double ax, ay, az, pot;
};

struct BuildResult {
  Node* tree;
  size_t tree_bytes;
  double box_side, xmin, ymin, zmin;
};
// ---------------- distributed-particle layout ----------------
// Fixed per-node slices of the global sorted order: slice s = [los[s], los[s+1])
// with los[s] = floor(s*n/(ninter*gran)) * gran, gran = cfg.build_cutoff
// (constant across steps, so every body pointer is broadcast once at init).
// Each node sorts its own slice; the master's merge of the per-node runs
// yields the exact global (key, idx)-ordered sequence.
constexpr int MAX_INTER_NODES = 64;

struct SliceLayout {
  size_t n = 0;
  size_t ninter = 0;
  size_t nslice = 0;      // this node's slice length (0 possible for tiny n / many nodes)
  size_t slice_lo = 0;    // this node's slice start in the global sorted order
  size_t los[MAX_INTER_NODES + 1] = {};   // los[s] per node + sentinel los[ninter] = n
};

inline void compute_slice_layout(size_t n, size_t ninter, size_t gran, SliceLayout& L) {
  L.n = n;
  L.ninter = ninter;
  for (size_t s = 0; s <= ninter; ++s)
    L.los[s] = (s * n / (ninter * gran)) * gran;
  L.los[ninter] = n;   // sentinel: the last slice ends at n, not at the rounded bound
  int inter = ityr::common::topology::inter_my_rank();
  L.slice_lo = L.los[inter];
  L.nslice = (inter == (int)ninter - 1) ? n - L.los[inter] : L.los[inter + 1] - L.los[inter];
}

// The (key, body) pair gathered from each node's sorted run to the master (72 B/body).
struct GatherPair {
  uint64_t key;
  Body64 body;
};

// The alltoallv payload: the body + its exact global position (the merged
// (key, idx)-order position recorded during the per-node merge). The receiver
// scatter-writes each element straight into the live slice at that position —
// no merge, no key comparison.
struct PosPair {
  uint32_t pos;
  Body64 body;
};

// Master-side per-step build buffers (persistent allocations, reused every
// step; only the master's intra-0 touches them). keys doubles as the master's
// keys replica; bodies is the master's full sorted copy, gathered on demand at
// output/brute-force steps; compact is the master's replica (its own slice
// folded during the redistribute); vel_scratch is the energy gather.
struct MasterBuild {
  uint64_t* keys = nullptr;         // 8 B * n (also the master's keys replica)
  Body64* bodies = nullptr;         // 64 B * n
  CompactBody* compact = nullptr;   // 32 B * n
  double* vel_scratch = nullptr;    // 32 B * n (energy gather: mass,vx,vy,vz)
  bool owns = false;
};

// Everything the tree build needs, captured by value into closures (task
// migration byte-copies the stack frame, so all members are inline fixed-size
// arrays — no heap pointers that could dangle after a steal). Each node holds
// its own replica of the merged keys and the compact bodies; the pool is
// mapped at identical addresses only within a node, so a task must read the
// replica of the node that executes it. Every replica holds identical bytes,
// so a fold reads the executing node's copy and the subtree is identical
// everywhere.
struct BuildCtx {
  CompactBody* compact[MAX_INTER_NODES];
  uint64_t* keys[MAX_INTER_NODES];       // per-node merged-keys replicas (identical content)
};

// ---------------- geometry helpers for the fused-bounds path ----------------
struct Extremes {
  double xmin, xmax, ymin, ymax, zmin, zmax;
};
inline Bounds bounds_from_extremes(const Extremes& e) {
  double box_side = max({e.xmax - e.xmin, e.ymax - e.ymin, e.zmax - e.zmin});
  if (box_side <= 0) box_side = 1.0;
  return {e.xmin, e.ymin, e.zmin, 1.0 / box_side, box_side};
}

// ---------------- parallel phase tasks ----------------

// Parallel for with the binary range splitting the scheduler tolerates (the
// only task idiom: spawn + wait_all). Leaves write their results in place and
// return nothing, so there is no per-leaf buffer and no concat. The splits are
// chunk-aligned: every leaf covers exactly one chunk [c*cutoff, (c+1)*cutoff)
// except the last partial one, which makes chunk ownership unambiguous (each
// chunk is computed by exactly one leaf, hence on exactly one node) — what the
// combine's chunk-bitmask exchange relies on.
template <typename LeafFn>
int par_for(size_t lo, size_t hi, size_t cutoff, LeafFn leaf) {
  size_t n = hi - lo;
  if (n <= cutoff) {
    leaf(lo, hi);
    return 0;
  }
  // Aligned midpoint, always strictly inside (lo, hi): at least one full chunk past lo.
  size_t mid = lo + std::max(cutoff, (n / 2 / cutoff) * cutoff);
  auto lf = ityr::spawn<int>([=] { return par_for(lo, mid, cutoff, leaf); });
  auto rf = ityr::spawn<int>([=] { return par_for(mid, hi, cutoff, leaf); });
  ityr::ito::wait_all(lf, rf);
  return 0;
}

// Broadcast a per-node uintptr table: each node's intra-0 wrote its slot,
// MPI_MAX picks the one non-zero entry per node (fixed MAX_INTER_NODES count —
// nodes can be unequal), and the collective also orders every replica's writes
// before the next step reads them. Every rank ends up with the same table.
void broadcast_uintptr_table(uintptr_t* tab) {
  collective_timed_cat(tc_ptr, [&] {
    ityr::common::mpi_allreduce_inplace(tab, MAX_INTER_NODES, MPI_COMM_WORLD, MPI_MAX);
  });
}

// ---------------- distributed build: keys + sorted runs ----------------

// Stable LSD radix sort on the 64-bit key: 16-bit passes over ping-pong
// buffers. Stability keeps equal keys in input order (the ascending global
// idx), so the output is exactly the (key, idx) order.
inline void radix_sort_keypairs(KeyPair* a, size_t n) {
  static thread_local vector<KeyPair> buf;
  static thread_local vector<size_t> cnt(65536);
  buf.resize(n);
  KeyPair* src = a, *dst = buf.data();
  for (int shift = 0; shift < 64; shift += 16) {
    size_t* c = cnt.data();
    std::fill(c, c + 65536, (size_t)0);
    for (size_t i = 0; i < n; ++i) ++c[(src[i].first >> shift) & 0xffff];
    size_t off = 0;
    for (int k = 0; k < 65536; ++k) {
      size_t t = c[k];
      c[k] = off;
      off += t;
    }
    for (size_t i = 0; i < n; ++i) {
      const KeyPair e = src[i];
      dst[c[(e.first >> shift) & 0xffff]++] = e;
    }
    std::swap(src, dst);
  }
  if (src != a) std::copy(src, src + n, a);
}

// SPMD, all intra ranks. The node's slice is a run of cfg.build_cutoff-sized
// blocks in global index order. Each intra rank sorts its share of the blocks
// (radix_sort_keypairs), merges its position-ordered sorted blocks into one
// rank-run and, from the rank runs, emits its own chunk of the node's final
// sorted run. run/pairs are the node's persistent scratch buffers.
void slice_keys_phase(const SliceLayout& L, Body64* live, const Bounds& bd,
                      KeyPair* run, GatherPair* pairs) {
#ifdef ITYR_PROF
  t_pkeys.start();
#endif
  auto& topo = ityr::common::topology::instance::get();
  const int intra = topo.intra_my_rank();
  const int nintra = topo.intra_n_ranks();
  const size_t lo = L.slice_lo, nslice = L.nslice;
  const size_t gran = cfg.build_cutoff;      // slice granularity = block size
  const size_t nblocks = (nslice + gran - 1) / gran;
  const size_t b0 = (size_t)intra * nblocks / nintra, b1 = (size_t)(intra + 1) * nblocks / nintra;
#ifdef ITYR_PROF
  TimePoint t0k = Clock::now();
#endif
  for (size_t b = b0; b < b1; ++b) {
    const size_t bl = b * gran, bh = std::min(bl + gran, nslice);
    for (size_t j = bl; j < bh; ++j) {
      const Body64& bd2 = live[j];
      run[j] = {make_key(bd2.x, bd2.y, bd2.z, bd.xmin, bd.ymin, bd.zmin, bd.inv), (int)(lo + j)};
    }
    radix_sort_keypairs(run + bl, bh - bl);
  }
#ifdef ITYR_PROF
  t_pk1.add_local(t0k);
  t0k = Clock::now();
#endif
  // Merge this rank's position-ordered sorted blocks into one rank-run (a
  // k-way merge into a KeyPair view of the pairs buffer; lowest-block-wins
  // ties = the stable global order).
  {
    const size_t nb = b1 - b0;
    const size_t lb = (size_t)b0 * gran, hb = std::min((size_t)b1 * gran, nslice);
    if (nb > 0) {
      vector<size_t> pidx(nb);
      vector<uint64_t> hk(nb);
      vector<int> hblk(nb);
      size_t hn = 0;
      auto hpush = [&](uint64_t k, int b) {
        size_t i = hn++;
        while (i > 0) {
          size_t par = (i - 1) / 2;
          if (hk[par] < k || (hk[par] == k && hblk[par] <= b)) break;
          hk[i] = hk[par];
          hblk[i] = hblk[par];
          i = par;
        }
        hk[i] = k;
        hblk[i] = b;
      };
      auto hpop = [&]() -> int {
        int b = hblk[0];
        uint64_t kk = hk[--hn];
        int bb = hblk[hn];
        size_t i = 0;
        for (;;) {
          size_t c = 2 * i + 1;
          if (c >= hn) break;
          if (c + 1 < hn && (hk[c + 1] < hk[c] || (hk[c + 1] == hk[c] && hblk[c + 1] < hblk[c]))) ++c;
          if (kk < hk[c] || (kk == hk[c] && bb <= hblk[c])) break;
          hk[i] = hk[c];
          hblk[i] = hblk[c];
          i = c;
        }
        hk[i] = kk;
        hblk[i] = bb;
        return b;
      };
      KeyPair* out = (KeyPair*)pairs + lb;
      for (size_t bi = 0; bi < nb; ++bi) {
        const size_t bl = (b0 + bi) * gran, bh = std::min(bl + gran, nslice);
        pidx[bi] = bl;
        if (bl < bh) hpush(run[bl].first, (int)bi);
      }
      for (size_t j = lb; j < hb; ++j) {
        const size_t bi = (size_t)hpop();
        const size_t bl = (b0 + bi) * gran, bh = std::min(bl + gran, nslice);
        out[j - lb] = run[pidx[bi]];
        if (++pidx[bi] < bh) hpush(run[pidx[bi]].first, (int)bi);
      }
    }
  }
#ifdef ITYR_PROF
  t_pk2.add_local(t0k);
  t0k = Clock::now();
#endif
  collective_timed_cat(tc_bar, [&] { ityr::common::mpi_barrier(topo.intra_mpicomm()); });
#ifdef ITYR_PROF
  t0k = Clock::now();
#endif
  // Each rank computes its own output chunk [j0, j1) of the final sorted run
  // directly from the nintra rank-runs. The chunk's boundary keys come from an
  // exact value-space binary search (the last v in [0, 2^64] with count(<v)
  // <= j), and the equal-key tails split by run order (= position order, the
  // stable merge's tie rule). One k-way with a (key, run) min-heap emits the
  // chunk into `run`.
  {
    const size_t j0 = (size_t)intra * nslice / nintra, j1 = (size_t)(intra + 1) * nslice / nintra;
    const KeyPair* rp = (KeyPair*)pairs;
    auto rlo = [&](size_t r) {
      return std::min((size_t)r * nblocks / nintra * gran, nslice);
    };
    auto rhi = [&](size_t r) {
      return std::min((size_t)(r + 1) * nblocks / nintra * gran, nslice);
    };
    auto bykey = [](const KeyPair& e, uint64_t k) { return e.first < k; };
    auto byval = [](uint64_t k, const KeyPair& e) { return k < e.first; };
    auto count_lt = [&](uint64_t v) {
      size_t c = 0;
      for (size_t r = 0; r < (size_t)nintra; ++r)
        c += (size_t)(std::lower_bound(rp + rlo(r), rp + rhi(r), v, bykey) - (rp + rlo(r)));
      return c;
    };
    auto select = [&](size_t j) -> uint64_t {
      unsigned __int128 lo = 0, hi = (unsigned __int128)1 << 64;
      while (lo < hi) {
        unsigned __int128 mid = lo + (hi - lo) / 2;
        if (count_lt((uint64_t)mid) <= j) lo = mid + 1;
        else hi = mid;
      }
      return (uint64_t)(lo - 1);
    };
    const uint64_t vlo = select(j0), vhi = select(j1);
    const size_t skip_lo = j0 - count_lt(vlo);   // the key==vlo elements before the chunk
    const size_t keep_hi = j1 - count_lt(vhi);   // the key==vhi elements inside the chunk
    vector<size_t> alo((size_t)nintra), ahi((size_t)nintra);
    size_t cum = 0;
    for (size_t r = 0; r < (size_t)nintra; ++r) {
      const KeyPair* b = rp + rlo(r), *e = rp + rhi(r);
      const size_t lb = (size_t)(std::lower_bound(b, e, vlo, bykey) - b);
      const size_t ec = (size_t)(std::upper_bound(b, e, vlo, byval) - b) - lb;
      alo[r] = lb + std::min(ec, skip_lo > cum ? skip_lo - cum : 0);
      cum += ec;
    }
    cum = 0;
    for (size_t r = 0; r < (size_t)nintra; ++r) {
      const KeyPair* b = rp + rlo(r), *e = rp + rhi(r);
      const size_t ub = (size_t)(std::lower_bound(b, e, vhi, bykey) - b);
      const size_t ec = (size_t)(std::upper_bound(b, e, vhi, byval) - b) - ub;
      ahi[r] = ub + std::min(ec, keep_hi > cum ? keep_hi - cum : 0);
      cum += ec;
    }
    // the k-way emission (a (key, run) min-heap = the stable (key, position) order)
    vector<uint64_t> hk((size_t)nintra);
    vector<int> hr((size_t)nintra);
    size_t hn = 0;
    auto hpush = [&](uint64_t k, int r) {
      size_t i = hn++;
      while (i > 0) {
        size_t par = (i - 1) / 2;
        if (hk[par] < k || (hk[par] == k && hr[par] <= r)) break;
        hk[i] = hk[par];
        hr[i] = hr[par];
        i = par;
      }
      hk[i] = k;
      hr[i] = r;
    };
    auto hpop = [&]() -> int {
      int r = hr[0];
      uint64_t kk = hk[--hn];
      int rr = hr[hn];
      size_t i = 0;
      for (;;) {
        size_t c = 2 * i + 1;
        if (c >= hn) break;
        if (c + 1 < hn && (hk[c + 1] < hk[c] || (hk[c + 1] == hk[c] && hr[c + 1] < hr[c]))) ++c;
        if (kk < hk[c] || (kk == hk[c] && rr <= hr[c])) break;
        hk[i] = hk[c];
        hr[i] = hr[c];
        i = c;
      }
      hk[i] = kk;
      hr[i] = rr;
      return r;
    };
    for (size_t r = 0; r < (size_t)nintra; ++r)
      if (alo[r] < ahi[r]) hpush(rp[rlo(r) + alo[r]].first, (int)r);
    for (size_t o = 0; o < j1 - j0; ++o) {
      const size_t r = (size_t)hpop();
      run[j0 + o] = rp[rlo(r) + alo[r]];
      if (++alo[r] < ahi[r]) hpush(rp[rlo(r) + alo[r]].first, (int)r);
    }
  }
  KeyPair* src = run;
  // The materialization below writes the GatherPairs into pairs[j0..j1) (sorted positions),
  // which overlaps other ranks' rank-runs (position ranges) — order it after all the chunk
  // merges have finished reading the rank-runs.
  collective_timed_cat(tc_bar, [&] { ityr::common::mpi_barrier(topo.intra_mpicomm()); });
#ifdef ITYR_PROF
  t_pk3.add_local(t0k);
  t0k = Clock::now();
#endif
  // Materialize the gather pairs (key + body at the sorted position — a random
  // permutation read of live).
  const size_t j0 = (size_t)intra * nslice / nintra, j1 = (size_t)(intra + 1) * nslice / nintra;
  for (size_t j = j0; j < j1; ++j) {
    if (j + 40 < j1) __builtin_prefetch(&live[src[j + 40].second - lo]);
    pairs[j].key = src[j].first;
    pairs[j].body = live[src[j].second - lo];
  }
  // The fused pass in redistribute_build writes the live (bodies into the
  // sorted order) while the materialization above reads the live at the old
  // positions (a cross-rank random permutation) — the barrier orders the reads
  // before the writes.
  collective_timed_cat(tc_bar, [&] { ityr::common::mpi_barrier(topo.intra_mpicomm()); });
#ifdef ITYR_PROF
  t_pk4.add_local(t0k);
  t_pkeys.stop();
#endif
}

// (key, source run, source position) triple carried through the pairwise merge
// rounds (16 B, padded).
struct MergeTriple { uint64_t k; uint32_t p; uint16_t s; uint16_t pad; };

// Intra-parallel merge of the gathered key runs into the full keys replica,
// recording each node's own elements' global positions (mypos). ninter==2:
// every intra rank merges its output chunk (the input split per chunk found by
// a binary search on the equal-key rule — r0 wins the ties, matching
// std::merge). ninter>2: bottom-up pairwise stable merge (see below).
void merge_keys_parallel(const SliceLayout& L, const uint64_t* gathered, uint64_t* out,
                         uint32_t* mypos, MergeTriple* r_mbufA,
                         MergeTriple* r_mbufB) {
  auto& topo = ityr::common::topology::instance::get();
  const int intra = topo.intra_my_rank();
  const int nintra = topo.intra_n_ranks();
  const int inter = topo.inter_my_rank();
  if (L.ninter == 2) {
    const uint64_t* r0 = gathered + L.los[0];
    const uint64_t* r1 = gathered + L.los[1];
    const size_t n0 = L.los[1] - L.los[0], n1 = L.los[2] - L.los[1];
    const size_t n = n0 + n1;
    const size_t o0 = (size_t)intra * n / nintra, o1 = (size_t)(intra + 1) * n / nintra;
    // the input split for o0 (the merge rule: r1[i1] is taken only when strictly smaller).
    // The first mid with r1[o0-mid-1] < r0[mid] is the valid split (A is false for mid below
    // the split and true from it on).
    size_t lo = (o0 > n1) ? o0 - n1 : 0, hi = (o0 < n0) ? o0 : n0;
    while (lo < hi) {
      size_t mid = lo + (hi - lo) / 2;
      size_t jm = o0 - mid;
      bool ok = (jm == 0) || (r1[jm - 1] < r0[mid]);
      if (ok) hi = mid; else lo = mid + 1;
    }
    size_t i = lo, j = o0 - lo, o = o0;
    while (o < o1) {
      if (j < n1 && (i >= n0 || r1[j] < r0[i])) {
        out[o] = r1[j];
        if (inter == 1) mypos[j] = (uint32_t)o;
        ++j;
      } else {
        out[o] = r0[i];
        if (inter == 0) mypos[i] = (uint32_t)o;
        ++i;
      }
      ++o;
    }
    return;
  }
  // ninter > 2: bottom-up pairwise stable merge. Round 1 merges run pairs
  // (0,1), (2,3), ...; each later round merges consecutive group results; an
  // odd trailing group passes through against an empty partner. The tie rule at
  // every merge is left-group-first, and groups always carry ascending sender
  // ranges, so the composed order is exactly the lexicographic (key, sender,
  // pos) a priority-queue merge would produce. All merges are intra-parallel
  // (output chunks), and the (key, src, src_pos) triples carried through the
  // rounds let the final pass record mypos.
  {
    const size_t ntot = L.los[L.ninter];   // the sentinel is the global n
    MergeTriple* cur = r_mbufA;
    size_t goff[MAX_INTER_NODES + 1];
    int g = L.ninter;
    for (int s = 0; s <= g; ++s) goff[s] = L.los[s];   // round-1 groups = the gathered runs
    const int nintra = topo.intra_n_ranks();
    int rounds = 0;
    while (g > 1) {
      MergeTriple* out3 = (rounds == 0) ? r_mbufA : (cur == r_mbufA ? r_mbufB : r_mbufA);
      size_t doff = 0;
      int ng = 0;   // groups after this round
      size_t ngoff[MAX_INTER_NODES + 1];
      ngoff[0] = 0;
      for (int p = 0; 2 * p < g; ++p) {
        int a = 2 * p, b = 2 * p + 1;
        size_t pa, la, pb, lb;
        if (rounds == 0) {
          pa = L.los[a]; la = L.los[a + 1] - L.los[a];
          pb = (b < g) ? L.los[b] : 0; lb = (b < g) ? L.los[b + 1] - L.los[b] : 0;
        } else {
          pa = goff[a]; la = goff[a + 1] - goff[a];
          pb = (b < g) ? goff[b] : 0; lb = (b < g) ? goff[b + 1] - goff[b] : 0;
        }
        const size_t tot = la + lb;
        // this intra rank's chunk of the pair's output range
        const size_t c0 = doff + (size_t)intra * tot / nintra;
        const size_t c1 = doff + (size_t)(intra + 1) * tot / nintra;
        // split: how many elements of the left group precede c0 (left wins ties)
        const size_t oloc = c0 - doff;
        size_t lo = (oloc > lb) ? oloc - lb : 0, hi = (oloc < la) ? oloc : la;
        while (lo < hi) {
          size_t mid = lo + (hi - lo) / 2;
          size_t jm = oloc - mid;
          bool ok = (jm == 0);
          if (rounds == 0) ok = ok || (gathered[L.los[b] + jm - 1] < gathered[L.los[a] + mid]);
          else ok = ok || (cur[pb + jm - 1].k < cur[pa + mid].k);
          if (ok) hi = mid; else lo = mid + 1;
        }
        size_t i = lo, j = oloc - lo, o = c0;
        if (rounds == 0) {
          const uint64_t* ra = gathered + pa;
          const uint64_t* rb = (b < g) ? gathered + pb : nullptr;
          while (o < c1) {
            if (j < lb && (i >= la || rb[j] < ra[i])) {
              out3[o] = {rb[j], (uint32_t)j, (uint16_t)b, 0}; ++j;
            } else {
              out3[o] = {ra[i], (uint32_t)i, (uint16_t)a, 0}; ++i;
            }
            ++o;
          }
        } else {
          while (o < c1) {
            if (j < lb && (i >= la || cur[pb + j].k < cur[pa + i].k)) {
              out3[o] = cur[pb + j]; ++j;
            } else {
              out3[o] = cur[pa + i]; ++i;
            }
            ++o;
          }
        }
        doff += tot;
        ngoff[++ng] = doff;
      }
      // Ordering: the next round reads the full output of this round (its
      // input chunks span other ranks' output chunks), so an intra barrier
      // between rounds is required.
      collective_timed_cat(tc_bar, [&] { ityr::common::mpi_barrier(topo.intra_mpicomm()); });
      cur = out3;
      for (int s2 = 0; s2 <= ng; ++s2) goff[s2] = ngoff[s2];
      g = ng;
      ++rounds;
    }
    // final buffer = cur: copy the keys into `out` and record mypos for this node's elements
    const size_t c0 = (size_t)intra * ntot / nintra, c1 = (size_t)(intra + 1) * ntot / nintra;
    for (size_t o = c0; o < c1; ++o) {
      out[o] = cur[o].k;
      if (cur[o].s == (uint16_t)inter) mypos[cur[o].p] = (uint32_t)o;
    }
  }
}

// SPMD redistribution after the keys phase:
//   1. Allgather each node's sorted keys (8 B/body) so every node holds the
//      full keys set.
//   2. Intra-parallel merge producing the full keys replica and each node's
//      own elements' exact global positions (mypos).
//   3. A monotonic histogram of the positions into per-destination send counts
//      + an in-place pack (the position rides in the pair's key field, so the
//      send payload is the (pos, body) pair).
//   4. MPI_Alltoallv of the packed payload: for a stationary distribution each
//      node's run is its own slice, so only boundary-crossers actually move.
//   5. A straight scatter-write into the live slice: every received element
//      lands at its exact position, so each slice position is written exactly
//      once and no merge is needed.
//   6. Per-node compact of the node's own slice.
void redistribute_build(const SliceLayout& L, GatherPair* pairs, KeyPair* run,
                        uint64_t* krun, Body64* live, uint64_t* keys_rep, uint64_t* gkeys,
                        GatherPair* recv, uint32_t* mypos, CompactBody* compact,
                        MergeTriple* mbufA,
                        MergeTriple* mbufB) {
  auto& topo = ityr::common::topology::instance::get();
  const int intra = topo.intra_my_rank();
  const int nintra = topo.intra_n_ranks();
  const int ninter = topo.inter_n_ranks();
  const size_t nslice = L.nslice, slice_lo = L.slice_lo;
#ifdef ITYR_PROF
  t_pdist.start();
#endif
  if (ninter == 1) {
#ifdef ITYR_PROF
    TimePoint t0n = Clock::now();
#endif
    // Fast path for one node: the fully sorted run already lives in the
    // node-wide shm, a sorted element's position is its index, and the only
    // node owns every element — the gather, merge, histogram and exchange are
    // all no-ops. One fused local pass per rank: copy the chunk of keys into
    // the replica (the tree's range scans need contiguous uint64s) and
    // scatter-write the chunk's bodies into the live in sorted order
    // (pairs[j] = the element at sorted position j, materialized by the keys
    // phase). Zero collectives, zero barriers (every read/write here is the
    // rank's own chunk).
    const size_t j0 = (size_t)intra * nslice / nintra, j1 = (size_t)(intra + 1) * nslice / nintra;
    for (size_t j = j0; j < j1; ++j) {
      keys_rep[j] = run[j].first;
      live[j - slice_lo] = pairs[j].body;
    }
#ifdef ITYR_PROF
    t_pnmerge.add_local(t0n);
#endif
  } else {
  // 1. Allgather the sorted keys so every node has the full set.
  if (intra == 0) {
    for (size_t j = 0; j < nslice; ++j) krun[j] = run[j].first;   // keep the key only
  }
  int kcnts[MAX_INTER_NODES] = {}, kdspls[MAX_INTER_NODES] = {};
  for (int s = 0; s < ninter; ++s) {
    kcnts[s] = (int)((L.los[s + 1] - L.los[s]) * sizeof(uint64_t));
    kdspls[s] = (int)(L.los[s] * sizeof(uint64_t));
  }
  if (intra == 0) {
#ifdef ITYR_PROF
    TimePoint t0g = Clock::now();   // intra-0-only scope: local delta (no barrier possible)
#endif
    collective_timed_cat(tc_dist, [&] {
      MPI_Allgatherv(krun, (int)(nslice * sizeof(uint64_t)), MPI_BYTE,
                     gkeys, kcnts, kdspls, MPI_BYTE, topo.inter_mpicomm());
    });
#ifdef ITYR_PROF
    t_pagather.add_local(t0g);
#endif
  }
  // The intra peers' merge reads the gathered keys — order it after the
  // leader's Allgatherv.
  collective_timed_cat(tc_bar, [&] { ityr::common::mpi_barrier(topo.intra_mpicomm()); });
  // 2. Every rank merges its output chunk of the full keys replica and records
  //    its own run elements' exact global positions (mypos).
#ifdef ITYR_PROF
  TimePoint t0m = Clock::now();
#endif
  merge_keys_parallel(L, gkeys, keys_rep, mypos, mbufA, mbufB);
#ifdef ITYR_PROF
  t_pmerge_all.add_local(t0m);
#endif
  // The mypos entries of a rank's run chunk may be written by other ranks'
  // merge chunks (the node's own elements are scattered across the output):
  // order before the histogram.
  collective_timed_cat(tc_bar, [&] { ityr::common::mpi_barrier(topo.intra_mpicomm()); });
  // 3. The positions drive the destination split (a monotonic walk per run
  //    chunk — the positions increase along the run) and the in-place pack (the
  //    position rides in the pair's key field). All intra ranks.
  int scnts[MAX_INTER_NODES] = {}, sdspls[MAX_INTER_NODES] = {};
  int rcnts[MAX_INTER_NODES] = {}, rdspls[MAX_INTER_NODES] = {};
  {
#ifdef ITYR_PROF
    TimePoint t0s = Clock::now();
#endif
    int lscnts[MAX_INTER_NODES] = {};
    const size_t j0 = (size_t)intra * nslice / nintra, j1 = (size_t)(intra + 1) * nslice / nintra;
    for (size_t j = j0; j < j1; ++j) {
      uint32_t p = mypos[j];
      int d = 0;
      while (d < (int)ninter - 1 && p >= (uint32_t)L.los[d + 1]) ++d;
      ++lscnts[d];
      pairs[j].key = p;   // the pack
    }
    collective_timed_cat(tc_dist, [&] {
      MPI_Reduce(lscnts, scnts, ninter, MPI_INT, MPI_SUM, 0, topo.intra_mpicomm());
    });
#ifdef ITYR_PROF
    t_psearch.add_local(t0s);
#endif
  }
  // 4. The leader exchanges the packed payload (ninter==1: nothing to exchange
  //    — the scatter below reads the local pairs directly).
  if (intra == 0 && ninter > 1) {
#ifdef ITYR_PROF
    TimePoint t0a = Clock::now();
#endif
    collective_timed_cat(tc_dist, [&] { MPI_Alltoall(scnts, 1, MPI_INT, rcnts, 1, MPI_INT,
                                        topo.inter_mpicomm()); });
#ifdef ITYR_PROF
    t_palltoall.add_local(t0a);
#endif
    size_t off = 0;
    for (int s = 0; s < ninter; ++s) {
      rdspls[s] = (int)off;
      off += (size_t)rcnts[s];
    }
    assert(off == nslice && "alltoallv counts must partition the slice");
    int bscnts[MAX_INTER_NODES] = {}, bsdspls[MAX_INTER_NODES] = {};
    int brcnts[MAX_INTER_NODES] = {}, brdspls[MAX_INTER_NODES] = {};
    sdspls[0] = 0;
    for (int s = 1; s < ninter; ++s) sdspls[s] = sdspls[s - 1] + scnts[s - 1];
    for (int s = 0; s < ninter; ++s) {
      bscnts[s] = scnts[s] * (int)sizeof(GatherPair);
      bsdspls[s] = sdspls[s] * (int)sizeof(GatherPair);
      brcnts[s] = rcnts[s] * (int)sizeof(GatherPair);
      brdspls[s] = rdspls[s] * (int)sizeof(GatherPair);
    }
#ifdef ITYR_PROF
    TimePoint t0v = Clock::now();
#endif
    collective_timed_cat(tc_dist, [&] {
      MPI_Alltoallv(pairs, bscnts, bsdspls, MPI_BYTE,
                    recv, brcnts, brdspls, MPI_BYTE, topo.inter_mpicomm());
    });
#ifdef ITYR_PROF
    t_palltoallv.add_local(t0v);
#endif
  }
  // The intra peers read the received buffer (or the packed pairs) after the leader's
  // exchange completes.
  collective_timed_cat(tc_bar, [&] { ityr::common::mpi_barrier(topo.intra_mpicomm()); });
  if (nslice > 0) {
    // 5. Intra-parallel scatter-write: every (pos, body) lands at its exact
    //    position in the live slice; each slice position is written exactly
    //    once, and the per-rank shares of the payload are disjoint.
#ifdef ITYR_PROF
    TimePoint t0n = Clock::now();
#endif
    const PosPair* rp = (const PosPair*)recv;
    const size_t r0 = (size_t)intra * nslice / nintra, r1 = (size_t)(intra + 1) * nslice / nintra;
    for (size_t k = r0; k < r1; ++k) {
      const PosPair& e = rp[k];
      live[e.pos - slice_lo] = e.body;
    }
#ifdef ITYR_PROF
    t_pnmerge.add_local(t0n);
#endif
  }
  // The live slice has no RMA readers (the tree folds read the
  // locally-derived compact; the gather/output paths are two-sided). The intra
  // barrier below orders the scatter's shm writes before the compact's reads
  // (POSIX shm is coherent; the barrier is the ordering).
  collective_timed_cat(tc_bar, [&] { ityr::common::mpi_barrier(topo.intra_mpicomm()); });
  }
  // 6. Per-node compact of the node's own slice.
  {
#ifdef ITYR_PROF
    t_pcompact.start();
#endif
    const size_t c0 = (size_t)intra * nslice / nintra, c1 = (size_t)(intra + 1) * nslice / nintra;
    for (size_t i = c0; i < c1; ++i)
      compact[slice_lo + i] = {live[i].mass, live[i].x, live[i].y, live[i].z};
#ifdef ITYR_PROF
    t_pcompact.stop();
#endif
  }
  // Order the intra peers' compact writes before the leader's Allgatherv in
  // build_step reads them (POSIX shm is coherent; this is the ordering barrier).
  collective_timed_cat(tc_bar, [&] { ityr::common::mpi_barrier(topo.intra_mpicomm()); });
#ifdef ITYR_PROF
  t_pdist.stop();
#endif
}

// ---------------- tree build ----------------

// Gather the live slices into the master's full sorted copy on demand (output
// and brute-force steps only; the build does not produce it). The master's
// own slice is copied into place and sent via MPI_IN_PLACE.
inline void gather_full_bodies(const SliceLayout& L, Body64* live, Body64* bodies) {
  auto& topo = ityr::common::topology::instance::get();
  const int inter = topo.inter_my_rank();
  const int intra = topo.intra_my_rank();
  if (intra != 0) return;
  int cnts[MAX_INTER_NODES] = {}, dspls[MAX_INTER_NODES] = {};
  for (int s = 0; s < (int)L.ninter; ++s) {
    cnts[s] = (int)((L.los[s + 1] - L.los[s]) * sizeof(Body64));
    dspls[s] = (int)(L.los[s] * sizeof(Body64));
  }
  if (inter == 0) memcpy(bodies + L.slice_lo, live, L.nslice * sizeof(Body64));
  collective_timed_cat(tc_dist, [&] {
    MPI_Gatherv(inter == 0 ? MPI_IN_PLACE : live,
                inter == 0 ? 0 : (int)(L.nslice * sizeof(Body64)), MPI_BYTE,
                bodies, cnts, dspls, MPI_BYTE, 0, topo.inter_mpicomm());
  });
}

// build_rec over the local compact replica: the fold reads mass/x/y/z at the
// sorted positions directly. All ranks read identical replica bytes, so the
// accumulated tree is the same everywhere.
inline void build_rec_direct(const uint64_t* keys, const CompactBody* compact, size_t lo,
                             size_t hi, int level, int leaf, vector<Node>& vec) {
  size_t own = vec.size();
  vec.push_back(Node{});
  Node& n = vec[own];
  n.first = (int)lo;
  n.count = (int)(hi - lo);
  n.nchild = 0;
  if (hi - lo <= (size_t)leaf || level >= NLEVELS) {
    n.leaf = true;
    for (size_t i = lo; i < hi; ++i) {
      const CompactBody& b = compact[i];
      add_to_com(n, b.mass, b.x, b.y, b.z);
    }
    finalize_com(n);
    return;
  }
  n.leaf = false;
  int shift = 3 * (NLEVELS - level);
  size_t i = lo;
  while (i < hi) {
    int c = octant(keys[i], shift);
    size_t j = octant_run_end(keys, i + 1, hi, shift, c);
    int slot = vec[own].nchild++;
    vec[own].child[slot] = (int)(vec.size() - own);
    build_rec_direct(keys, compact, i, j, level + 1, leaf, vec);
    i = j;
  }
  Node& m = vec[own];
  for (int k = 0; k < m.nchild; ++k) {
    const Node& ch = vec[own + m.child[k]];
    add_to_com(m, ch.mass, ch.cx, ch.cy, ch.cz);
  }
  finalize_com(m);
}

// ---- per-node replicas of the tree + compact ----
// The walk needs the full tree + compact on every rank. Both are derived
// structures, so replicating them is fine — only the particle set must be
// distributed. The tree is built on whatever rank the build landed on; each
// node's copy is delivered with a two-sided MPI_Bcast over inter_mpicomm (the
// color-0 group is the set of node leaders). get() on a same-node buffer is a
// zero-copy direct pointer, so every force leaf reads its own node's replica
// with plain loads (zero interconnect). The per-node pointers live in inline
// fixed arrays: they are captured into spawn closures, and task migration
// byte-copies the stack frame — a heap vector would dangle after a steal.
struct NodeRepPointers {
  Node* tree[MAX_INTER_NODES];
  CompactBody* compact[MAX_INTER_NODES];
  ForceResult* acc[MAX_INTER_NODES];   // force leaves write here in place
  uint64_t* cmask[MAX_INTER_NODES];    // per-node "chunk computed here" bitmask
};

struct NodeReplicas {
  NodeRepPointers rep;
  size_t tree_bytes = 0;
  bool owns_replica = false;              // this rank allocated the local tree copy
  void* my_tree = nullptr;                // this rank's tree replica buffer
};

// make_node_replicas: Bcast the tree (from its owner node) and the compact (from the master)
// to every node; publish the per-node pointers to every rank (leaves migrate).
NodeReplicas make_node_replicas(const BuildResult& br, CompactBody* compact,
                                const NodeRepPointers& accs) {
  auto& topo = ityr::common::topology::instance::get();
  const int inter = topo.inter_my_rank();
  const int intra = topo.intra_my_rank();
  const int ninter = topo.inter_n_ranks();
  NodeReplicas r{};   // value-init zeroes rep.tree[]/rep.compact[] + the scalars
  r.tree_bytes = br.tree_bytes;
  // Carry the persistent per-node accel buffers + chunk masks (allocated once
  // in main) into the pointer table the force leaves receive, so a leaf can
  // find its own node's buffer and mark its chunk.
  memcpy(r.rep.acc, accs.acc, sizeof(r.rep.acc));
  memcpy(r.rep.cmask, accs.cmask, sizeof(r.rep.cmask));

  // The tree build output (br.tree) lives on whatever rank the build task
  // landed on — not necessarily the master — so plain-loading it from another
  // node faults. Reuse it only when it is already on this node
  // (is_locally_accessible == same inter node); otherwise pull a replica.
  if (intra == 0) {
    auto& alloc = ityr::ito::worker::instance::get().sched().fpool().remote_bufs_allocator_;
    const bool tree_local = alloc.is_locally_accessible(br.tree);
    if (tree_local) {
      r.rep.tree[inter] = br.tree;
      r.my_tree = br.tree;
    } else {
      void* tr = ityr::ito::alloc_future_data(br.tree_bytes);
      r.rep.tree[inter] = (Node*)tr;
      r.my_tree = tr;
    }
    if (topo.inter_n_ranks() > 1) {
      // Two-sided Bcast orders the owner's plain stores before any reader (a
      // collective both sides complete), and it is a tree rather than a star.
      // The owner's node is derivable by pointer arithmetic
      // (allocator::get_owner -> inter_rank).
      int root = ityr::common::topology::inter_rank(alloc.get_owner(br.tree));
      collective_timed_cat(tc_ptr, [&] {
        MPI_Bcast(tree_local ? (void*)br.tree : r.my_tree, (int)br.tree_bytes, MPI_BYTE, root,
                  topo.inter_mpicomm());
      });
    }
    r.owns_replica = true;    // this node frees whatever it ended up holding
  }

  // The compact replicas were completed in build_step (in-place Allgatherv
  // before the tree build) — the walk just takes the local replica pointer.
  if (intra == 0) {
    r.rep.compact[inter] = compact;
  }

  // Every rank needs every node's replica pointer (leaves migrate).
  uintptr_t tbuf[MAX_INTER_NODES] = {}, cbuf[MAX_INTER_NODES] = {};
  if (intra == 0) {
    tbuf[inter] = (uintptr_t)r.rep.tree[inter];
    cbuf[inter] = (uintptr_t)r.rep.compact[inter];
  }
  // Fixed MAX_INTER_NODES count (nodes can be unequal).
  broadcast_uintptr_table(tbuf);
  broadcast_uintptr_table(cbuf);
  for (int i = 0; i < ninter; ++i) {
    r.rep.tree[i] = (Node*)tbuf[i];
    r.rep.compact[i] = (CompactBody*)cbuf[i];
  }
  return r;
}

void free_node_replicas(NodeReplicas& r) {
  if (r.owns_replica) {
    ityr::ito::dealloc_future_data(r.my_tree, r.tree_bytes);
  }
}

// ---------------- per-node build state ----------------

// Per-node build state. A node holds only its live slice (maintained in place
// by kick + scatter) and its merged-keys replica (derived metadata, allowed to
// be replicated).
struct BuildReplicas {
  SliceLayout L;
  Body64* live = nullptr;            // this node's slice of the sorted order (intra-0 buffer)
  size_t live_bytes = 0;
  uint64_t* keys = nullptr;          // this node's merged-keys replica (8 B * n)
  uint64_t* krun = nullptr;          // scratch: this node's sorted keys (8 B * nslice)
  size_t keys_bytes = 0;
  KeyPair* run = nullptr;            // scratch: this node's sorted key run (nslice * 16 B)
  GatherPair* pairs = nullptr;       // scratch: this node's gather run (nslice * 72 B)
  GatherPair* recv = nullptr;        // scratch: alltoallv receive (nslice * 72 B)
  uint64_t* gkeys = nullptr;         // scratch: allgathered key runs (8 B * n)
  uint32_t* mypos = nullptr;         // scratch: my run elements' global positions (4 B * nslice)
  // ping-pong scratch for the ninter>2 intra-parallel merge (16 B * n each):
  // (key, src run, src pos) triples carried through the pairwise rounds
  MergeTriple* mbufA = nullptr;
  MergeTriple* mbufB = nullptr;
  uint64_t* keys_table[MAX_INTER_NODES] = {};  // per-node merged-keys replica pointers
  bool owns = false;                 // this rank (intra-0) allocated the above
};

void init_build_replicas(BuildReplicas& r, size_t n, size_t gran) {
  auto& topo = ityr::common::topology::instance::get();
  const int inter = topo.inter_my_rank();
  const int intra = topo.intra_my_rank();
  const int ninter = topo.inter_n_ranks();
  compute_slice_layout(n, (size_t)ninter, gran, r.L);
  r.live = nullptr;
  r.live_bytes = 0;
  r.keys = nullptr;
  r.keys_bytes = 0;
  r.run = nullptr;
  r.pairs = nullptr;
  r.recv = nullptr;
  r.gkeys = nullptr;
  r.mbufA = nullptr;
  r.mbufB = nullptr;
  if (intra == 0) {
    if (r.L.nslice > 0) {
      r.live = (Body64*)ityr::ito::alloc_future_data(r.L.nslice * sizeof(Body64));
      r.live_bytes = r.L.nslice * sizeof(Body64);
      r.run = (KeyPair*)ityr::ito::alloc_future_data(r.L.nslice * sizeof(KeyPair));
      r.pairs = (GatherPair*)ityr::ito::alloc_future_data(r.L.nslice * sizeof(GatherPair));
      r.recv = (GatherPair*)ityr::ito::alloc_future_data(r.L.nslice * sizeof(GatherPair));
      r.krun = (uint64_t*)ityr::ito::alloc_future_data(r.L.nslice * sizeof(uint64_t));
      r.mypos = (uint32_t*)ityr::ito::alloc_future_data(r.L.nslice * sizeof(uint32_t));
    }
    r.gkeys = (uint64_t*)ityr::ito::alloc_future_data(n * sizeof(uint64_t));
    r.keys = (uint64_t*)ityr::ito::alloc_future_data(n * sizeof(uint64_t));
    r.keys_bytes = n * sizeof(uint64_t);
    if (ninter > 2) {
      r.mbufA = (MergeTriple*)ityr::ito::alloc_future_data(n * sizeof(MergeTriple));
      r.mbufB = (MergeTriple*)ityr::ito::alloc_future_data(n * sizeof(MergeTriple));
    }
    r.owns = true;
  }
  // Publish the per-node live/keys/run/pairs pointers to every rank (fixed
  // tables, broadcast once). run/pairs are written by all intra ranks in the
  // keys phase, so every rank needs them.
  uintptr_t lbuf[MAX_INTER_NODES] = {}, kbuf[MAX_INTER_NODES] = {};
  uintptr_t rbuf[MAX_INTER_NODES] = {}, pbuf[MAX_INTER_NODES] = {}, vbuf[MAX_INTER_NODES] = {};
  uintptr_t gbuf[MAX_INTER_NODES] = {}, mbuf[MAX_INTER_NODES] = {};
  uintptr_t mabuf[MAX_INTER_NODES] = {}, mbbuf[MAX_INTER_NODES] = {};
  if (intra == 0) {
    lbuf[inter] = (uintptr_t)r.live;
    kbuf[inter] = (uintptr_t)r.keys;
    rbuf[inter] = (uintptr_t)r.run;
    pbuf[inter] = (uintptr_t)r.pairs;
    vbuf[inter] = (uintptr_t)r.recv;
    gbuf[inter] = (uintptr_t)r.gkeys;
    mbuf[inter] = (uintptr_t)r.mypos;
    mabuf[inter] = (uintptr_t)r.mbufA;
    mbbuf[inter] = (uintptr_t)r.mbufB;
  }
  broadcast_uintptr_table(lbuf);
  broadcast_uintptr_table(kbuf);
  broadcast_uintptr_table(rbuf);
  broadcast_uintptr_table(pbuf);
  broadcast_uintptr_table(vbuf);
  broadcast_uintptr_table(gbuf);
  broadcast_uintptr_table(mbuf);
  broadcast_uintptr_table(mabuf);
  broadcast_uintptr_table(mbbuf);
  r.live = (Body64*)lbuf[inter];
  r.keys = (uint64_t*)kbuf[inter];
  r.run = (KeyPair*)rbuf[inter];
  r.pairs = (GatherPair*)pbuf[inter];
  r.recv = (GatherPair*)vbuf[inter];
  r.gkeys = (uint64_t*)gbuf[inter];
  r.mypos = (uint32_t*)mbuf[inter];
  r.mbufA = (MergeTriple*)mabuf[inter];
  r.mbufB = (MergeTriple*)mbbuf[inter];
  for (int i = 0; i < ninter; ++i) r.keys_table[i] = (uint64_t*)kbuf[i];
}

// SPMD, step 0 only: distribute the input (read on the master) across the live
// slices. Contiguous input-order chunks land at the slice offsets; the first
// build's sort re-orders them.
void scatter_input(const Body64* input, const SliceLayout& L, Body64* live) {
  auto& topo = ityr::common::topology::instance::get();
  const int intra = topo.intra_my_rank();
  const int ninter = topo.inter_n_ranks();
  int cnts[MAX_INTER_NODES] = {}, dspls[MAX_INTER_NODES] = {};
  for (int s = 0; s < ninter; ++s) {
    cnts[s] = (int)((L.los[s + 1] - L.los[s]) * sizeof(Body64));
    dspls[s] = (int)(L.los[s] * sizeof(Body64));
  }
  collective_timed([&] {
    if (intra == 0) {
      MPI_Scatterv(input, cnts, dspls, MPI_BYTE, live, (int)(L.nslice * sizeof(Body64)), MPI_BYTE,
                   0, topo.inter_mpicomm());
      if (L.nslice > 0)
        ityr::ito::flush_window_data(live, L.nslice * sizeof(Body64));
    }
    // The Scatterv is on inter_mpicomm (node leaders only); non-leader ranks must not race
    // ahead and read the live slice before the receive completes — this world barrier orders
    // the scatter before every rank's first bounds/keys scan.
    ityr::common::mpi_barrier(MPI_COMM_WORLD);
  });
}

// ---------------- tree-build plan (SPMD) ----------------

// One plan entry per planned subtree: its (lo, hi, level), parent/child
// indices, preorder offset in the shared array, and subtree size.
struct PlanNode {
  size_t lo, hi;
  int level;
  int parent;
  int child[8];
  int nchild;
  size_t off;      // the preorder offset (the Node units)
  size_t size;     // the subtree size (the Node units)
};

// The count walk and the fold walk are the same octant walk per leaf. The
// count pass records the subtree's preorder structure (the per-node lo/hi/
// level/nchild + the subtree node count), and the fold pass replays the
// record without re-probing the keys (only the body accumulation).
struct SubRec {
  uint32_t lo, hi, size;
  uint8_t level, nchild;
};
inline size_t rec_node_count_rec(const uint64_t* keys, size_t lo, size_t hi, int level, int leaf,
                                 vector<SubRec>& rec) {
  const size_t idx = rec.size();
  rec.push_back({(uint32_t)lo, (uint32_t)hi, 0, (uint8_t)level, 0});
  if (hi - lo <= (size_t)leaf || level >= NLEVELS) {
    rec[idx].size = 1;
    return 1;
  }
  int shift = 3 * (NLEVELS - level);
  size_t total = 1;
  size_t i = lo;
  while (i < hi) {
    int c = octant(keys[i], shift);
    size_t j = octant_run_end(keys, i + 1, hi, shift, c);
    ++rec[idx].nchild;
    total += rec_node_count_rec(keys, i, j, level + 1, leaf, rec);
    i = j;
  }
  rec[idx].size = (uint32_t)total;
  return total;
}
// The fold from the record: walk the preorder, derive each node's offset within the subtree
// (the sizes are recorded), fill the header, accumulate the leaves' bodies bottom-up. The
// children of node i sit at i+1, i+1+size(child1), ... (the preorder).
inline size_t fold_from_rec(const SubRec* rec, const CompactBody* compact, size_t idx,
                            int leaf, Node* base, size_t& off) {
  const SubRec& r = rec[idx];
  Node& nd = base[off];
  nd = Node{};
  nd.first = (int)r.lo;
  nd.count = (int)(r.hi - r.lo);
  nd.nchild = r.nchild;
  nd.leaf = (r.nchild == 0);
  const size_t myoff = off;
  ++off;
  if (r.nchild == 0) {
    for (size_t i = r.lo; i < r.hi; ++i) {
      const CompactBody& b = compact[i];
      add_to_com(nd, b.mass, b.x, b.y, b.z);
    }
    finalize_com(nd);
  } else {
    size_t child = idx + 1;
    for (int k = 0; k < r.nchild; ++k) {
      const size_t child_off = off;
      fold_from_rec(rec, compact, child, leaf, base, off);
      nd.child[k] = (int)(child_off - myoff);
      add_to_com(nd, base[child_off].mass, base[child_off].cx, base[child_off].cy, base[child_off].cz);
      child += rec[child].size;
    }
    finalize_com(nd);
  }
  return off;
}

// The plan's split structure only: the leaf subtree counts + the offsets are
// filled afterwards by the distributed count + prefix passes in
// build_tree_spmd.
inline void plan_node_struct(vector<PlanNode>& plan, const uint64_t* keys, size_t lo, size_t hi,
                             int level, size_t cutoff, int parent) {
  size_t idx = plan.size();
  plan.push_back({lo, hi, level, parent, {}, 0, 0, 0});
  if (hi - lo <= cutoff || level >= NLEVELS) return;
  int shift = 3 * (NLEVELS - level);
  size_t n = hi - lo;
  size_t i = 0;
  while (i < n) {
    int c = octant(keys[lo + i], shift);
    size_t j = octant_run_end(keys, lo + i + 1, lo + n, shift, c) - lo;
    plan[idx].child[plan[idx].nchild++] = (int)plan.size();
    plan_node_struct(plan, keys, lo + i, lo + j, level + 1, cutoff, (int)idx);
    i = j;
  }
}

// SPMD bulk-synchronous tree build. Every rank plans the tree identically
// (the keys replicas are identical on every rank), folds its static share of
// the leaves into local temp buffers, bulk-puts them to the master's single
// tree array at the precomputed offsets (one flush for all the puts), and the
// master assembles the internal nodes serially on top (reverse preorder — the
// children are always in place by construction). One barrier orders the puts
// before the assembly; make_node_replicas then delivers the per-node replicas.
BuildResult build_tree_spmd(const BuildCtx& ctx, size_t n, const Bounds& bd) {
  const size_t cutoff = cfg.build_cutoff, leaf = cfg.leaf;
  auto& topo = ityr::common::topology::instance::get();
  const int my = topo.my_rank();
  const int P = topo.n_ranks();
  const int inode = topo.inter_rank(my);
  const uint64_t* keys = ctx.keys[inode];
  // 1. The plan: the split structure on every rank (identical inputs =>
  //    identical plan), the leaf subtree counts distributed (leaf k -> rank
  //    (k % P)) + one Allgather, then the sizes/offsets derived identically
  //    everywhere.
#ifdef ITYR_PROF
  TimePoint t0pl = Clock::now();
#endif
  vector<PlanNode> plan;
  plan.reserve(2 * n / leaf + 16);
  plan.push_back({0, n, 0, -1, {}, 0, 0, 0});
  // The plan is distributed: every rank walks only the root's octant runs (one
  // O(n) pass), plans its share of the level-1 children (child c -> rank
  // c % P, round-robin) with local indices, fixes its sub-plan's parent/child
  // refs (+ its global base, the sub-root's parent sentinel -1 -> the global
  // root 0), and the sub-plans are exchanged via one MPI_Allgatherv. The
  // resulting plan order differs from a serial DFS but is a valid
  // parents-before-children order, so identical splits + folds give the
  // identical tree content (the array layout is internal — every rank derives
  // the same offsets).
  size_t rlo[8], rhi[8];
  int nroot = 0;
  if (n > cutoff) {
    const int shift = 3 * NLEVELS;
    size_t i = 0;
    while (i < n) {
      int c = octant(keys[i], shift);
      size_t j = octant_run_end(keys, i + 1, n, shift, c);
      rlo[nroot] = i;
      rhi[nroot] = j;
      ++nroot;
      i = j;
    }
  }
  vector<PlanNode> sub;
  sub.reserve(2 * n / leaf / (size_t)P + 16);
  vector<int> cpos((size_t)nroot, -1);   // the local sub-plan index of each level-1 child
  for (int c = my; c < nroot; c += P) {
    cpos[c] = (int)sub.size();
    plan_node_struct(sub, keys, rlo[c], rhi[c], 1, cutoff, -1);
  }
  vector<int> nplans((size_t)P);
  int my_n = (int)sub.size();
  collective_timed_cat(tc_ptr, [&] {
    MPI_Allgather(&my_n, 1, MPI_INT, nplans.data(), 1, MPI_INT, MPI_COMM_WORLD);
  });
  vector<size_t> rank_base((size_t)P + 1);
  rank_base[0] = 1;
  for (int q = 0; q < P; ++q) rank_base[q + 1] = rank_base[q] + (size_t)nplans[q];
  // publish each child's position within its owner's sub-plan (MPI_MAX over the -1 init).
  if (nroot > 0)
    collective_timed_cat(tc_ptr, [&] {
      MPI_Allreduce(MPI_IN_PLACE, cpos.data(), nroot, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    });
  plan.resize(rank_base[(size_t)P]);
  vector<int> rcnts((size_t)P), rdspls((size_t)P);
  int roff = 0;
  for (int q = 0; q < P; ++q) {
    rcnts[q] = nplans[q] * (int)sizeof(PlanNode);
    rdspls[q] = roff;
    roff += rcnts[q];
  }
  collective_timed_cat(tc_ptr, [&] {
    MPI_Allgatherv(sub.data(), (int)(sub.size() * sizeof(PlanNode)), MPI_BYTE,
                   plan.data() + 1, rcnts.data(), rdspls.data(), MPI_BYTE, MPI_COMM_WORLD);
  });
  // receiver-side fixups: every rank remaps all the blocks' parent/child refs (+ the block's
  // global base; the sub-root's parent sentinel -1 -> the global root 0), then the root's
  // child array from the published positions.
  for (int q = 0; q < P; ++q)
    for (size_t i = rank_base[q]; i < rank_base[q + 1]; ++i) {
      PlanNode& pn = plan[i];
      if (pn.parent != -1) pn.parent += (int)rank_base[q];
      else pn.parent = 0;
      for (int k = 0; k < pn.nchild; ++k) pn.child[k] += (int)rank_base[q];
    }
  for (int c = 0; c < nroot; ++c)
    plan[0].child[plan[0].nchild++] = (int)rank_base[c % P] + cpos[c];
  const size_t nplan = plan.size();
  vector<int> leaves;
  vector<int> ord(nplan, -1);
  leaves.reserve(nplan);
  for (size_t i = 0; i < nplan; ++i)
    if (plan[i].nchild == 0) {
      ord[i] = (int)leaves.size();
      leaves.push_back((int)i);
    }
  const size_t nleaves = leaves.size();
  // The count walk records each leaf's subtree preorder structure (SubRec);
  // the fold pass below replays it without re-probing the keys.
  vector<SubRec> subrec;
  subrec.reserve(2 * n / leaf / (size_t)P + 16);
  vector<size_t> lsize(nleaves, 0);
  for (size_t k = my; k < nleaves; k += P)
    lsize[k] = rec_node_count_rec(keys, plan[leaves[k]].lo, plan[leaves[k]].hi,
                                  plan[leaves[k]].level, (int)leaf, subrec);
  // sum the scattered per-rank contributions (each rank counted leaf k % P).
  collective_timed_cat(tc_ptr, [&] {
    MPI_Allreduce(MPI_IN_PLACE, lsize.data(), (int)nleaves, MPI_UINT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
  });
  for (size_t i = nplan; i-- > 0;) {
    if (plan[i].nchild == 0) {
      plan[i].size = lsize[ord[i]];
    } else {
      size_t s = 1;
      for (int k = 0; k < plan[i].nchild; ++k) s += plan[plan[i].child[k]].size;
      plan[i].size = s;
    }
  }
  size_t off = 0;
  for (size_t i = 0; i < nplan; ++i) {
    plan[i].off = off;
    off += plan[i].size;
  }
  const size_t total_nodes = off;
#ifdef ITYR_PROF
  t_splan.add_local(t0pl);
  TimePoint t0fl = Clock::now();
#endif
  // 2. the master's single tree array (the pointer is Bcast — the pool addresses are
  //    global, and the putting ranks need it to derive the target displacements).
  Node* array = nullptr;
  if (my == 0)
    array = (Node*)ityr::ito::alloc_future_data(total_nodes * sizeof(Node));
  uint64_t ap = (uint64_t)array;
  collective_timed_cat(tc_ptr, [&] { MPI_Bcast(&ap, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD); });
  array = (Node*)ap;
  // 3. the per-rank fold + bulk put: leaf k -> rank (k % P) (interleaved for balance).
  //    Node-0 ranks fold straight into the master's array (the pool is a
  //    node-wide shm); ranks on other nodes fold into local temps and put
  //    them, all under one flush.
  auto& rmr = ityr::ito::worker::instance::get().sched().fpool().remote_bufs_allocator_;
  vector<Node*> tmps;
  vector<size_t> tsz;
  tmps.reserve(nleaves / P + 2);
  tsz.reserve(nleaves / P + 2);
  bool any_remote = false;
  size_t rec_i = 0;
  for (size_t k = my; k < nleaves; k += P) {
    const PlanNode& pn = plan[leaves[k]];
    const SubRec* rec = subrec.data() + rec_i;
    const size_t nrec = lsize[k];   // the leaf's recorded subtree node count
    if (rmr.is_locally_accessible(array + pn.off)) {
      size_t o = 0;
      fold_from_rec(rec, ctx.compact[inode], 0, (int)leaf, array + pn.off, o);
    } else {
      Node* tmp = (Node*)ityr::ito::alloc_future_data(pn.size * sizeof(Node));
      size_t o = 0;
      fold_from_rec(rec, ctx.compact[inode], 0, (int)leaf, tmp, o);
      ityr::common::mpi_put_nb(tmp, pn.size, 0, rmr.get_disp(array + pn.off), rmr.win());
      tmps.push_back(tmp);
      tsz.push_back(pn.size);
      any_remote = true;
    }
    rec_i += nrec;
  }
  if (any_remote) ityr::common::mpi_win_flush(0, rmr.win());
  for (size_t t = 0; t < tmps.size(); ++t)
    ityr::ito::dealloc_future_data(tmps[t], tsz[t] * sizeof(Node));
  // 4. every fold+put is flushed before the assembly starts.
  collective_timed_cat(tc_bar, [&] { ityr::common::mpi_barrier(MPI_COMM_WORLD); });
#ifdef ITYR_PROF
  t_sfold.add_local(t0fl);
  TimePoint t0as = Clock::now();
#endif
  // 5. the master assembles the internal nodes serially (reverse preorder).
  if (my == 0) {
    for (size_t i = nplan; i-- > 0;) {
      if (plan[i].nchild == 0) continue;
      Node& nd = array[plan[i].off];
      nd.first = (int)plan[i].lo;
      nd.count = (int)(plan[i].hi - plan[i].lo);
      nd.nchild = plan[i].nchild;
      nd.leaf = false;
      nd.mass = 0;
      nd.cx = nd.cy = nd.cz = 0;
      for (int k = 0; k < plan[i].nchild; ++k) {
        const Node& ch = array[plan[plan[i].child[k]].off];
        nd.child[k] = (int)(plan[plan[i].child[k]].off - plan[i].off);
        add_to_com(nd, ch.mass, ch.cx, ch.cy, ch.cz);
      }
      finalize_com(nd);
    }
#ifdef ITYR_PROF
    t_sasm.add_local(t0as);
#endif
  }
  return {array, total_nodes * sizeof(Node), bd.box_side, bd.xmin, bd.ymin, bd.zmin};
}

// SPMD, every step: slice keys -> redistribute -> tree build. All parts sit
// inside the build timer.
BuildResult build_step(BuildReplicas& r, size_t n, bool first_step,
                       Body64* input_buf, const Bounds& bd,
                       CompactBody* const compact_table[MAX_INTER_NODES]) {
  const int intra = ityr::common::topology::intra_my_rank();
  const int inter = ityr::common::topology::inter_my_rank();
  CompactBody* compact = compact_table[inter];   // this node's replica
  w_build.start();
  slice_keys_phase(r.L, r.live, bd, r.run, r.pairs);
  redistribute_build(r.L, r.pairs, r.run, r.krun, r.live, r.keys, r.gkeys, r.recv, r.mypos, compact,
                     r.mbufA, r.mbufB);
  // Complete the compact replicas before the tree build (in-place Allgatherv;
  // every node's slice is already folded), so the leaf folds can read the
  // local replica instead of pulling remote body ranges.
  if (intra == 0) {
    int ccnts[MAX_INTER_NODES] = {}, cdspls[MAX_INTER_NODES] = {};
    for (int s = 0; s < (int)r.L.ninter; ++s) {
      ccnts[s] = (int)((r.L.los[s + 1] - r.L.los[s]) * sizeof(CompactBody));
      cdspls[s] = (int)(r.L.los[s] * sizeof(CompactBody));
    }
    collective_timed_cat(tc_dist, [&] {
      MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_BYTE, compact, ccnts, cdspls, MPI_BYTE,
                     ityr::common::topology::instance::get().inter_mpicomm());
    });
  }
  // Order the leader's in-place Allgatherv before the intra peers read the
  // compact in the tree fold (the replica is the node-wide shm buffer).
  collective_timed_cat(tc_bar, [&] { ityr::common::mpi_barrier(ityr::common::topology::instance::get().intra_mpicomm()); });
  BuildCtx ctx;
  // Per-node keys/compact replica pointers, indexed by executing node.
  for (int i = 0; i < MAX_INTER_NODES; ++i) {
    ctx.keys[i] = r.keys_table[i];
    ctx.compact[i] = compact_table[i];
  }
#ifdef ITYR_PROF
  t_ptree.start();
#endif
  BuildResult br = build_tree_spmd(ctx, n, bd);
  ityr::ito::flush_copy_cache();
#ifdef ITYR_PROF
  t_ptree.stop();
#endif
  // The step-0 input buffer was consumed by the scatter and is not needed after this.
  if (first_step && ityr::is_master() && input_buf)
    ityr::ito::dealloc_future_data(input_buf, n * sizeof(Body64));
  w_build.stop();
  return br;
}

// Per-body forces for [lo,hi): writes in place into the executing node's accel
// buffer. Runs entirely on the node-local tree + compact replica (zero-copy
// direct pointer — no get(), no per-rank interconnect pull). Migration-safe:
// it reads the replica of the node that executes it, which is identical on
// every node. Batches K = forceN().lanes() consecutive compact bodies through
// the shared depth-first SIMD traversal (dispatched at runtime to the best ISA
// the hardware supports); per-body node-visit order is unchanged vs the scalar
// walk, so each body's (ax,ay,az,pot) matches the scalar result. Writing
// another rank's buffer with plain stores is legal within a node: the future
// pool is a node-wide POSIX shm segment mapped read-write at identical
// addresses in every intra-node peer. Leaves cannot be pinned to a node, so
// each node ends up with only the slices its own ranks happened to execute;
// combine_accel() completes them afterwards.
int force_range(NodeRepPointers rep, size_t lo, size_t hi, size_t chunk, WalkParams wp) {
  return par_for(lo, hi, chunk, [=](size_t l, size_t r) mutable {
    int inode = ityr::common::topology::inter_rank(ityr::my_rank());
    Node* tree = rep.tree[inode];
    CompactBody* bodies = rep.compact[inode];
    ForceResult* out = rep.acc[inode];
    size_t n = r - l;
    forceN_desc f = forceN();   // runtime-dispatched SIMD batch kernel (once)
    // Mark every chunk this leaf covers as computed by this node. Leaves are
    // chunk-aligned, so this is normally a single bit. Relaxed atomic OR is
    // enough: the bit is only observed after the walk join, and the intra-node
    // barrier in combine orders it. The buffer is this node's own shm mapping,
    // so the atomic is a same-node RMW.
    for (size_t c = l / chunk; c <= (r - 1) / chunk; ++c)
      __atomic_fetch_or(&rep.cmask[inode][c >> 6], 1ull << (c & 63), __ATOMIC_RELAXED);
    size_t i = 0;
    // SIMD batches of K (lanes are the bodies at compact indices l+i..l+i+K-1).
    for (; i + f.lanes <= n; i += f.lanes) {
      double ax[64], ay[64], az[64], pot[64];
      f.fn(l + i, bodies, tree, wp, ax, ay, az, pot);
      for (size_t k = 0; k < f.lanes; ++k) {
        ForceResult& fr = out[l + i + k];
        fr.ax = ax[k];
        fr.ay = ay[k];
        fr.az = az[k];
        fr.pot = pot[k];
      }
    }
    // scalar tail (n not a multiple of f.lanes)
    for (; i < n; ++i) {
      CompactBody& b = bodies[l + i];
      ForceResult& fr = out[l + i];
      fr.ax = fr.ay = fr.az = fr.pot = 0;
      walk_compact(0, 0, b, tree, bodies, wp, fr.ax, fr.ay, fr.az, fr.pot);
    }
  });
}

void forces_phase(const NodeReplicas& reps, size_t n,
                  double box_side, double xmin, double ymin, double zmin) {
  WalkParams wp = make_walk_params(box_side, xmin, ymin, zmin, cfg.theta * cfg.theta,
                                   cfg.G, cfg.eps2);
  NodeRepPointers rep = reps.rep;   // copy into the closure (migration-safe)
  auto ff = ityr::spawn<int>(
      [=] { return force_range(rep, 0, n, cfg.chunk, wp); });
  ff.wait();
  // No gather: every leaf has already written its slice into its own node's
  // accel buffer; combine_accel() (SPMD) completes the per-node copies after.
}

// One persistent accel buffer per node, allocated once and reused every step.
// Returns this node's buffer and publishes every node's pointer, so a force
// leaf that runs here can find its own node's slot. Also allocates the
// per-node chunk mask (1 bit per `chunk` bodies), which the combine's
// chunk-bitmask exchange reads to know which chunks this node computed.
void alloc_node_accel(size_t n, NodeRepPointers& rep, void*& my_acc, bool& owns,
                      size_t& acc_bytes, void*& my_cmask, bool& owns_cmask) {
  auto& topo = ityr::common::topology::instance::get();
  const int inter = topo.inter_my_rank();
  const int intra = topo.intra_my_rank();
  const int ninter = topo.inter_n_ranks();
  acc_bytes = n * sizeof(ForceResult);
  size_t mask_words = ((n + cfg.chunk - 1) / cfg.chunk + 63) / 64;
  if (intra == 0) {
    void* p = ityr::ito::alloc_future_data(acc_bytes);
    rep.acc[inter] = (ForceResult*)p;
    my_acc = p;
    owns = true;
    void* mp = ityr::ito::alloc_future_data(mask_words * sizeof(uint64_t));
    memset(mp, 0, mask_words * sizeof(uint64_t));
    rep.cmask[inter] = (uint64_t*)mp;
    my_cmask = mp;
    owns_cmask = true;
  }
  uintptr_t abuf[MAX_INTER_NODES] = {}, mbuf[MAX_INTER_NODES] = {};
  if (intra == 0) {
    abuf[inter] = (uintptr_t)rep.acc[inter];
    mbuf[inter] = (uintptr_t)rep.cmask[inter];
  }
  // Fixed MAX_INTER_NODES count (nodes can be unequal).
  broadcast_uintptr_table(abuf);
  broadcast_uintptr_table(mbuf);
  for (int i = 0; i < ninter; ++i) {
    rep.acc[i] = (ForceResult*)abuf[i];
    rep.cmask[i] = (uint64_t*)mbuf[i];
  }
}

// SPMD. Zero this node's accel buffer before the walk, split across the node's
// intra ranks. Required for correctness of the combine below (a slot this node
// did not compute must be bit-zero) and it must run after the kick has consumed
// the previous step's values.
void zero_node_accel(NodeRepPointers& rep, size_t n) {
  const int inter = ityr::common::topology::inter_rank(ityr::my_rank());
  const int intra = ityr::common::topology::intra_rank(ityr::my_rank());
  const int nintra = ityr::common::topology::intra_n_ranks();
  size_t lo = (size_t)intra * n / nintra, hi = (size_t)(intra + 1) * n / nintra;
  memset(rep.acc[inter] + lo, 0, (hi - lo) * sizeof(ForceResult));
  // The chunk mask is tiny; one rank zeroes it, ordered before the walk by the barrier below.
  if (intra == 0) {
    size_t mask_words = ((n + cfg.chunk - 1) / cfg.chunk + 63) / 64;
    memset(rep.cmask[inter], 0, mask_words * sizeof(uint64_t));
  }
  collective_timed_cat(tc_bar, [&] { ityr::common::mpi_barrier(ityr::common::topology::intra_mpicomm()); });
}

// SPMD. Complete every node's accel buffer after the walk. Instead of
// Allreduce-ing the whole array, each node tracks which `chunk`-aligned ranges
// it computed in a per-node bitmask (rep.cmask, set by the force leaves). Every
// chunk is computed by exactly one node, so the leaders Allgather the masks and
// each node pulls exactly the chunks it lacks from their owner as contiguous
// runs. Each pulled slot carries the owner's complete value (the owner computed
// the whole chunk) and every other slot was already written locally, so the
// result is the same as a full-array exchange and no FP arithmetic is involved.
// Ordering: the writer's plain stores must be visible to the reader's MPI_Get.
// Each leader flushes its acc + mask (cache flush + MPI_Win_sync), the Allgather
// then orders everything (a collective both sides complete), and only then do
// the RMA pulls run. The missing chunks are gathered per remote owner and
// pulled with one MPI_Get over an hindexed MPI_BYTE datatype (blocklengths and
// displacements in bytes, relative to the owner's acc base): with flat work
// stealing the missing set fragments into many short runs, and per-op flush
// latency would dominate a one-get-per-run loop. The acc buffers are indexed by
// global body index on every node, so the origin and target displacement lists
// are identical and one datatype serves both sides.
void combine_accel(NodeRepPointers& rep, size_t n, size_t chunk) {
  auto& topo = ityr::common::topology::instance::get();
  if (topo.inter_n_ranks() <= 1) return;   // single node: nothing to combine
#ifdef ITYR_PROF
  t_pcombine.start();
#endif
  const int inter = topo.inter_my_rank();
  const int intra = topo.intra_my_rank();
  const int ninter = topo.inter_n_ranks();
  const size_t nchunks = (n + chunk - 1) / chunk;
  const size_t mask_words = (nchunks + 63) / 64;
  collective_timed([&] {
#ifdef ITYR_PROF
    // leader-side sub-step breakdown (barrier waits = rank-arrival skew)
    static long cb_step = 0;
    bool cb_leader = (intra == 0);
    double cb_t0 = cb_leader ? std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now().time_since_epoch()).count() : 0;
#endif
    ityr::common::mpi_barrier(topo.intra_mpicomm());   // all intra ranks' mask bits + acc stores done
#ifdef ITYR_PROF
    double cb_bar1 = 0, cb_fl = 0, cb_ag = 0, cb_pl = 0, cb_fl2 = 0, cb_bar2 = 0;
    if (cb_leader) cb_bar1 = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now().time_since_epoch()).count() - cb_t0;
#endif
    if (intra == 0) {
      auto& alloc = ityr::ito::worker::instance::get().sched().fpool().remote_bufs_allocator_;
#ifdef ITYR_PROF
      double cb_p = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
      ityr::ito::flush_window_data(rep.acc[inter], n * sizeof(ForceResult));
      ityr::ito::flush_window_data(rep.cmask[inter], mask_words * sizeof(uint64_t));
#ifdef ITYR_PROF
      if (cb_leader) cb_fl = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now().time_since_epoch()).count() - cb_p;
      cb_p = std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
      auto* all = (uint64_t*)ityr::ito::alloc_future_data(mask_words * ninter * sizeof(uint64_t));
      memset(all, 0, mask_words * ninter * sizeof(uint64_t));
      ityr::common::mpi_allgather(rep.cmask[inter], mask_words, all, mask_words,
                                  topo.inter_mpicomm());
#ifdef ITYR_PROF
      if (cb_leader) cb_ag = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now().time_since_epoch()).count() - cb_p;
      cb_p = std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
      // collect the contiguous missing-chunk runs, grouped by their owner node
      // (SPMD code, not a migrated task -> heap containers are safe here).
      struct Run { size_t off, len; };   // in ForceResult elements
      std::vector<Run> runs_by_owner[MAX_INTER_NODES];
      size_t c = 0;
      while (c < nchunks) {
        // The owner of chunk c: exactly one node has the bit set. Self-owned chunks need nothing.
        int owner = inter;
        for (int j = 0; j < ninter; ++j)
          if (all[j * mask_words + (c >> 6)] & (1ull << (c & 63))) { owner = j; break; }
        size_t run_end = c + 1;
        while (run_end < nchunks &&
               all[owner * mask_words + (run_end >> 6)] & (1ull << (run_end & 63)))
          ++run_end;
        if (owner != inter) {
          size_t first = c * chunk;
          size_t last = std::min(run_end * chunk, n);
          runs_by_owner[owner].push_back({first, last - first});
        }
        c = run_end;
      }
      for (int j = 0; j < ninter; ++j) {
        const auto& runs = runs_by_owner[j];
        if (runs.empty()) continue;
        if (alloc.is_locally_accessible(rep.acc[j])) {
          // same-node peer cannot occur at >1 node; defensive fallback only
          for (const auto& r : runs)
            std::memcpy(rep.acc[inter] + r.off, rep.acc[j] + r.off, r.len * sizeof(ForceResult));
          continue;
        }
        std::vector<int> lens(runs.size());
        std::vector<MPI_Aint> disps(runs.size());
        for (size_t k = 0; k < runs.size(); ++k) {
          lens[k] = (int)(runs[k].len * sizeof(ForceResult));
          disps[k] = (MPI_Aint)(runs[k].off * sizeof(ForceResult));
        }
        MPI_Datatype t;
        MPI_Type_create_hindexed((int)runs.size(), lens.data(), disps.data(), MPI_BYTE, &t);
        MPI_Type_commit(&t);
        int peer = alloc.get_owner(rep.acc[j]);
        MPI_Get(rep.acc[inter], 1, t, peer, alloc.get_disp(rep.acc[j]), 1, t, alloc.win());
        MPI_Win_flush(peer, alloc.win());
        MPI_Type_free(&t);
      }
      ityr::ito::dealloc_future_data(all, mask_words * ninter * sizeof(uint64_t));
#ifdef ITYR_PROF
      cb_p = std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
      if (cb_leader) cb_pl = cb_p - (cb_t0 + cb_bar1 + cb_fl + cb_ag);
      cb_p = std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
      ityr::ito::flush_window_data(rep.acc[inter], n * sizeof(ForceResult));   // order for intra peers
#ifdef ITYR_PROF
      if (cb_leader) cb_fl2 = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now().time_since_epoch()).count() - cb_p;
#endif
    }
#ifdef ITYR_PROF
    double cb_q = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
    ityr::common::mpi_barrier(topo.intra_mpicomm());
#ifdef ITYR_PROF
    if (cb_leader) {
      cb_bar2 = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now().time_since_epoch()).count() - cb_q;
      printf("PROF combine step %ld node %d: bar1 %.1f  flush1 %.1f  allg %.1f  pulls %.1f  flush2 %.1f  bar2 %.1f (ms)\n",
             cb_step++, inter, cb_bar1, cb_fl, cb_ag, cb_pl, cb_fl2, cb_bar2);
      fflush(stdout);
    }
#endif
  });
#ifdef ITYR_PROF
  t_pcombine.stop();
#endif
}

// ---------------- integrate / bounds (slice-local) ----------------

// SPMD. Kick (and optionally drift) this node's live slice in place, reading
// acc for the slice's bodies from the node-local accel buffer (complete after
// combine_accel). The leapfrog applies each acceleration as two half-kicks —
// one at the end of the step that produced it, one at the start of the next
// step — and both read the same acc array; fusing them into one pass performs
// the same per-body operations in the same order while saving a pass over the
// bodies + acc per step. `half_kicks` says how many of them this call performs:
//   step 1         -> 1 (a_0 has no trailing kick: step 0 is outside the schedule)
//   step k >= 2    -> 2 (a_{k-1}'s trailing kick fused with its leading kick)
//   after the loop -> 1 (a_S's trailing kick)
// The drift must use the velocity after all the half-kicks of this call — that
// is what the sequential kick(); drift() sequence does; passing 2 at step 1
// would double-apply a_0. Also computes min/max over this node's slice of the
// (drifted) positions, so the caller can Allreduce them into the next build's
// bounds (min/max are order-independent, so any split is exact).
void kick_live(BuildReplicas& r, ForceResult* acc_rep[], int half_kicks, bool drift,
               Extremes* ex = nullptr) {
  const int inter = ityr::common::topology::inter_rank(ityr::my_rank());
  Body64* b = r.live;
  const ForceResult* acc = acc_rep[inter];
  const int intra = ityr::common::topology::intra_rank(ityr::my_rank());
  const int nintra = ityr::common::topology::intra_n_ranks();
  size_t lo = (size_t)intra * r.L.nslice / nintra, hi = (size_t)(intra + 1) * r.L.nslice / nintra;
  const size_t base = r.L.slice_lo;
#ifdef ITYR_PROF
  t_pkloop.start();
#endif
  // The extremes must start from a post-drift value: initializing from b[lo]
  // before the loop would capture the pre-drift position and keep the stale
  // value for the rest of the step. The first iteration therefore initializes
  // the extremes instead of updating them.
  double xmin = 0, xmax = 0, ymin = 0, ymax = 0, zmin = 0, zmax = 0;
  bool ex_first = true;
  for (size_t i = lo; i < hi; ++i) {
    for (int k = 0; k < half_kicks; ++k) {
      b[i].vx += acc[base + i].ax * cfg.dt / 2;
      b[i].vy += acc[base + i].ay * cfg.dt / 2;
      b[i].vz += acc[base + i].az * cfg.dt / 2;
    }
    if (drift) {
      b[i].x += b[i].vx * cfg.dt;
      b[i].y += b[i].vy * cfg.dt;
      b[i].z += b[i].vz * cfg.dt;
    }
    if (ex_first) {
      xmin = xmax = b[i].x; ymin = ymax = b[i].y; zmin = zmax = b[i].z;
      ex_first = false;
    } else {
      xmin = std::min(xmin, b[i].x); xmax = std::max(xmax, b[i].x);
      ymin = std::min(ymin, b[i].y); ymax = std::max(ymax, b[i].y);
      zmin = std::min(zmin, b[i].z); zmax = std::max(zmax, b[i].z);
    }
  }
  if (ex) *ex = {xmin, xmax, ymin, ymax, zmin, zmax};
#ifdef ITYR_PROF
  t_pkloop.stop();
  t_pkflush.start();
#endif
  ityr::ito::flush_window_data(b + lo, (hi - lo) * sizeof(Body64));
#ifdef ITYR_PROF
  t_pkflush.stop();
#endif
}

// SPMD. Global bounds over this node's slice scan (used at step 0, where no
// kick has run).
Bounds bounds_from_replica(BuildReplicas& rep) {
  const Body64* b = rep.live;
  const int intra = ityr::common::topology::intra_rank(ityr::my_rank());
  const int nintra = ityr::common::topology::intra_n_ranks();
  size_t lo = (size_t)intra * rep.L.nslice / nintra, hi = (size_t)(intra + 1) * rep.L.nslice / nintra;
  double xmin = b[lo].x, xmax = b[lo].x, ymin = b[lo].y, ymax = b[lo].y;
  double zmin = b[lo].z, zmax = b[lo].z;
  for (size_t i = lo + 1; i < hi; ++i) {
    xmin = std::min(xmin, b[i].x); xmax = std::max(xmax, b[i].x);
    ymin = std::min(ymin, b[i].y); ymax = std::max(ymax, b[i].y);
    zmin = std::min(zmin, b[i].z); zmax = std::max(zmax, b[i].z);
  }
  // min(x) = -max(-x): negate the minima before the MPI_MAX reduce.
  double loc[6] = {-xmin, xmax, -ymin, ymax, -zmin, zmax};
  double glb[6];
  collective_timed_cat(tc_misc, [&] { MPI_Allreduce(loc, glb, 6, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); });
  xmin = -glb[0]; xmax = glb[1]; ymin = -glb[2]; ymax = glb[3]; zmin = -glb[4]; zmax = glb[5];
  Bounds bd2 = bounds_from_extremes({xmin, xmax, ymin, ymax, zmin, zmax});
  return bd2;
}

// ---------------- energy (canonical blocked sums) ----------------

// SPMD. Every ENERGY_BLOCK block sum is computed serially, in index order, by
// exactly one rank reading local memory; the per-node partials travel in a
// tiny Allgather (2*nblocks doubles per node), and the master tree_combines
// the block sums in the canonical order (see blocked_sum in
// barneshut_common.hpp). Each block sum is the same serial sum over identical
// bytes: EK from the slice-owning node's live slice, EP from any node's
// complete acc replica. So the result matches the sequential program.
// Requires slice boundaries (build_cutoff) to be multiples of ENERGY_BLOCK so
// an EK block never spans two slices; otherwise the master-gather fallback
// runs (correct for any build_cutoff).
std::pair<double, double> energy_phase(BuildReplicas& r, const NodeRepPointers& accs,
                                       MasterBuild& m, size_t n) {
  auto& topo = ityr::common::topology::instance::get();
  const int inter = topo.inter_my_rank();
  const int intra = topo.intra_my_rank();
  const int ninter = topo.inter_n_ranks();
  const size_t nblocks = (n + ENERGY_BLOCK - 1) / ENERGY_BLOCK;
  double ek = 0, ep = 0;

  if (cfg.build_cutoff % ENERGY_BLOCK != 0) {
    // ---- fallback (any build_cutoff): master-gather path ----
    struct Mv { double mass, vx, vy, vz; };
    const size_t nslice = r.L.nslice;
    if (intra == 0) {
      for (size_t j = 0; j < nslice; ++j) {
        const Body64& b = r.live[j];
        ((Mv*)r.pairs)[j] = {b.mass, b.vx, b.vy, b.vz};
      }
    }
    int cnts[MAX_INTER_NODES] = {}, dspls[MAX_INTER_NODES] = {};
    for (int s = 0; s < ninter; ++s) {
      cnts[s] = (int)((r.L.los[s + 1] - r.L.los[s]) * sizeof(Mv));
      dspls[s] = (int)(r.L.los[s] * sizeof(Mv));
    }
    if (intra == 0)
      collective_timed_cat(tc_energy, [&] {
        MPI_Gatherv(r.pairs, (int)(nslice * sizeof(Mv)), MPI_BYTE,
                    m.vel_scratch, cnts, dspls, MPI_BYTE, 0, topo.inter_mpicomm());
      });
    if (inter == 0 && intra == 0) {
      vector<double> v(n);
      const Mv* mv = (const Mv*)m.vel_scratch;
      for (size_t i = 0; i < n; ++i)
        v[i] = 0.5 * mv[i].mass * (mv[i].vx * mv[i].vx + mv[i].vy * mv[i].vy + mv[i].vz * mv[i].vz);
      ek = blocked_sum(v.data(), n);
      vector<double> pot(n);
      const ForceResult* acc = accs.acc[0];
      for (size_t i = 0; i < n; ++i) pot[i] = acc[i].pot;
      ep = 0.5 * blocked_sum(pot.data(), n);
    }
  } else {
    // ---- distributed path (build_cutoff is a multiple of ENERGY_BLOCK) ----
    // EK block k is owned by the node whose slice contains body index
    // k*ENERGY_BLOCK (slices are ENERGY_BLOCK-aligned, so a block never spans
    // two slices); the owner sums it serially from its local live slice.
    // EP block k is owned by round-robin inter rank; every node's acc replica
    // is complete and identical, so the owner reads its own local copy.
    auto owner_ek = [&](size_t k) {
      size_t a = k * ENERGY_BLOCK;
      for (int s = 0; s < ninter; ++s)
        if (a < r.L.los[s + 1]) return s;   // los[0] = 0 <= a always; empty slices skipped
      return ninter - 1;
    };
    auto owner_ep = [&](size_t k) {
      size_t s = (k * (size_t)ninter) / nblocks;
      return s < (size_t)ninter - 1 ? (int)s : ninter - 1;
    };
    size_t my_ek_lo = nblocks, my_ek_hi = 0, my_ep_lo = nblocks, my_ep_hi = 0;
    if (intra == 0) {
      for (size_t k = 0; k < nblocks; ++k)
        if (owner_ek(k) == inter) { my_ek_lo = k; break; }
      for (size_t k = nblocks; k-- > 0;)
        if (owner_ek(k) == inter) { my_ek_hi = k + 1; break; }
      for (size_t k = 0; k < nblocks; ++k)
        if (owner_ep(k) == inter) { my_ep_lo = k; break; }
      for (size_t k = nblocks; k-- > 0;)
        if (owner_ep(k) == inter) { my_ep_hi = k + 1; break; }
    }
    vector<double> parts(2 * (size_t)ninter * nblocks, 0.0);
    double* ek_parts = parts.data();
    double* ep_parts = parts.data() + (size_t)ninter * nblocks;
    const size_t glo = r.L.slice_lo;   // r.live is slice-local: global index i = glo + local i
    if (intra == 0) {
      const Body64* b = r.live;
      for (size_t k = my_ek_lo; k < my_ek_hi; ++k) {   // serial within each block, index order
        size_t a = k * ENERGY_BLOCK, e = std::min(a + ENERGY_BLOCK, n);
        double acc = 0;
        for (size_t i = a; i < e; ++i) {
          const Body64& bb = b[i - glo];               // blocks never span slices -> local index valid
          const double vx = bb.vx, vy = bb.vy, vz = bb.vz;
          acc += 0.5 * bb.mass * (vx * vx + vy * vy + vz * vz);
        }
        ek_parts[(size_t)inter * nblocks + k] = acc;
      }
      const ForceResult* acc = accs.acc[inter];        // acc buffers are global-indexed [0,n)
      for (size_t k = my_ep_lo; k < my_ep_hi; ++k) {
        size_t a = k * ENERGY_BLOCK, e = std::min(a + ENERGY_BLOCK, n);
        double s = 0;
        for (size_t i = a; i < e; ++i) s += acc[i].pot;
        ep_parts[(size_t)inter * nblocks + k] = s;
      }
    }
    // Only the node leaders are members of inter_mpicomm — running this
    // collective on intra != 0 ranks is undefined behavior.
    if (intra == 0)
      collective_timed_cat(tc_energy, [&] {
        // Two Allgathers, one per half of `parts`: a single Allgather over
        // parts.data() would only exchange the EK half. Each buffer's
        // rank-local segment is [inter*nblocks, ...) == the MPI_IN_PLACE
        // contribution.
        MPI_Allgather(MPI_IN_PLACE, (int)nblocks, MPI_DOUBLE,
                      ek_parts, (int)nblocks, MPI_DOUBLE, topo.inter_mpicomm());
        MPI_Allgather(MPI_IN_PLACE, (int)nblocks, MPI_DOUBLE,
                      ep_parts, (int)nblocks, MPI_DOUBLE, topo.inter_mpicomm());
      });
    if (inter == 0 && intra == 0) {
      vector<double> eb(nblocks), pb(nblocks);
      for (size_t k = 0; k < nblocks; ++k) {
        eb[k] = ek_parts[(size_t)owner_ek(k) * nblocks + k];
        pb[k] = ep_parts[(size_t)owner_ep(k) * nblocks + k];
      }
      ek = (nblocks == 1) ? eb[0] : tree_combine(eb.data(), 0, nblocks);
      ep = 0.5 * ((nblocks == 1) ? pb[0] : tree_combine(pb.data(), 0, nblocks));
    }
  }
  double both[2] = {ek, ep};
  collective_timed_cat(tc_energy, [&] { MPI_Bcast(both, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD); });
  return {both[0], both[1]};
}

// ---------------- output ----------------

// VTK snapshot with pot/acc taken from the node-local accel buffer.
// Positions/velocities/mass come from the (sorted) live bodies — on the master
// the gathered full copy, in the same index order as acc.
void write_vtk_acc(const Body64* b, const ForceResult* fr, size_t n, int step) {
  char path[512];
  snprintf(path, sizeof(path), "%s_%05d.vtk", cfg.prefix.c_str(), step);
  ofstream f(path);
  f.precision(17);
  f << "# vtk DataFile Version 3.0\nbarneshut snapshot " << step
    << "\nASCII\nDATASET POLYDATA\nPOINTS " << n << " double\n";
  for (size_t i = 0; i < n; ++i) f << b[i].x << " " << b[i].y << " " << b[i].z << "\n";
  f << "POINT_DATA " << n << "\nSCALARS id long\nLOOKUP_TABLE default\n";
  for (size_t i = 0; i < n; ++i) f << b[i].id << "\n";
  f << "SCALARS mass double 1\nLOOKUP_TABLE default\n";
  for (size_t i = 0; i < n; ++i) f << b[i].mass << "\n";
  f << "SCALARS potential double 1\nLOOKUP_TABLE default\n";
  for (size_t i = 0; i < n; ++i) f << fr[i].pot << "\n";
  f << "VECTORS velocity double\n";
  for (size_t i = 0; i < n; ++i) f << b[i].vx << " " << b[i].vy << " " << b[i].vz << "\n";
  f << "VECTORS acceleration double\n";
  for (size_t i = 0; i < n; ++i) f << fr[i].ax << " " << fr[i].ay << " " << fr[i].az << "\n";
}

void output_phase(const Body64* b, const ForceResult* fr, size_t n, int step) {
  write_vtk_acc(b, fr, n, step);
}

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);
  if (a == Args::error) return 1;
  if (a == Args::help) return 0;

  setenv("ITYR_ITO_SUSPENDED_THREAD_ALLOCATOR_SIZE", "536870912", 1);
  setenv("ITYR_ITO_THREAD_STATE_ALLOCATOR_SIZE", "536870912", 1);
  setenv("ITYR_ITO_WSQUEUE_CAPACITY", "32768", 1);

  ityr::init();

  w_total.start();
  w_input.start();
  auto in = ityr::root_exec([=]() -> pair<Body64*, size_t> {
    vector<Body64> v;
    if (!read_input_t(cfg.input, v)) return {nullptr, 0};
    auto* buf = (Body64*)ityr::ito::alloc_future_data(v.size() * sizeof(Body64));
    memcpy(buf, v.data(), v.size() * sizeof(Body64));
    return {buf, v.size() * sizeof(Body64)};
  });
  ityr::ito::flush_copy_cache();
  w_input.stop();
  size_t n = in.second / sizeof(Body64);
  if (n == 0) {
    ityr::fini();
    return 1;
  }

  // Fixed slice layout + per-node live slice / keys replica / scratch
  // (allocated once; the live slice is maintained in place by kick + scatter,
  // the keys replica is refreshed every step). The master additionally holds
  // the transient build buffers.
  BuildReplicas rep;
  init_build_replicas(rep, n, cfg.build_cutoff);
  MasterBuild m;
  if (ityr::is_master()) {
    m.keys = rep.keys;   // the master's keys replica is the merge output
    m.bodies = (Body64*)ityr::ito::alloc_future_data(n * sizeof(Body64));
    m.compact = (CompactBody*)ityr::ito::alloc_future_data(n * sizeof(CompactBody));
    m.vel_scratch = (double*)ityr::ito::alloc_future_data(n * 4 * sizeof(double));
    m.owns = true;
  }

  // One persistent accel buffer per node, allocated once. The force leaves
  // write into it directly, combine_accel() completes it across nodes, and the
  // kick + master's ep/output read it.
  NodeRepPointers accs{};
  void* my_acc = nullptr;
  size_t acc_bytes = 0;
  bool owns_acc = false;
  void* my_cmask = nullptr;
  bool owns_cmask = false;
  alloc_node_accel(n, accs, my_acc, owns_acc, acc_bytes, my_cmask, owns_cmask);

  // Per-node persistent compact replica: every intra rank folds its own share
  // into it, so all ranks of a node must share one buffer. On the master it is
  // m.compact, and the alias must happen before the pointer broadcast or the
  // peers write an orphaned buffer.
  void* my_compact = nullptr;
  if (ityr::common::topology::intra_my_rank() == 0 && !ityr::is_master()) {
    my_compact = ityr::ito::alloc_future_data(n * sizeof(CompactBody));
  }
  if (ityr::is_master()) my_compact = m.compact;
  CompactBody* compact_table[MAX_INTER_NODES] = {};
  {
    uintptr_t cbuf[MAX_INTER_NODES] = {};
    const int inter = ityr::common::topology::inter_my_rank();
    if (ityr::common::topology::intra_my_rank() == 0)
      cbuf[inter] = (uintptr_t)my_compact;
    broadcast_uintptr_table(cbuf);
    my_compact = (void*)cbuf[inter];
    for (int i = 0; i < MAX_INTER_NODES; ++i) compact_table[i] = (CompactBody*)cbuf[i];
  }

  // Distribute the input across the live slices (read on one node, distribute).
  scatter_input(in.first, rep.L, rep.live);

  if (cfg.brute_force) {
    Bounds bd = bounds_from_replica(rep);
    BuildResult br = build_step(rep, n, true, in.first, bd, compact_table);
    NodeReplicas reps = make_node_replicas(br, (CompactBody*)my_compact, accs);
    zero_node_accel(accs, n);
    w_forces.start();
    ityr::root_exec([=] {
      forces_phase(reps, n, br.box_side, br.xmin, br.ymin, br.zmin);
    });
    ityr::ito::flush_copy_cache();
    w_forces.stop();
    combine_accel(accs, n, cfg.chunk);
    free_node_replicas(reps);
    // The walk's total potential uses the shared canonical blocked sum (the
    // same code path as the main loop).
    double ep_walk = 0;
    if (ityr::is_master()) {
      vector<double> pot(n);
      const ForceResult* bacc = accs.acc[0];
      for (size_t i = 0; i < n; ++i) pot[i] = bacc[i].pot;
      ep_walk = 0.5 * blocked_sum(pot.data(), n);
    }
    int code = 0;
    if (ityr::is_master()) {
      // The brute-force check needs the walk results inside the full Body (the
      // parallel path normally keeps them in the accel buffer), so materialize
      // a temporary copy from the master's gathered bodies.
      gather_full_bodies(rep.L, rep.live, m.bodies);
      vector<Body> fat(n);
      for (size_t i = 0; i < n; ++i) {
        const Body64& s = m.bodies[i];
        Body& o = fat[i];
        o.id = s.id; o.mass = s.mass;
        o.x = s.x; o.y = s.y; o.z = s.z;
        o.vx = s.vx; o.vy = s.vy; o.vz = s.vz;
        o.ax = accs.acc[0][i].ax; o.ay = accs.acc[0][i].ay;
        o.az = accs.acc[0][i].az; o.pot = accs.acc[0][i].pot;
      }
      code = brute_force_check(fat.data(), n, ep_walk);
    }
    collective_timed_cat(tc_misc, [&] { ityr::common::mpi_bcast(&code, 1, 0, MPI_COMM_WORLD); });
    if (owns_acc) ityr::ito::dealloc_future_data(my_acc, acc_bytes);
    if (owns_cmask) ityr::ito::dealloc_future_data(my_cmask,
                       ((n + cfg.chunk - 1) / cfg.chunk + 63) / 64 * sizeof(uint64_t));
    if (rep.owns) {
      ityr::ito::dealloc_future_data(rep.live, rep.live_bytes);
      ityr::ito::dealloc_future_data(rep.keys, rep.keys_bytes);
      ityr::ito::dealloc_future_data(rep.run, rep.L.nslice * sizeof(KeyPair));
      ityr::ito::dealloc_future_data(rep.pairs, rep.L.nslice * sizeof(GatherPair));
      ityr::ito::dealloc_future_data(rep.recv, rep.L.nslice * sizeof(GatherPair));
      ityr::ito::dealloc_future_data(rep.gkeys, n * sizeof(uint64_t));
      ityr::ito::dealloc_future_data(rep.krun, rep.L.nslice * sizeof(uint64_t));
      if (rep.mbufA) ityr::ito::dealloc_future_data(rep.mbufA, n * sizeof(MergeTriple));
      if (rep.mbufB) ityr::ito::dealloc_future_data(rep.mbufB, n * sizeof(MergeTriple));
    }
    if (m.owns) {
      ityr::ito::dealloc_future_data(m.bodies, n * sizeof(Body64));
      ityr::ito::dealloc_future_data(m.compact, n * sizeof(CompactBody));
      ityr::ito::dealloc_future_data(m.vel_scratch, n * 4 * sizeof(double));
    }
    ityr::fini();
    return code;
  }

  double box_side;
  double ek = 0, ep = 0, et0 = 0;
  for (int step = 0; step <= cfg.steps; ++step) {
    // The next build's bounds come from the fused kick (the drift loop already
    // touches every position). min/max are exact, so the Allreduce result is
    // the same as a dedicated bounds pass. Step 0 has no kick (positions are
    // the input), so its bounds come from the local slice scan.
    Bounds bd;
    if (step > 0) {
      w_integrate.start();
      // One pass does the previous step's trailing half-kick, this step's
      // leading half-kick (same acc) and the drift. At step 1 there is no
      // trailing kick to fuse (a_0 is applied once).
      Extremes ex;
      kick_live(rep, accs.acc, step == 1 ? 1 : 2, true, &ex);
      double loc[6] = {-ex.xmin, ex.xmax, -ex.ymin, ex.ymax, -ex.zmin, ex.zmax};
      double glb[6];
      collective_timed([&] { MPI_Allreduce(loc, glb, 6, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); });
      // MPI_MAX on negated minima recovers the true minima.
      bd = bounds_from_extremes({-glb[0], glb[1], -glb[2], glb[3], -glb[4], glb[5]});
      w_integrate.stop();
    } else {
      bd = bounds_from_replica(rep);
    }

    BuildResult br = build_step(rep, n, step == 0, in.first, bd, compact_table);
    box_side = br.box_side;
    NodeReplicas reps = make_node_replicas(br, (CompactBody*)my_compact, accs);

    // Must run after the kick above has consumed the previous step's accelerations.
    zero_node_accel(accs, n);
    w_forces.start();
#ifdef ITYR_PROF
    t_pcompute.start();
#endif
    ityr::root_exec([=] {
      forces_phase(reps, n, box_side, br.xmin, br.ymin, br.zmin);
    });
#ifdef ITYR_PROF
    t_pcompute.stop();
    t_pflush.start();
#endif
    ityr::ito::flush_copy_cache();
#ifdef ITYR_PROF
    t_pflush.stop();
#endif
    free_node_replicas(reps);

#ifdef ITYR_PROF
    t_pcombine.start();
#endif
    combine_accel(accs, n, cfg.chunk);
#ifdef ITYR_PROF
    t_pcombine.stop();
#endif

    // ep/ek: the canonical blocked sums (see energy_phase). Only needed on
    // report steps.
    if (step == 0 ||
        (cfg.energy_every > 0 && (step % cfg.energy_every == 0 || step == cfg.steps))) {
#ifdef ITYR_PROF
      t_penergy.start();
#endif
      auto [ek_sum, ep_sum] = energy_phase(rep, accs, m, n);
      ep = ep_sum;
      ek = ek_sum;
#ifdef ITYR_PROF
      t_penergy.stop();
#endif
    }
    w_forces.stop();

    if (step == 0) {
      et0 = ek + ep;
      if (ityr::is_master()) {
        printf("# N=%d theta=%.3f eps2=%.1e leaf=%d dt=%.3e G=%.3f chunk=%d ranks=%d simd=%s\n",
               (int)n, cfg.theta, cfg.eps2, cfg.leaf, cfg.dt, cfg.G, (int)cfg.chunk,
               (int)ityr::n_ranks(), forceN_isa_name());
        printf("%8s %18s %18s %18s %12s\n", "step", "EK", "EP", "ET", "dET/ET");
        printf("%8d %18.8e %18.8e %18.8e %12s\n", 0, ek, ep, et0, "-");
      }
    } else if (ityr::is_master() &&
               (cfg.energy_every > 0 && (step % cfg.energy_every == 0 || step == cfg.steps)))
      report_energy(step, ek, ep, et0);

    if (cfg.output && (step == 0 || (cfg.output_every > 0 && step % cfg.output_every == 0))) {
      gather_full_bodies(rep.L, rep.live, m.bodies);
      w_output.start();
      ityr::root_exec([=] {
        if (ityr::is_master()) output_phase(m.bodies, accs.acc[0], n, step);
      });
      ityr::ito::flush_copy_cache();
      w_output.stop();
    }

    // The trailing half-kick is fused into the next iteration's kick_live;
    // the final one is applied after the loop.
  }

  // Final trailing half-kick, so the end state matches the two-pass schedule.
  // Nothing reads it (output/energy for the last step already happened above),
  // but a half-finished integrator would bite anyone adding a post-loop dump.
  if (cfg.steps > 0) {
    w_integrate.start();
    kick_live(rep, accs.acc, 1, false);
    w_integrate.stop();
  }

  if (owns_acc) ityr::ito::dealloc_future_data(my_acc, acc_bytes);
  if (owns_cmask) ityr::ito::dealloc_future_data(my_cmask,
                     ((n + cfg.chunk - 1) / cfg.chunk + 63) / 64 * sizeof(uint64_t));
  if (rep.owns) {
    ityr::ito::dealloc_future_data(rep.live, rep.live_bytes);
    ityr::ito::dealloc_future_data(rep.keys, rep.keys_bytes);
    ityr::ito::dealloc_future_data(rep.run, rep.L.nslice * sizeof(KeyPair));
    ityr::ito::dealloc_future_data(rep.pairs, rep.L.nslice * sizeof(GatherPair));
    ityr::ito::dealloc_future_data(rep.recv, rep.L.nslice * sizeof(GatherPair));
    ityr::ito::dealloc_future_data(rep.gkeys, n * sizeof(uint64_t));
    ityr::ito::dealloc_future_data(rep.krun, rep.L.nslice * sizeof(uint64_t));
    if (rep.mbufA) ityr::ito::dealloc_future_data(rep.mbufA, n * sizeof(MergeTriple));
    if (rep.mbufB) ityr::ito::dealloc_future_data(rep.mbufB, n * sizeof(MergeTriple));
  }
  if (ityr::common::topology::intra_my_rank() == 0 && my_compact != m.compact)
    ityr::ito::dealloc_future_data(my_compact, n * sizeof(CompactBody));
  if (m.owns) {
    ityr::ito::dealloc_future_data(m.bodies, n * sizeof(Body64));
    ityr::ito::dealloc_future_data(m.compact, n * sizeof(CompactBody));
    ityr::ito::dealloc_future_data(m.vel_scratch, n * 4 * sizeof(double));
  }

  w_total.stop();
  double wloc[7] = {w_input.ms, w_build.ms, w_forces.ms, w_integrate.ms, w_output.ms,
                    w_total.ms, t_collective.ms};
  double wmax[7];
  collective_timed([&] { MPI_Allreduce(wloc, wmax, 7, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); });
  if (ityr::is_master() && cfg.timers)
    printf("timings (ms): input %.2f  build %.2f  forces+energy %.2f  integrate %.2f  "
           "collective %.2f (max-rank %.2f | dist %.2f bar %.2f energy %.2f ptr %.2f misc %.2f)"
           "  output %.2f  total %.2f\n",
           wmax[0], wmax[1], wmax[2], wmax[3], t_collective.ms, wmax[6],
           tc_dist.ms, tc_bar.ms, tc_energy.ms, tc_ptr.ms, tc_misc.ms, wmax[4], wmax[5]);

#ifdef ITYR_PROF
  {
    double loc[10] = {t_pkeys.ms, t_pcompact.ms, t_pdist.ms, t_ptree.ms,
                      t_pcompute.ms, t_pflush.ms, t_pcombine.ms, t_penergy.ms, t_pkloop.ms,
                      t_pkflush.ms};
    double wall[10];
    MPI_Allreduce(loc, wall, 10, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (ityr::is_master() && cfg.timers) {
      printf("PROF build (ms): keys %.2f  compact %.2f  dist %.2f  tree %.2f\n",
             wall[0], wall[1], wall[2], wall[3]);
      printf("PROF tree (ms): plan %.2f  fold+put %.2f  assembly %.2f\n",
             t_splan.ms, t_sfold.ms, t_sasm.ms);
      printf("PROF keys (ms): make+sort %.2f  kway %.2f  tree %.2f  pairs %.2f\n",
             t_pk1.ms, t_pk2.ms, t_pk3.ms, t_pk4.ms);
      printf("PROF dist (ms): agather %.2f  merge_all %.2f  alltoall %.2f  alltoallv %.2f  search %.2f  nmerge %.2f\n",
             t_pagather.ms, t_pmerge_all.ms, t_palltoall.ms, t_palltoallv.ms,
             t_psearch.ms, t_pnmerge.ms);
      printf("PROF forces (ms): compute %.2f  flush %.2f  combine %.2f  energy %.2f  kick %.2f  kflush %.2f\n",
             wall[4], wall[5], wall[6], wall[7], wall[8], wall[9]);
    }
  }
#endif

  ityr::fini();
  return 0;
}
