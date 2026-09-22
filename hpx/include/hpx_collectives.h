#pragma once

// Thin shim providing MPI-collective-shaped helpers built on top of HPX's
// own distributed collectives (hpx::collectives). Despite lacking an
// Alltoallv/Allgatherv-named API, all_to_all/all_gather/gather/scatter are
// fully generic over their element type T; passing T = std::vector<U> gives
// each site an independently variable-length payload (hpx::serialization
// length-prefixes vectors regardless of size), which is exactly the MPI *v
// semantics. No custom point-to-point plumbing is needed.

#include <hpx/future.hpp>
#include <hpx/collectives/create_communicator.hpp>
#include <hpx/collectives/all_reduce.hpp>
#include <hpx/collectives/all_to_all.hpp>
#include <hpx/collectives/all_gather.hpp>
#include <hpx/collectives/gather.hpp>
#include <hpx/collectives/scatter.hpp>
#include <hpx/collectives/broadcast.hpp>
#include <hpx/collectives/barrier.hpp>

#include "body.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace hpxc {

// Bundles the four per-body parallel arrays (position, mass, velocity, id)
// into a single payload so migrating/scattering a body is one transfer
// instead of four.
struct TransferBody {
    Position pos;
    double   mass;
    Velocity vel;
    uint64_t id;
};

}    // namespace hpxc

HPX_IS_BITWISE_SERIALIZABLE(hpxc::TransferBody);

namespace hpxc {

using hpx::collectives::communicator;
using hpx::collectives::create_communicator;
using hpx::collectives::generation_arg;
using hpx::collectives::num_sites_arg;
using hpx::collectives::this_site_arg;

// ---------------------------------------------------------------------------
// Named, default-constructible reduction functors.
// (HPX serializes the op passed to all_reduce as part of the action call;
// capturing/closure lambdas are not default-constructible enough for that,
// so plain function objects are used instead - same reason std::plus<T> works.)
// ---------------------------------------------------------------------------
struct MinArr3 {
    std::array<double, 3> operator()(
        std::array<double, 3> a, std::array<double, 3> const& b) const {
        a[0] = std::min(a[0], b[0]);
        a[1] = std::min(a[1], b[1]);
        a[2] = std::min(a[2], b[2]);
        return a;
    }
};

struct MaxArr3 {
    std::array<double, 3> operator()(
        std::array<double, 3> a, std::array<double, 3> const& b) const {
        a[0] = std::max(a[0], b[0]);
        a[1] = std::max(a[1], b[1]);
        a[2] = std::max(a[2], b[2]);
        return a;
    }
};

struct VecSumLL {
    std::vector<long long> operator()(
        std::vector<long long> a, std::vector<long long> const& b) const {
        for (std::size_t i = 0; i < a.size(); ++i) a[i] += b[i];
        return a;
    }
};

struct SumArr2 {
    std::array<double, 2> operator()(
        std::array<double, 2> a, std::array<double, 2> const& b) const {
        a[0] += b[0];
        a[1] += b[1];
        return a;
    }
};

// Number of PhaseTimers::Phase slots (src/main.cpp) - must match
// PhaseTimers::N there; a static_assert next to that enum enforces it.
inline constexpr std::size_t kTimerSlots = 8;

struct MaxTimers {
    std::array<double, kTimerSlots> operator()(
        std::array<double, kTimerSlots> a, std::array<double, kTimerSlots> const& b) const {
        for (std::size_t i = 0; i < kTimerSlots; ++i) a[i] = std::max(a[i], b[i]);
        return a;
    }
};

// Flattens the per-site vectors an all_gather<std::vector<T>> returns into a
// single concatenated vector, in rank order.
template <typename T>
std::vector<T> flatten(std::vector<std::vector<T>> per_site) {
    std::vector<T> out;
    for (auto& v : per_site) out.insert(out.end(), v.begin(), v.end());
    return out;
}

// ---------------------------------------------------------------------------
// Per-locality collective context: every named communicator used by the
// simulation is created exactly once (a cheap handle, not a per-call setup)
// and reused for the whole run.
//
// HPX's generation_arg is NOT a free-form disambiguating key: each
// communicator gates on it internally with a counter that only ever advances
// by exactly 1 per completed round. Passing anything other than "the Nth
// invocation of this exact communicator, for N = 1, 2, 3, ..." either
// collides with a still-pending round or asks the gate to count up to a
// generation it will never organically reach (an unrecoverable hang). So
// every communicator here owns its own strictly-sequential counter instead
// of callers deriving a number (e.g. from the simulation step) themselves.
// ---------------------------------------------------------------------------
struct Ctx {
    std::size_t rank;
    std::size_t size;

    communicator comm_bbox_min;
    communicator comm_bbox_max;
    communicator comm_hist;
    communicator comm_bcast_ntotal;
    communicator comm_error_sum;
    communicator comm_scatter_init;
    communicator comm_migrate;
    communicator comm_let;
    communicator comm_fulltree;
    communicator comm_gather_ref;
    communicator comm_energy;
    communicator comm_energy_gather;
    communicator comm_timers;
    // Also used by the phase timers (PhaseTimers::start in main.cpp) as a
    // start-of-phase synchronization point, in addition to its original use
    // as the visualization-step sync.
    hpx::distributed::barrier barrier;

    std::size_t gen_bbox_min = 1;
    std::size_t gen_bbox_max = 1;
    std::size_t gen_hist = 1;
    std::size_t gen_bcast_ntotal = 1;
    std::size_t gen_error_sum = 1;
    std::size_t gen_scatter_init = 1;
    std::size_t gen_migrate = 1;
    std::size_t gen_let = 1;
    std::size_t gen_fulltree = 1;
    std::size_t gen_gather_ref = 1;
    std::size_t gen_energy = 1;
    std::size_t gen_energy_gather = 1;
    std::size_t gen_timers = 1;

    Ctx(std::size_t rank_, std::size_t size_)
        : rank(rank_)
        , size(size_)
        , comm_bbox_min(create_communicator(
              "bbox_min", num_sites_arg(size_), this_site_arg(rank_)))
        , comm_bbox_max(create_communicator(
              "bbox_max", num_sites_arg(size_), this_site_arg(rank_)))
        , comm_hist(create_communicator(
              "hist", num_sites_arg(size_), this_site_arg(rank_)))
        , comm_bcast_ntotal(create_communicator("bcast_ntotal",
              num_sites_arg(size_), this_site_arg(rank_)))
        , comm_error_sum(create_communicator("error_sum",
              num_sites_arg(size_), this_site_arg(rank_)))
        , comm_scatter_init(create_communicator("scatter_init",
              num_sites_arg(size_), this_site_arg(rank_)))
        , comm_migrate(create_communicator(
              "migrate", num_sites_arg(size_), this_site_arg(rank_)))
        , comm_let(create_communicator(
              "let", num_sites_arg(size_), this_site_arg(rank_)))
        , comm_fulltree(create_communicator(
              "fulltree", num_sites_arg(size_), this_site_arg(rank_)))
        , comm_gather_ref(create_communicator("gather_ref",
              num_sites_arg(size_), this_site_arg(rank_)))
        , comm_energy(create_communicator(
              "energy", num_sites_arg(size_), this_site_arg(rank_)))
        , comm_energy_gather(create_communicator("energy_gather",
              num_sites_arg(size_), this_site_arg(rank_)))
        , comm_timers(create_communicator(
              "timers", num_sites_arg(size_), this_site_arg(rank_)))
        , barrier("main_barrier", size_, rank_) {}
};

}    // namespace hpxc
