#include "load_balancing.h"
#include <numeric>
#include <algorithm>
#include <cassert>


void update_rank_domains(
    int size,
    const std::vector<uint64_t>& codes,
    int bucket_bits,
    std::vector<std::vector<uint64_t>>& out_rank_domain_keys,
    std::vector<std::pair<long long, int>>& out_global_hist,
    std::vector<int>& out_splitters,
    hpxc::Ctx& ctx) {

    out_splitters.clear();

    // With a single locality there is no other rank to balance against or
    // send LET data to: every bucket trivially belongs to rank 0, with no
    // histogram, collective, or splitter computation needed to know that.
    if (size <= 1) {
        out_rank_domain_keys.assign(1, std::vector<uint64_t>());
        out_global_hist.clear();
        return;
    }

    int NUM_BUCKETS  = 1 << bucket_bits;

    // build local histogram and reduce to global
    std::vector<long long> local_hist(NUM_BUCKETS, 0);
    for (uint64_t key : codes) {
        unsigned bucket_idx = key >> (63 - bucket_bits);
        ++local_hist[bucket_idx];
    }
    std::vector<long long> global_hist = hpx::collectives::all_reduce(
        ctx.comm_hist, local_hist, hpxc::VecSumLL{},
        hpxc::this_site_arg(ctx.rank), hpxc::generation_arg(ctx.gen_hist++)).get();

    // choose P‑1 splitter buckets (inclusive)
    long long total_particles = std::accumulate(global_hist.begin(),
                                                global_hist.end(), 0LL);
    const long long ideal = (total_particles + size - 1) / size; // ceil

    out_splitters.reserve(size - 1);

    long long run_sum = 0;
    for (int i = 0; i < NUM_BUCKETS && out_splitters.size() < static_cast<size_t>(size - 1); ++i) {
        run_sum += global_hist[i];

        if (run_sum >= static_cast<long long>(out_splitters.size() + 1) * ideal) {
            out_splitters.push_back(i);
        }
    }

    // NOTE: The output parameters are renamed here, but the logic is identical.
    out_global_hist.resize(NUM_BUCKETS);
    out_rank_domain_keys.assign(size, std::vector<uint64_t>());
    for (int i = 0; i < NUM_BUCKETS; ++i) {
        int dest_rank = std::upper_bound(out_splitters.begin(), out_splitters.end(), i) - out_splitters.begin();

        out_global_hist[i].first = global_hist[i];
        out_global_hist[i].second = dest_rank;

        // Only consider buckets that actually contain particles
        if (global_hist[i] > 0) {
            uint64_t bucket_prefix = static_cast<uint64_t>(i) << (63 - bucket_bits);
            out_rank_domain_keys[dest_rank].push_back(bucket_prefix);
        }
    }
}

void rebalance_bodies(
    int rank, int size,
    const std::vector<int>& splitters,
    const std::vector<uint64_t>& codes,
    std::vector<Position> &local_pos,
    std::vector<double> &local_mass,
    std::vector<Velocity> &local_vel,
    std::vector<uint64_t> &local_ids,
    int bucket_bits,
    hpxc::Ctx& ctx) {

    (void)rank;

    // bucket each local body into its destination rank
    std::vector<std::vector<hpxc::TransferBody>> send_per_dest(size);
    for (size_t i = 0; i < local_pos.size(); ++i) {
        unsigned b = codes[i] >> (63 - bucket_bits);
        int dest = std::upper_bound(splitters.begin(), splitters.end(),
                                    static_cast<int>(b)) - splitters.begin();
        send_per_dest[dest].push_back({local_pos[i], local_mass[i], local_vel[i], local_ids[i]});
    }

    auto recv_per_src = hpx::collectives::all_to_all(ctx.comm_migrate, std::move(send_per_dest),
        hpxc::this_site_arg(ctx.rank), hpxc::generation_arg(ctx.gen_migrate++)).get();

    local_pos.clear();
    local_mass.clear();
    local_vel.clear();
    local_ids.clear();
    for (auto& incoming : recv_per_src) {
        for (auto& m : incoming) {
            local_pos.push_back(m.pos);
            local_mass.push_back(m.mass);
            local_vel.push_back(m.vel);
            local_ids.push_back(m.id);
        }
    }
}
