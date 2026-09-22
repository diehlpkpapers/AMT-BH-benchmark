#pragma once

// Groups HPX worker threads by NUMA domain: first-touch grow() places a
// flat array's pages on the domain that will mostly use them, and
// domain-sliced dispatch runs the hot loops matching that placement.
//
// Why: measured on buran00, Interact's DRAM traffic is 99.66% local at
// --hpx:threads=24 (one socket) but 38.4% remote at 32.
//
// Mechanism: HPX's scheduler ignores a NUMA hint, so this hints one
// representative worker per domain and relies on its same-domain
// work-stealing tier to cover the rest.

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <hpx/execution.hpp>
#include <hpx/runtime_local/get_os_thread_count.hpp>
#include <hpx/runtime_local/runtime_local_fwd.hpp>
#include <hpx/coroutines/thread_enums.hpp>

class NumaLayout {
public:
    // Fixed once --hpx:threads is parsed, so a static suffices (instance()).
    NumaLayout() {
        const std::size_t n = hpx::get_os_thread_count();
        const hpx::threads::topology& topo = hpx::threads::get_topology();

        std::vector<std::size_t> node_of(n);
        std::size_t max_node = 0;
        for (std::size_t i = 0; i < n; ++i) {
            node_of[i] = topo.get_numa_node_number(i);
            max_node = std::max(max_node, node_of[i]);
        }

        domain_workers_.assign(max_node + 1, {});
        for (std::size_t i = 0; i < n; ++i) domain_workers_[node_of[i]].push_back(i);

        // Drop empty domains so callers can index [0, num_domains) densely.
        std::vector<std::vector<std::size_t>> nonempty;
        for (auto& w : domain_workers_) if (!w.empty()) nonempty.push_back(std::move(w));
        domain_workers_ = std::move(nonempty);

        domain_exec_.reserve(domain_workers_.size());
        for (const auto& workers : domain_workers_) {
            domain_exec_.emplace_back(
                hpx::threads::thread_priority::normal,
                hpx::threads::thread_stacksize::default_,
                hpx::threads::thread_schedule_hint(
                    hpx::threads::thread_schedule_hint_mode::thread,
                    static_cast<std::int16_t>(workers.front())));
        }
    }

    static const NumaLayout& instance() {
        static const NumaLayout layout;
        return layout;
    }

    std::size_t num_domains() const { return domain_workers_.size(); }
    std::size_t workers_in_domain(std::size_t d) const { return domain_workers_[d].size(); }

    // Hints to one representative worker; same-domain stealing spreads it
    // across the rest (see file header).
    const hpx::execution::parallel_executor& domain_executor(std::size_t d) const {
        return domain_exec_[d];
    }

    // Contiguous ranges sized by each domain's worker count. Deterministic,
    // so first-touch and later dispatch stay matched.
    std::vector<std::pair<std::size_t, std::size_t>> splitRange(std::size_t n) const {
        std::vector<std::pair<std::size_t, std::size_t>> ranges;
        ranges.reserve(domain_workers_.size());
        std::size_t total_workers = 0;
        for (const auto& w : domain_workers_) total_workers += w.size();

        std::size_t begin = 0;
        std::size_t assigned_workers = 0;
        for (std::size_t d = 0; d < domain_workers_.size(); ++d) {
            assigned_workers += domain_workers_[d].size();
            // Last domain absorbs any rounding remainder so the ranges
            // always sum to exactly n.
            std::size_t end = (d + 1 == domain_workers_.size())
                ? n
                : (n * assigned_workers) / total_workers;
            end = std::max(end, begin);   // a domain can legitimately get 0 elements at small n
            ranges.emplace_back(begin, end);
            begin = end;
        }
        return ranges;
    }

    // Value-constructs the new tail in per-domain slices - that's the first
    // write to those pages, so a plain resize() would misplace them all.
    template <class T>
    void growWithFirstTouch(std::vector<T>& v, std::size_t newN) const {
        const std::size_t oldN = v.size();
        if (newN <= oldN) { v.resize(newN); return; }   // shrink: no new pages, no touch needed

        v.reserve(newN);
        std::size_t next = oldN;
        auto ranges = splitRange(newN);
        for (std::size_t d = 0; d < domain_workers_.size(); ++d) {
            std::size_t target_end = ranges[d].second;
            if (target_end <= next) continue;   // this domain's whole share is below oldN already
            hpx::async(domain_exec_[d], [&v, target_end]() { v.resize(target_end); }).get();
            next = target_end;
        }
    }

private:
    std::vector<std::vector<std::size_t>> domain_workers_;
    std::vector<hpx::execution::parallel_executor> domain_exec_;
};
