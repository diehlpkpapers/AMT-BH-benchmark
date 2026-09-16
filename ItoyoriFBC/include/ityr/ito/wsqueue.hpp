#pragma once

#include <atomic>
#include <optional>
#include <type_traits>
#include <algorithm>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/mpi_rma.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/global_lock.hpp"

namespace ityr::ito {

class wsqueue_full_exception : public std::exception {
public:
  const char* what() const noexcept override { return "Work stealing queue is full."; }
};


/**
 * @brief A distributed work-stealing queue implementation supporting multiple queues per Worker.
 *
 * The `wsqueue` class provides a concurrent, distributed work-stealing queue suitable for parallel
 * and distributed task scheduling. Each instance manages one or more queues, each of which supports
 * local push/pop operations (LIFO order) and remote steal operations (FIFO order) via MPI RMA.
 * It is possible to init multiple queues per worker, but this is not used currently. 
 * The var name idx, often refers to the queue not to the place in the queue.
 *
 * Key features:
 * - Multiple queues per process, each independently accessible.
 * - Lock-based synchronization for safe concurrent access and work stealing.
 * - Exception handling for queue overflow.
 * - Efficient memory management using MPI windows for both queue state and entries.
 * - Support for both local and remote operations, including stealing and aborting steals.
 * - Utility functions for iterating over entries, checking queue state, and concurrent testing.
 *
 * @tparam Entry The type of entries stored in the queue. Must be trivially copyable for MPI RMA.
 */
template <typename Entry>
class wsqueue {
public:
  wsqueue(int n_entries, int n_queues = 1)
    : n_entries_(n_entries),
      n_queues_(n_queues),
      initial_pos_(0),
      queue_state_win_(common::topology::mpicomm(), n_queues_ * 2, initial_pos_),
      entries_win_(common::topology::mpicomm(), n_entries_ * n_queues_),
      queue_lock_(n_queues_),
      local_empty_(n_queues_, false) {}

  void push_bottom(const Entry& entry, int idx = 0) {
    ITYR_CHECK(idx < n_queues_);

    // We MUST lock because we are modifying 'base', which thieves are accessing
    queue_lock_.priolock(common::topology::my_rank(), idx);

    queue_state& qs = local_queue_state(idx);
    auto entries = local_entries(idx);

    int b = qs.base.load(std::memory_order_relaxed);
    int t = qs.top.load(std::memory_order_relaxed);

    if (b == 0) {
      // No space at the bottom (index 0).
      // Check if we can shift everything up by 1.
      if (t == n_entries_) {
        queue_lock_.unlock(common::topology::my_rank(), idx);
        throw wsqueue_full_exception{};
      }

      // Shift entries [b, t) to [b + 1, t + 1)
      // We use move_backward because the range overlaps to the right
      std::move_backward(&entries[b], &entries[t], &entries[t + 1]);

      // Update top and base to reflect the shift
      b = b + 1;
      t = t + 1;
      qs.top.store(t, std::memory_order_relaxed);
      qs.base.store(b, std::memory_order_relaxed);
    }

    // Now there is guaranteed space at b - 1
    int target_idx = b - 1;
    entries[target_idx] = entry;

    // Update base to include the new bottom element
    qs.base.store(target_idx, std::memory_order_release);

    local_empty_[idx] = false;

    queue_lock_.unlock(common::topology::my_rank(), idx);
  }

  void push(const Entry& entry, int idx = 0) {

    ITYR_CHECK(idx < n_queues_);

    queue_state& qs = local_queue_state(idx);
    auto entries = local_entries(idx);

    int t = qs.top.load(std::memory_order_relaxed);

    if (t == n_entries_) {
      queue_lock_.priolock(common::topology::my_rank(), idx);

      int b = qs.base.load(std::memory_order_relaxed);
      int offset = -(b + 1) / 2;
      move_entries(offset, idx);
      t += offset;

      queue_lock_.unlock(common::topology::my_rank(), idx);
    }

    entries[t] = entry;

    qs.top.store(t + 1, std::memory_order_release);


    local_empty_[idx] = false;


    struct wsq_e {
      void*       frame_base;
      std::size_t frame_size;
    };

    wsq_e *wsq_entry = (wsq_e*)&entry;
    ityr::common::verbose("pushed wsqueue entry into queue: frame_base=%p, frame_size=%u", wsq_entry->frame_base, wsq_entry->frame_size);
    context_frame *cf = (context_frame*)wsq_entry->frame_base;
    ityr::common::verbose("pushed wsqueue entry into queue: rbp=%p, rsp=%p, rip=%p, parent_frame=%p", cf->rbp, cf->rsp, cf->rip, cf->parent_frame);
  }

  template <bool EnsureEmpty = true>
  std::optional<Entry> pop(int idx = 0) {

    ITYR_CHECK(idx < n_queues_);

    queue_state& qs = local_queue_state(idx);

    if (local_empty_[idx]) {
      return std::nullopt;
    }
  

    // We often need to ensure the queue is truly empty because the thief might be concurrently
    // operating on the queue (the stack copy might be ongoing) and can abort stealing after
    // updating the `base` variable.
    // TODO: any better way to handle this ordering?
    if constexpr (!EnsureEmpty) {
      if (qs.empty()) {
        return std::nullopt;
      }
    }

    std::optional<Entry> ret;
    auto entries = local_entries(idx);

    int t = qs.top.load(std::memory_order_relaxed) - 1;
    qs.top.store(t, std::memory_order_relaxed);

    std::atomic_thread_fence(std::memory_order_seq_cst);

    int b = qs.base.load(std::memory_order_relaxed);

    if (b <= t) {
      ret = entries[t];
    } else {
      qs.top.store(t + 1, std::memory_order_relaxed);

      queue_lock_.priolock(common::topology::my_rank(), idx);

      qs.top.store(t, std::memory_order_relaxed);
      int b = qs.base.load(std::memory_order_relaxed);

      if (b < t) {
        ret = entries[t];
      } else if (b == t) {
        ret = entries[t];

        qs.top.store(initial_pos_, std::memory_order_relaxed);
        qs.base.store(initial_pos_, std::memory_order_relaxed);

        // Once we confirm that this queue is empty by taking the lock, later pop operations will
        // always fail if the "pass" operation is not allowed for the queue
        local_empty_[idx] = true;

      } else {
        ret = std::nullopt;

        qs.top.store(initial_pos_, std::memory_order_relaxed);
        qs.base.store(initial_pos_, std::memory_order_relaxed);

        local_empty_[idx] = true;
        
      }

      queue_lock_.unlock(common::topology::my_rank(), idx);
    }

    return ret;
  }

  std::pair<std::optional<Entry>, int> steal_nolock_copy(common::topology::rank_t target_rank, int idx = 0) {

    ITYR_CHECK(idx < n_queues_);

    ITYR_CHECK(queue_lock_.is_locked(target_rank, idx));

    std::optional<Entry> ret;

    int b = common::mpi_get_value<int>(target_rank, queue_state_base_disp(idx), queue_state_win_.win());
    int t = common::mpi_get_value<int>(target_rank, queue_state_top_disp(idx), queue_state_win_.win());

    ityr::common::verbose("steal_nolock b=%d t=%d", b, t);

    if (b < t) {
      ret = common::mpi_get_value<Entry>(target_rank, entries_disp(b, idx), entries_win_.win());
      struct wsq_e {
        void*       frame_base;
        std::size_t frame_size;
      };

      wsq_e *wsq_entry = (wsq_e*)&ret.value();
      ityr::common::verbose("steal_nolock wsqueue entry from rank %d: frame_base=%p, frame_size=%u", target_rank, wsq_entry->frame_base, wsq_entry->frame_size);
    } else {
      ret = std::nullopt;
    }

    return std::make_pair(ret, b);
  }

  bool steal_nolock_remove(common::topology::rank_t target_rank, int b_correct, int idx = 0) {

    ITYR_CHECK(idx < n_queues_);

    ITYR_CHECK(queue_lock_.is_locked(target_rank, idx));

    int b = common::mpi_atomic_faa_value<int>(1, target_rank, queue_state_base_disp(idx), queue_state_win_.win());
    int t = common::mpi_get_value<int>(target_rank, queue_state_top_disp(idx), queue_state_win_.win());

    if (b != b_correct || b >= t) {
      common::mpi_atomic_faa_value<int>(-1, target_rank, queue_state_base_disp(idx), queue_state_win_.win());
      return false;
    } else {
      return true;
    }
  }

  std::optional<Entry> steal_nolock(common::topology::rank_t target_rank, int idx = 0) {
    ITYR_CHECK(idx < n_queues_);

    ITYR_CHECK(queue_lock_.is_locked(target_rank, idx));

    std::optional<Entry> ret;

    int b = common::mpi_atomic_faa_value<int>(1, target_rank, queue_state_base_disp(idx), queue_state_win_.win());
    int t = common::mpi_get_value<int>(target_rank, queue_state_top_disp(idx), queue_state_win_.win());

    ityr::common::verbose("steal_nolock b=%d t=%d", b, t);

    if (b < t) {
      ret = common::mpi_get_value<Entry>(target_rank, entries_disp(b, idx), entries_win_.win());
      struct wsq_e {
        void*       frame_base;
        std::size_t frame_size;
      };

      wsq_e *wsq_entry = (wsq_e*)&ret.value();
      ityr::common::verbose("steal_nolock wsqueue entry from rank %d: frame_base=%p, frame_size=%u", target_rank, wsq_entry->frame_base, wsq_entry->frame_size);
    } else {
      common::mpi_atomic_faa_value<int>(-1, target_rank, queue_state_base_disp(idx), queue_state_win_.win());
      ret = std::nullopt;
    }

    return ret;
  }

  std::optional<Entry> steal(common::topology::rank_t target_rank, int idx = 0) {
    ITYR_CHECK(idx < n_queues_);

    queue_lock_.lock(target_rank, idx);
    auto ret = steal_nolock(target_rank, idx);
    queue_lock_.unlock(target_rank, idx);
    return ret;
  }

  void abort_steal(common::topology::rank_t target_rank, int idx = 0) {

    ITYR_CHECK(idx < n_queues_);
    ITYR_CHECK(queue_lock_.is_locked(target_rank, idx));

    common::mpi_atomic_faa_value<int>(-1, target_rank, queue_state_base_disp(idx), queue_state_win_.win());
  }


  template <typename Fn, bool EnsureEmpty = true>
  void for_each_entry_nolock(Fn fn, int idx = 0) {
    ITYR_CHECK(idx < n_queues_);


    if (local_empty_[idx]) {
      return;
    }
    

    queue_state& qs = local_queue_state(idx);
    if constexpr (!EnsureEmpty) {
      if (qs.empty()) {
        return;
      }
    }

    auto entries = local_entries(idx);

    //queue_lock_.priolock(common::topology::my_rank(), idx);

    int t = qs.top.load(std::memory_order_relaxed);
    int b = qs.base.load(std::memory_order_relaxed);
    for (int i = b; i < t; i++) {
      fn(entries[i]);
    }


    if (t <= b) {
      local_empty_[idx] = true;
    }
    

    //queue_lock_.unlock(common::topology::my_rank(), idx);
  }

  template <typename Fn, bool EnsureEmpty = true>
  void for_each_entry(Fn fn, int idx = 0) {
    ITYR_CHECK(idx < n_queues_);

    if (local_empty_[idx]) {
      return;
    }
    

    queue_state& qs = local_queue_state(idx);
    if constexpr (!EnsureEmpty) {
      if (qs.empty()) {
        return;
      }
    }

    auto entries = local_entries(idx);

    queue_lock_.priolock(common::topology::my_rank(), idx);

    int t = qs.top.load(std::memory_order_relaxed);
    int b = qs.base.load(std::memory_order_relaxed);
    for (int i = b; i < t; i++) {
      fn(entries[i]);
    }

    if (t <= b) {
      local_empty_[idx] = true;
    }
    

    queue_lock_.unlock(common::topology::my_rank(), idx);
  }

  int size(int idx = 0) const {
    ITYR_CHECK(idx < n_queues_);
    return local_queue_state(idx).size();
  }

  bool empty(common::topology::rank_t target_rank, int idx = 0) const {
    ITYR_CHECK(idx < n_queues_);

    //common::verbose("wsq empty before mpi_get_value");
    auto remote_qs = common::mpi_get_value<queue_state>(target_rank, queue_state_disp(idx), queue_state_win_.win());
    //common::verbose("wsq empty after mpi_get_value");
    return remote_qs.empty();
  }

  template <typename Fn>
  void for_each_nonempty_queue(common::topology::rank_t target_rank,
                               int idx_begin, int idx_end, bool reverse, Fn fn) {
    {
      common::mpi_get(&local_queue_state_buf(idx_begin), idx_end - idx_begin,
                      target_rank, queue_state_disp(idx_begin), queue_state_win_.win());
    }
    if (reverse) {
      for (int idx = idx_end - 1; idx >= idx_begin; idx--) {
        if (!local_queue_state_buf(idx).empty()) {
          // `fn` should return a boolean (whether to break the loop or not)
          if (fn(idx)) break;
        }
      }
    } else {
      for (int idx = idx_begin; idx < idx_end; idx++) {
        if (!local_queue_state_buf(idx).empty()) {
          if (fn(idx)) break;
        }
      }
    }
  }

  const common::global_lock& lock() const { return queue_lock_; }
  int n_queues() const { return n_queues_; }
  common::mpi_win_manager<Entry>  entrieswin() { return entries_win_; }

private:
  struct queue_state {
    std::atomic<int> top;
    std::atomic<int> base;
    // Check if they are safe to be accessed by MPI RMA
    static_assert(sizeof(std::atomic<int>) == sizeof(int));

    queue_state(int initial_pos = 0) : top(initial_pos), base(initial_pos) {}

    // Copy constructors for std::atomic are deleted
    queue_state(const queue_state& qs)
      : top(qs.top.load(std::memory_order_relaxed)),
        base(qs.base.load(std::memory_order_relaxed)) {}
    queue_state& operator=(const queue_state& qs) {
      top.store(qs.top.load(std::memory_order_relaxed), std::memory_order_relaxed);
      base.store(qs.base.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    int size() const {
      return std::max(0, top.load(std::memory_order_relaxed) -
                         base.load(std::memory_order_relaxed));
    }

    bool empty() const {
      return top.load(std::memory_order_relaxed) <=
             base.load(std::memory_order_relaxed);
    }
  };

  static_assert(std::is_standard_layout_v<queue_state>);
  // FIXME: queue_state is no longer trivially copyable.
  //        Thus, strictly speaking, using MPI RMA for queue_state is illegal.
  // static_assert(std::is_trivially_copyable_v<queue_state>);

  /* static constexpr std::size_t queue_state_align = common::hardware_destructive_interference_size; */
  static constexpr std::size_t queue_state_align = sizeof(queue_state);

  struct alignas(queue_state_align) queue_state_wrapper {
    template <typename... Args>
    queue_state_wrapper(Args&&... args) : value(std::forward<Args>(args)...) {}
    queue_state value;
  };

  std::size_t queue_state_disp(int idx) const {
    return idx * sizeof(queue_state_wrapper) + offsetof(queue_state_wrapper, value);
  }

  std::size_t queue_state_top_disp(int idx) const {
    return idx * sizeof(queue_state_wrapper) + offsetof(queue_state_wrapper, value) + offsetof(queue_state, top);
  }

  std::size_t queue_state_base_disp(int idx) const {
    return idx * sizeof(queue_state_wrapper) + offsetof(queue_state_wrapper, value) + offsetof(queue_state, base);
  }

  std::size_t entries_disp(int entry_num, int idx) const {
    return (entry_num + idx * n_entries_) * sizeof(Entry);
  }

  queue_state& local_queue_state(int idx) const {
    return queue_state_win_.local_buf()[idx].value;
  }

  queue_state& local_queue_state_buf(int idx) const {
    // Memory twice as large as the local queue states is allocated for the MPI window
    return queue_state_win_.local_buf()[n_queues_ + idx].value;
  }

  auto local_entries(int idx) const {
    return entries_win_.local_buf().subspan(idx * n_entries_, n_entries_);
  }

  void move_entries(int offset, int idx) {
    ITYR_CHECK(queue_lock_.is_locked(common::topology::my_rank(), idx));

    queue_state& qs = local_queue_state(idx);
    auto entries = local_entries(idx);

    int t = qs.top.load(std::memory_order_relaxed);
    int b = qs.base.load(std::memory_order_relaxed);

    ITYR_CHECK(b <= t);

    int new_b = b + offset;
    int new_t = t + offset;

    if (offset == 0 || new_b < 0 || n_entries_ < new_t) {
      throw wsqueue_full_exception{};
    }

    std::move(&entries[b], &entries[t], &entries[new_b]);

    qs.top.store(new_t, std::memory_order_relaxed);
    qs.base.store(new_b, std::memory_order_relaxed);
  }

  int                                          n_entries_;
  int                                          n_queues_;
  int                                          initial_pos_;
  common::mpi_win_manager<queue_state_wrapper> queue_state_win_;
  common::mpi_win_manager<Entry>               entries_win_;
  common::global_lock                          queue_lock_;
  std::vector<bool>                            local_empty_;
};

ITYR_TEST_CASE("[ityr::ito::wsqueue] single queue") {
  int n_entries = 1000;
  using entry_t = int;

  common::runtime_options common_opts;
  common::singleton_initializer<common::topology::instance> topo;
  wsqueue<entry_t> wsq(n_entries);

  auto my_rank = common::topology::my_rank();
  auto n_ranks = common::topology::n_ranks();

  ITYR_SUBCASE("local push and pop") {
    int n_trial = 3;
    for (int t = 0; t < n_trial; t++) {
      for (int i = 0; i < n_entries; i++) {
        wsq.push(i);
      }
      for (int i = 0; i < n_entries; i++) {
        auto result = wsq.pop();
        ITYR_CHECK(result.has_value());
        ITYR_CHECK(*result == n_entries - i - 1); // LIFO order
      }
    }
  }

  ITYR_SUBCASE("should throw exception when full") {
    for (int i = 0; i < n_entries; i++) {
      wsq.push(i);
    }
    ITYR_CHECK_THROWS_AS(wsq.push(n_entries), wsqueue_full_exception);
  }

  ITYR_SUBCASE("steal") {
    if (n_ranks == 1) return;

    for (common::topology::rank_t target_rank = 0; target_rank < n_ranks; target_rank++) {
      ITYR_CHECK(wsq.empty(target_rank));

      common::mpi_barrier(common::topology::mpicomm());

      entry_t sum_expected = 0;
      if (target_rank == my_rank) {
        for (int i = 0; i < n_entries; i++) {
          wsq.push(i);
          sum_expected += i;
        }
      }

      common::mpi_barrier(common::topology::mpicomm());

      entry_t local_sum = 0;

      ITYR_SUBCASE("remote steal by only one process") {
        if ((target_rank + 1) % n_ranks == my_rank) {
          for (int i = 0; i < n_entries; i++) {
            auto result = wsq.steal(target_rank);
            ITYR_CHECK(result.has_value());
            ITYR_CHECK(*result == i); // FIFO order
            local_sum += *result;
          }
        }
      }

      ITYR_SUBCASE("remote steal concurrently") {
        if (target_rank != my_rank) {
          while (!wsq.empty(target_rank)) {
            auto result = wsq.steal(target_rank);
            if (result.has_value()) {
              local_sum += *result;
            }
          }
        }
      }

      ITYR_SUBCASE("local pop and remote steal concurrently") {
        if (target_rank == my_rank) {
          while (!wsq.empty(my_rank)) {
            auto result = wsq.pop();
            if (result.has_value()) {
              local_sum += *result;
            }
          }
        } else {
          while (!wsq.empty(target_rank)) {
            auto result = wsq.steal(target_rank);
            if (result.has_value()) {
              local_sum += *result;
            }
          }
        }
      }

      common::mpi_barrier(common::topology::mpicomm());
      entry_t sum_all = common::mpi_reduce_value(local_sum, target_rank, common::topology::mpicomm());

      ITYR_CHECK(wsq.empty(target_rank));

      if (target_rank == my_rank) {
        ITYR_CHECK(sum_all == sum_expected);
      }

      common::mpi_barrier(common::topology::mpicomm());
    }
  }

  ITYR_SUBCASE("all operations concurrently") {
    int n_repeats = 5;

    for (common::topology::rank_t target_rank = 0; target_rank < n_ranks; target_rank++) {
      ITYR_CHECK(wsq.empty(target_rank));

      common::mpi_barrier(common::topology::mpicomm());

      if (target_rank == my_rank) {
        entry_t sum_expected = 0;
        entry_t local_sum = 0;

        // repeat push and pop
        for (int r = 0; r < n_repeats; r++) {
          for (int i = 0; i < n_entries; i++) {
            wsq.push(i);
            sum_expected += i;
          }
          while (!wsq.empty(my_rank)) {
            auto result = wsq.pop();
            if (result.has_value()) {
              local_sum += *result;
            }
          }
        }

        auto req = common::mpi_ibarrier(common::topology::mpicomm());
        common::mpi_wait(req);

        entry_t sum_all = common::mpi_reduce_value(local_sum, target_rank, common::topology::mpicomm());

        ITYR_CHECK(sum_all == sum_expected);

      } else {
        entry_t local_sum = 0;

        auto req = common::mpi_ibarrier(common::topology::mpicomm());
        while (!common::mpi_test(req)) {
          auto result = wsq.steal(target_rank);
          if (result.has_value()) {
            local_sum += *result;
          }
        }

        ITYR_CHECK(wsq.empty(target_rank));

        common::mpi_reduce_value(local_sum, target_rank, common::topology::mpicomm());
      }

      common::mpi_barrier(common::topology::mpicomm());
    }
  }

  ITYR_SUBCASE("resize queue") {
    if (n_ranks == 1) return;

    for (common::topology::rank_t target_rank = 0; target_rank < n_ranks; target_rank++) {
      ITYR_CHECK(wsq.empty(target_rank));

      common::mpi_barrier(common::topology::mpicomm());

      if (target_rank == my_rank) {
        for (int i = 0; i < n_entries; i++) {
          wsq.push(i);
        }
      }

      common::mpi_barrier(common::topology::mpicomm());

      // only one process steals
      if ((target_rank + 1) % n_ranks == my_rank) {
        // steal half of the queue
        for (int i = 0; i < n_entries / 2; i++) {
          auto result = wsq.steal(target_rank);
          ITYR_CHECK(result.has_value());
          ITYR_CHECK(*result == i);
        }
      }

      common::mpi_barrier(common::topology::mpicomm());

      if (target_rank == my_rank) {
        // push half of the queue
        for (int i = 0; i < n_entries / 2; i++) {
          wsq.push(i);
        }
        // pop all
        for (int i = 0; i < n_entries; i++) {
          auto result = wsq.pop();
          ITYR_CHECK(result.has_value());
        }
      }

      common::mpi_barrier(common::topology::mpicomm());

      ITYR_CHECK(wsq.empty(target_rank));
    }
  }
}

ITYR_TEST_CASE("[ityr::ito::wsqueue] multiple queues") {
  int n_entries = 1000;
  int n_queues = 3;
  using entry_t = int;

  common::runtime_options common_opts;
  common::singleton_initializer<common::topology::instance> topo;
  wsqueue<entry_t> wsq(n_entries, n_queues);

  auto my_rank = common::topology::my_rank();
  auto n_ranks = common::topology::n_ranks();

  int n_repeats = 5;

  for (common::topology::rank_t target_rank = 0; target_rank < n_ranks; target_rank++) {
    for (int q = 0; q < n_queues; q++) {
      ITYR_CHECK(wsq.empty(target_rank, q));
    }

    common::mpi_barrier(common::topology::mpicomm());

    if (target_rank == my_rank) {
      entry_t sum_expected = 0;
      entry_t local_sum = 0;

      // repeat push and pop
      for (int r = 0; r < n_repeats; r++) {
        for (int i = 0; i < n_entries; i++) {
          for (int q = 0; q < n_queues; q++) {
            wsq.push(i, q);
            sum_expected += i;
          }
        }
        for (int q = 0; q < n_queues; q++) {
          while (!wsq.empty(my_rank, q)) {
            auto result = wsq.pop(q);
            if (result.has_value()) {
              local_sum += *result;
            }
          }
        }
      }

      auto req = common::mpi_ibarrier(common::topology::mpicomm());
      common::mpi_wait(req);

      entry_t sum_all = common::mpi_reduce_value(local_sum, target_rank, common::topology::mpicomm());

      ITYR_CHECK(sum_all == sum_expected);

    } else {
      entry_t local_sum = 0;

      auto req = common::mpi_ibarrier(common::topology::mpicomm());
      while (!common::mpi_test(req)) {
        for (int q = 0; q < n_queues; q++) {
          auto result = wsq.steal(target_rank, q);
          if (result.has_value()) {
            local_sum += *result;
          }
        }
      }

      for (int q = 0; q < n_queues; q++) {
        ITYR_CHECK(wsq.empty(target_rank, q));
      }

      common::mpi_reduce_value(local_sum, target_rank, common::topology::mpicomm());
    }

    common::mpi_barrier(common::topology::mpicomm());
  }
}

}
