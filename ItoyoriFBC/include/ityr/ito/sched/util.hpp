#pragma once

#include <random>
#include <atomic>

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/mpi_rma.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/common/allocator.hpp"
#include "ityr/ito/util.hpp"
#include "ityr/ito/options.hpp"

namespace ityr::ito {


/*
 * Misc
 */
class task_general {
public:
  virtual ~task_general() = default;
  virtual void execute() = 0;
};

template <typename Fn, typename... Args>
class callable_task : public task_general {
public:
  template <typename Fn_, typename... Args_>
  callable_task(Fn_&& fn, Args_&&... args)
    : fn_(std::forward<Fn_>(fn)), arg_(std::forward<Args_>(args)...) {}
  void execute() { std::apply(std::move(fn_), std::move(arg_)); }
private:
  Fn                  fn_;
  std::tuple<Args...> arg_;
};

struct no_retval_t {};

inline common::topology::rank_t get_random_rank(common::topology::rank_t a,
                                                common::topology::rank_t b) {
  static std::mt19937 engine(std::random_device{}());

  ITYR_CHECK(0 <= a);
  ITYR_CHECK(a <= b);
  ITYR_CHECK(b < common::topology::n_ranks());
  std::uniform_int_distribution<common::topology::rank_t> dist(a, b);

  common::topology::rank_t rank;
  do {
    rank = dist(engine);
  } while (rank == common::topology::my_rank());

  ITYR_CHECK(a <= rank);
  ITYR_CHECK(rank != common::topology::my_rank());
  ITYR_CHECK(rank <= b);
  return rank;
}

template <typename T, typename Fn, typename ArgsTuple>
inline decltype(auto) invoke_fn(Fn&& fn, ArgsTuple&& args_tuple) {
  if constexpr (!std::is_same_v<T, no_retval_t>) {
    return std::apply(std::forward<Fn>(fn), std::forward<ArgsTuple>(args_tuple));
  } else {
    std::apply(std::forward<Fn>(fn), std::forward<ArgsTuple>(args_tuple));
    return no_retval_t{};
  }
}

template <typename Fn, typename... Args>
struct callback_retval {
  using type = std::invoke_result_t<Fn, Args...>;
};

template <typename... Args>
struct callback_retval<std::nullptr_t, Args...> {
  using type = void;
};

template <typename... Args>
struct callback_retval<std::nullptr_t&, Args...> {
  using type = void;
};

template <typename Fn, typename... Args>
using callback_retval_t = typename callback_retval<Fn, Args...>::type;

/*
 * Mailbox
 */

template <typename Entry>
class oneslot_mailbox {
  static_assert(std::is_trivially_copyable_v<Entry>);

public:
  oneslot_mailbox()
    : win_(common::topology::mpicomm(), 1) {}

  void put(const Entry& entry, common::topology::rank_t target_rank) {
    ITYR_CHECK(!common::mpi_get_value<int>(target_rank, offsetof(mailbox, arrived), win_.win()));
    common::mpi_put_value(entry, target_rank, offsetof(mailbox, entry), win_.win());
    common::mpi_atomic_put_value(1, target_rank, offsetof(mailbox, arrived), win_.win());
  }

  std::optional<Entry> pop() {
    mailbox& mb = win_.local_buf()[0];
    if (mb.arrived.load(std::memory_order_acquire)) {
      mb.arrived.store(0, std::memory_order_relaxed);
      return mb.entry;
    } else {
      return std::nullopt;
    }
  }

  bool arrived() const {
    return win_.local_buf()[0].arrived.load(std::memory_order_relaxed);
  }

private:
  struct mailbox {
    Entry            entry;
    std::atomic<int> arrived = 0; // TODO: better to use std::atomic_ref in C++20
  };

  common::mpi_win_manager<mailbox> win_;
};

template <>
class oneslot_mailbox<void> {
public:
  oneslot_mailbox()
    : win_(common::topology::mpicomm(), 1) {}

  void put(common::topology::rank_t target_rank) {
    ITYR_CHECK(!common::mpi_get_value<int>(target_rank, offsetof(mailbox, arrived), win_.win()));
    common::mpi_atomic_put_value(1, target_rank, offsetof(mailbox, arrived), win_.win());
  }

  bool pop() {
    mailbox& mb = win_.local_buf()[0];
    if (mb.arrived.load(std::memory_order_acquire)) {
      mb.arrived.store(0, std::memory_order_relaxed);
      return true;
    } else {
      return false;
    }
  }

  bool arrived() const {
    return win_.local_buf()[0].arrived.load(std::memory_order_relaxed);
  }

private:
  struct mailbox {
    std::atomic<int> arrived = 0; // TODO: better to use std::atomic_ref in C++20
  };

  common::mpi_win_manager<mailbox> win_;
};

}
