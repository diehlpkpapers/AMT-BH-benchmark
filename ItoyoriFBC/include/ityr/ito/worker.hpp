#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ito/sched/util.hpp"
#include "ityr/ito/sched/randws_header.hpp"
#include "future_header.hpp"

/**
 * @class worker
 * @brief Represents a worker entity responsible for managing task execution and scheduling in the ITYR framework.
 *
 * The worker class is a singleton representing a worker.
 * Its main responsibility is to provide the scheduler and to process root_exec at the worker level.
 * This class is a singleton; each worker is executed as its own program.
 */
namespace ityr::ito::worker {

class worker {
public:
  worker()
    : sched_() {}

  template <typename Fn, typename... Args>
  auto root_exec(Fn&& fn, Args&&... args) {
    ITYR_CHECK(is_spmd_);
    is_spmd_ = false;

    using retval_t = std::invoke_result_t<Fn, Args...>;
    if constexpr (std::is_void_v<retval_t>) {
      if (common::topology::my_rank() == coll_master_) {
        sched_.root_exec<no_retval_t>(std::forward<Fn>(fn), std::forward<Args>(args)...);
      } else {
        sched_.sched_loop();
      }

      is_spmd_ = true;

      common::mpi_barrier(common::topology::mpicomm());

    } else {
      retval_t retval {};
      if (common::topology::my_rank() == coll_master_) {
        retval = sched_.root_exec<retval_t>(std::forward<Fn>(fn), std::forward<Args>(args)...);
      } else {
        sched_.sched_loop();
      }

      is_spmd_ = true;

      common::mpi_barrier(common::topology::mpicomm());

      return common::mpi_bcast_value(retval, coll_master_, common::topology::mpicomm());
    }
  }


  bool is_spmd() const { return is_spmd_; }

  scheduler_randws& sched() { return sched_; }

private:
  scheduler_randws                sched_;
  bool                     is_spmd_ = true;
  common::topology::rank_t coll_master_ = 0;
};

using instance = common::singleton<worker>;

}
