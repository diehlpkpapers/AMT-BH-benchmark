#pragma once

#define UNUSED(...) (void)(__VA_ARGS__)

#include "ityr/common/util.hpp"
#include "ityr/common/options.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/root_exec.hpp"
#include "ityr/ito/sched/randws.hpp"
#include "ityr/ito/future.hpp"

namespace ityr {

namespace internal {

class ityr {
public:
  ityr(MPI_Comm comm)
    : mi_(comm),
      topo_(comm),
      ito_(comm){}

private:
  common::mpi_initializer                                    mi_;
  common::runtime_options                                    opts_;
  common::singleton_initializer<common::topology::instance>  topo_;
  common::singleton_initializer<common::wallclock::instance> clock_;
  common::singleton_initializer<ito::instance>               ito_;
};

using instance = common::singleton<ityr>;

}

/**
 * @brief Initialize Itoyori (collective).
 *
 * @param comm MPI communicator to be used in Itoyori (default: `MPI_COMM_WORLD`).
 *
 * This function initializes the Itoyori runtime system.
 * Any itoyori APIs (except for runtime option settings) cannot be called prior to this call.
 * `ityr::fini()` must be called to properly release allocated resources.
 *
 * If MPI is not initialized at this point, Itoyori calls `MPI_Init()` and finalizes MPI by calling
 * `MPI_Finalize()` when `ityr::fini()` is called. If MPI is already initialized at this point,
 * Itoyori does not have the responsibility to finalize MPI.
 *
 * @see `ityr::fini()`
 */
inline void init(MPI_Comm comm = MPI_COMM_WORLD) {
  internal::instance::init(comm);
}

/**
 * @brief Finalize Itoyori (collective).
 *
 * This function finalizes the Itoyori runtime system.
 * Any itoyori APIs cannot be called after this call unless `ityr::init()` is called again.
 *
 * `MPI_Finalize()` may be called in this function if MPI is not initialized by the user when
 * `ityr::init()` is called.
 *
 * @see `ityr::init()`
 */
inline void fini() {
  internal::instance::fini();
}

/**
 * @brief Process rank (ID) starting from 0 (corresponding to an MPI rank).
 * @see `ityr::my_rank()`
 * @see `ityr::n_ranks()`
 */
using rank_t = common::topology::rank_t;


/**
 * @brief A future is a container for a value that is asynchronously calculated.
 * 
 * A future is a container for a value that is asynchronously calculated.
 * If the value is needed, use `.get()` to wait for the value.
 * This may result in immediately obtaining the calculated value,
 * or in suspension of the task if the value is not ready yet.
 * 
 */
template <class T>
using future = ito::future<T>;
template <class T>
using promise = ito::future<T>;

/**
 * @brief Launches a function asynchronously and returns a future for its result.
 * 
 * This Library is a work first lib. This means thas the function spawns starts the passed function right away. The rest of the übergeordnete function is saved as a new task/thread called continuation in the task queue of the worker.
 * This is done because in some way the continuation will depend on the new spawned task.
 * 
 */

  template <typename T, typename Fn, typename... Args>
  inline auto spawn(Fn&& fn, Args&&... args) {
    if constexpr (std::is_void_v<T>) {
      // This block compiles only if T is void
      ito::spawn<void>(std::forward<Fn>(fn), std::forward<Args>(args)...);
      // Returns void implicitly
    } else {
      // This block compiles only if T is NOT void
      return ito::spawn<T>(std::forward<Fn>(fn), std::forward<Args>(args)...);
    }
  }

/**
 * @brief Return the rank of the process running the current thread.
 * @see `ityr::n_ranks()`
 */
inline rank_t my_rank() {
  return common::topology::my_rank();
}

/**
 * @brief Return the total number of processes.
 * @see `ityr::n_ranks()`
 */
inline rank_t n_ranks() {
  return common::topology::n_ranks();
}

/**
 * @brief Return true if `ityr::my_rank() == 0`.
 * @see `ityr::my_rank()`
 */
inline bool is_master() {
  return my_rank() == 0;
}

/**
 * @brief Return true if the current thread is the root thread.
 */
inline bool is_root() {
  return ito::is_root();
}

/**
 * @brief Return true if the current execution context is within the SPMD region.
 */
inline bool is_spmd() {
  return ito::is_spmd();
}

/**
 * @brief Barrier for all processes (collective).
 */
inline void barrier() {
  ITYR_CHECK(is_spmd());
  common::mpi_barrier(common::topology::mpicomm());
}

/**
 * @brief Wallclock time in nanoseconds.
 * @see `ityr::gettime_ns()`.
 */
using wallclock_t = common::wallclock::wallclock_t;

/**
 * @brief Return the current wallclock time in nanoseconds.
 *
 * The wallclock time is calibrated across different processes (that may reside in different machines)
 * at the program startup in a simple way, but the clock may be skewed due to various reasons.
 * To get an accurate execution time, it is recommended to call this function in the same process and
 * calculate the difference.
 */
inline wallclock_t gettime_ns() {
  return common::wallclock::gettime_ns();
}

/**
 * @brief Start the profiler (collective).
 * @see `ityr::profiler_end()`.
 * @see `ityr::profiler_flush()`.
 */
inline void profiler_begin() {
  
}

/**
 * @brief Stop the profiler (collective).
 * @see `ityr::profiler_begin()`.
 * @see `ityr::profiler_flush()`.
 */
inline void profiler_end() {
 
}

/**
 * @brief Print the profiled results to stdout (collective).
 * @see `ityr::profiler_begin()`.
 * @see `ityr::profiler_end()`.
 */
inline void profiler_flush() {
}

/**
 * @brief Print the compile-time options to stdout.
 * @see `ityr::print_runtime_options()`.
 */
inline void print_compile_options() {
  common::print_compile_options();
  ito::print_compile_options();
}

/**
 * @brief Print the runtime options to stdout.
 * @see `ityr::print_compile_options()`.
 */
inline void print_runtime_options() {
  common::print_runtime_options();
}

}
