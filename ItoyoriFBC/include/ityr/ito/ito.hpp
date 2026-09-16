#pragma once

#include <functional>

#include "ityr/common/util.hpp"
#include "ityr/common/options.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/ito/util.hpp"
#include "ityr/ito/options.hpp"
#include "ityr/ito/thread.hpp"
#include "ityr/ito/worker.hpp"

/**
 * @brief Main runtime initializer and manager for the ityr::ito system.
 *
 * The ito class encapsulates the initialization and management of all core runtime
 * components required for parallel and distributed execution using the ityr::ito framework.
 * It ensures proper setup of the MPI environment, system topology, wallclock timing,
 * runtime options, and the worker subsystem. The class is designed to be
 * used as a singleton, providing a single entry point for initializing and finalizing
 * the runtime environment.
 *
 * Key responsibilities:
 * - Initialize and finalize the MPI environment.
 * - Set up system topology and wallclock timer instances.
 * - Manage runtime configuration options.
 * - Perform ASLR (Address Space Layout Randomization) checks for stack safety.
 * - Initialize the worker subsystem for task and thread management.
 *
 * Typical usage:
 * - An instance is created as a singleton in the ityr::ito::instance type in ito.hpp.
 * - Call init() at the beginning of the program to initialize the runtime.
 * - Use fini() at the end to clean up resources.
 * - Functions like root_exec(), is_spmd(), and is_root() interact with the runtime and scheduler.
 * - These functions should not be used externally; they have wrappers in the ityr:: namespace with additional responsibilities.
 */

namespace ityr::ito {

class ito {
public:
  ito(MPI_Comm comm)
    : mi_(comm),
      topo_(comm) {}

private:
  common::mpi_initializer                                    mi_;
  common::runtime_options                                    common_opts_;
  common::singleton_initializer<common::topology::instance>  topo_;
  common::singleton_initializer<common::wallclock::instance> clock_;

  runtime_options                                            ito_opts_;
  aslr_checker                                               aslr_checker_;
  common::singleton_initializer<worker::instance>            worker_;
};

using instance = common::singleton<ito>;

inline void init(MPI_Comm comm = MPI_COMM_WORLD) {
  instance::init(comm);
}

inline void fini() {
  instance::fini();
}

template <typename Fn, typename... Args>
inline auto root_exec(Fn&& fn, Args&&... args) {
  auto& w = worker::instance::get();
  ITYR_CHECK(w.is_spmd());
  return w.root_exec(std::forward<Fn>(fn), std::forward<Args>(args)...);
}

inline bool is_spmd() {
  auto& w = worker::instance::get();
  return w.is_spmd();
}

inline bool is_root() {
  auto& w = worker::instance::get();
  return w.sched().is_executing_root();
}

}
