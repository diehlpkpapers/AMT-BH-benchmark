/**
 * @file options.hpp
 * @brief Defines runtime configuration options for the ITYR ITO module.
 *
 * This header provides a set of configurable options for the ITYR ITO runtime,
 * including stack size, work-stealing queue capacity, allocator sizes, checkpointing
 * intervals, and MPI progress behavior.
 * Each option is represented as a struct 
 * inheriting from the common::option template, allowing for type-safe and
 * extensible runtime configuration. The `runtime_options` struct aggregates
 * initializers for all available options, ensuring their registration and
 * initialization at runtime.
 *
 * The file also provides a utility function to print compile-time options,
 * such as the selected scheduler.
 *
 */
#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/options.hpp"



namespace ityr::ito {

inline void print_compile_options() {
#ifndef ITYR_ITO_SCHEDULER
#define ITYR_ITO_SCHEDULER randws
#endif
  ITYR_PRINT_MACRO(ITYR_ITO_SCHEDULER);
}

struct stack_size_option : public common::option<stack_size_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ITO_STACK_SIZE"; }
  static std::size_t default_value() { return std::size_t(2) * 1024 * 1024; }
};

struct wsqueue_capacity_option : public common::option<wsqueue_capacity_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ITO_WSQUEUE_CAPACITY"; }
  static std::size_t default_value() { return 1024; }
};

struct thread_state_allocator_size_option : public common::option<thread_state_allocator_size_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ITO_THREAD_STATE_ALLOCATOR_SIZE"; }
  static std::size_t default_value() { return std::size_t(2) * 1024 * 1024; }
};

struct suspended_thread_allocator_size_option : public common::option<suspended_thread_allocator_size_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ITO_SUSPENDED_THREAD_ALLOCATOR_SIZE"; }
  static std::size_t default_value() { return std::size_t(2) * 1024 * 1024; }
};

struct future_pool_size_option : public common::option<future_pool_size_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ITO_FUTURE_POOL_SIZE"; }
  static std::size_t default_value() { return std::size_t(1) * 1024 * 1024 * 1024; }
};

struct checkpoint_allocator_size_option : public common::option<checkpoint_allocator_size_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ITO_CHECKPOINT_ALLOCATOR_SIZE"; }
  static std::size_t default_value() { return std::size_t(256) * 1024 * 1024; }
};

struct regular_checkpoint_interval_option : public common::option<ityr::ito::regular_checkpoint_interval_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ITO_REGULAR_CHECKPOINT_INTERVAL"; }
  static std::size_t default_value() { return 10000; } // ms
};

struct sched_loop_make_mpi_progress_option : public common::option<sched_loop_make_mpi_progress_option, bool> {
  using option::option;
  static std::string name() { return "ITYR_ITO_SCHED_LOOP_MAKE_MPI_PROGRESS"; }
  static bool default_value() { return true; }
};


struct runtime_options {
  common::option_initializer<stack_size_option>                      ITYR_ANON_VAR;
  common::option_initializer<wsqueue_capacity_option>                ITYR_ANON_VAR;
  common::option_initializer<thread_state_allocator_size_option>     ITYR_ANON_VAR;
  common::option_initializer<suspended_thread_allocator_size_option> ITYR_ANON_VAR;
  common::option_initializer<future_pool_size_option>                 ITYR_ANON_VAR;
  common::option_initializer<checkpoint_allocator_size_option>       ITYR_ANON_VAR;
  common::option_initializer<regular_checkpoint_interval_option>     ITYR_ANON_VAR;
  common::option_initializer<sched_loop_make_mpi_progress_option>    ITYR_ANON_VAR;
};

}
