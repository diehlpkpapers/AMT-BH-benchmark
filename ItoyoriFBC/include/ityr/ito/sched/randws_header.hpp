#pragma once

#include "ityr/ito/callstack.hpp"
#include "ityr/ito/ctx/context.hpp"
#include "ityr/ito/wsqueue.hpp"
#include "ityr/common/allocator.hpp"

namespace ityr::ito {
  class future_pool;

  template <class T>
  class future;
}

/**
 * @class scheduler_randws
 * @brief A randomized work-stealing scheduler for managing parallel tasks and their execution contexts.
 *
 * This class implements a scheduler that supports randomized work-stealing for efficient parallel execution.
 * It manages task suspension, resumption, and inter-thread communication.
 * The scheduler provides mechanisms for spawning new tasks (fork), suspending and resuming execution,
 * and stealing tasks between worker threads.
 *
 * Key Features:
 *  - Provides the steal() functionality, which is called in sched_loop for load balancing.
 *  - Executes the sched_loop function as an entry point for workers in a distributed system with no initial task.
 * 
 */
namespace ityr::ito {


  class scheduler_randws {
  public:

    template <typename T>
    struct thread_retval {
      T value;
    };

    struct wsqueue_entry {
      void*       frame_base;
      std::size_t frame_size;
      void*       evacuation_ptr;
      // wait_all support: a non-null waitall_handoff marks a multi-future
      // waiter. Its producer's set() calls waitall_handoff instead of
      // resuming; the handler hands the state off to the next still-pending
      // future's suspended list (one extra push) until the LAST pending future
      // completes, then enqueues the state for the single resume.
      // waitall_entries[0..N] holds the pending futures' entry pointers
      // (0-terminated; the completing future is done by the time the state is
      // popped, so scanning for the first not-done entry is enough).
      // [16]: the variadic wait_all accepts up to 5 futures + a terminating 0;
      // the range wait_all(fs, n) accepts up to 15 + terminator.
      void (*waitall_handoff)(wsqueue_entry& se);
      intptr_t    waitall_entries[16];
    };

    using suspended_state = wsqueue_entry;

    struct suspended_list {
      suspended_state entry;
      suspended_list* next;
    };

    template <class T>
    struct entry {
      T value;
      size_t value_size; // in bytes
      suspended_list* next;
      int gc_counter;
      int done;
    };


    scheduler_randws();
    ~scheduler_randws();

    template <typename T, typename Fn, typename... Args>
    T root_exec(Fn&& fn, Args&&... args);

    template <typename Fn, typename... Args>
    void fork(Fn&& fn, Args&&... args);

    template <typename Fn>
    void suspend(Fn&& fn);

    void sched_loop();


    bool is_executing_root() const;

    future_pool& fpool();
    wsqueue<wsqueue_entry>& taskq();

    void resume(context_frame* cf);

    void resume(suspended_state ss);

    void resume_sched();
    void resume_top();

    void steal();

    suspended_state evacuate(context_frame* cf);
  private:

    void on_task_die();

    template <typename T>
    void on_die(entry<T>* ts, T&& ret);

    template <typename T>
    void on_root_die(entry<T>* ts, T&& ret);


    template <typename Fn>
    void root_on_stack(Fn&& fn);


    bool should_exit_sched_loop();

    template <typename T>
    thread_retval<T> get_retval_remote(entry<T>* ts);

    template <typename T>
    void put_retval_remote(entry<T>* ts, thread_retval<T>&& retval);

    callstack                        stack_;
    context_frame*                   stack_base_;
    oneslot_mailbox<void>            exit_request_mailbox_;
    oneslot_mailbox<suspended_state> migration_mailbox_;
    wsqueue<wsqueue_entry>           wsq_;
    common::remotable_resource       suspended_thread_allocator_;
    context_frame*                   cf_top_           = nullptr;
    context_frame*                   sched_cf_         = nullptr;
    future_pool* fpool_;
  };
}
