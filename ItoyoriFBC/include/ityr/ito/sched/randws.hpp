#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/mpi_rma.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/logger.hpp"
#include "ityr/common/allocator.hpp"
#include "ityr/ito/util.hpp"
#include "ityr/ito/options.hpp"
#include "ityr/ito/ctx/context.hpp"
#include "ityr/ito/callstack.hpp"
#include "ityr/ito/wsqueue.hpp"
#include "ityr/ito/sched/util.hpp"
#include "ityr/ito/sched/randws_header.hpp"
#include "ityr/ito/future_header.hpp"

namespace ityr::ito {
  scheduler_randws::scheduler_randws()
    : stack_(stack_size_option::value()),
      // Add a margin of sizeof(context_frame) to the bottom of the stack, because
      // this region can be accessed by the clear_parent_frame() function later.
      // This stack base is updated only in coll_exec().
      stack_base_(reinterpret_cast<context_frame*>(stack_.bottom()) - 1),
      wsq_(wsqueue_capacity_option::value()),
      //thread_state_allocator_(thread_state_allocator_size_option::value()),
      suspended_thread_allocator_(suspended_thread_allocator_size_option::value())
  {
    fpool_ = new future_pool(future_pool_size_option::value());
    fpool_->initialize();
    common::mpi_barrier(common::topology::mpicomm());
  }

  scheduler_randws::~scheduler_randws() {
    delete fpool_;
  }

  template <typename T, typename Fn, typename... Args>
  T scheduler_randws::root_exec(Fn&& fn, Args&&... args) {

    auto fut = future<T>::make();
    entry<T>* e = fut.e;

    auto prev_sched_cf = sched_cf_;

    common::verbose("IN RAND WS");

    suspend([&](context_frame* cf) {
      sched_cf_ = cf;
      root_on_stack([&, e, fn = std::forward<Fn>(fn),
                        args_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
        common::verbose("Starting root thread %p", e);

        T&& ret = invoke_fn<T>(std::forward<decltype(fn)>(fn), std::forward<decltype(args_tuple)>(args_tuple));
        if constexpr (!std::is_same_v<T, no_retval_t>) {
          put_retval_remote(e, {std::move(ret)});
        }

        common::verbose("Root thread %p is completed", e);

        exit_request_mailbox_.put(0);

        resume_sched();
      });
    });

    sched_loop();

    sched_cf_ = prev_sched_cf;

    return e->value;
  }

  template <typename Fn, typename... Args>
  void scheduler_randws::fork(Fn&& fn, Args&&... args) {

    suspend([&, fn = std::forward<Fn>(fn),
             args_tuple = std::make_tuple(std::forward<Args>(args)...)](context_frame* cf) mutable {
      std::size_t cf_size = reinterpret_cast<uintptr_t>(cf->parent_frame) - reinterpret_cast<uintptr_t>(cf);
      common::verbose<2>("push context frame [%p, %p) into task queue %u rsp=%p, %rbp=%p, rip=%p", cf, cf->parent_frame, cf_size, cf->rsp, cf->rbp, cf->rip);

      wsq_.push(wsqueue_entry{cf, cf_size, nullptr, nullptr, {0}});

      common::verbose<2>("Starting new thread");

      invoke_fn<void>(std::forward<decltype(fn)>(fn), std::forward<decltype(args_tuple)>(args_tuple));

      common::verbose<2>("Thread %p is completed");

      common::verbose<2>("Resume parent context frame [%p, %p) (fast path)", cf, cf->parent_frame);
    });
  }

  void scheduler_randws::sched_loop() {
    common::verbose("Enter scheduling loop p=%d", common::topology::my_rank());

    while (!should_exit_sched_loop()) {
      steal();
    }

    common::verbose("Exit scheduling loop");
  }


  bool scheduler_randws::is_executing_root() const {
    return cf_top_ && cf_top_ == stack_base_;
  }


  future_pool& scheduler_randws::fpool() { return *fpool_; }
  wsqueue<scheduler_randws::wsqueue_entry>& scheduler_randws::taskq() { return wsq_; }



  template <typename T>
  void scheduler_randws::on_root_die(entry<T>* ts, T&& ret) {
    common::verbose("on_root_die %p", ts);
    if constexpr (!std::is_same_v<T, no_retval_t>) {
      put_retval_remote(ts, {std::move(ret)});
    }

    exit_request_mailbox_.put(0);

    resume_sched();
  }

  void scheduler_randws::steal() {
    auto target_rank = get_random_rank(0, common::topology::n_ranks() - 1);

    if (wsq_.empty(target_rank)) {
      return;
    }

    // 1.
    if (!wsq_.lock().trylock(target_rank)) {
      return;
    }
    
    auto we = wsq_.steal_nolock(target_rank);
    if(!we.has_value()) {
      wsq_.lock().unlock(target_rank);
      return;
    }

    if (we->evacuation_ptr != nullptr) {
      wsq_.lock().unlock(target_rank);
      suspend([&](context_frame *cf) {
        sched_cf_ = cf;
        //context::clear_parent_frame(next_cf);
        resume(suspended_state(*we));
      });
    } else {
      // Release the queue lock BEFORE the frame copy. The owner's pop/priolock
      // spins while we hold this lock; the frame GET below needs the owner's
      // MPI progress, so holding the lock across it can deadlock (frame GET
      // starved vs. owner spinning on the lock).
      wsq_.lock().unlock(target_rank);

      stack_.direct_copy_from(we->frame_base, we->frame_size, target_rank);

      common::verbose("Steal context frame [%p, %p) from rank %d",
                      we->frame_base, reinterpret_cast<std::byte *>(we->frame_base) + we->frame_size, target_rank);

      context_frame *next_cf = reinterpret_cast<context_frame *>(we->frame_base);
      suspend([&](context_frame *cf) {
        sched_cf_ = cf;
        //context::clear_parent_frame(next_cf);
        resume(next_cf);
      });
    }
  }

  template <typename Fn>
  void scheduler_randws::suspend(Fn&& fn) {
    context_frame*        prev_cf_top = cf_top_;

    context::save_context_with_call(prev_cf_top,
        [](context_frame* cf, void* cf_top_p, void* fn_p) {
      context_frame*& cf_top = *reinterpret_cast<context_frame**>(cf_top_p);
      Fn              fn     = std::forward<Fn>(*reinterpret_cast<Fn*>(fn_p)); // copy closure to the new stack frame
      cf_top = cf;
      fn(cf);
    }, &cf_top_, &fn, nullptr);

    cf_top_ = prev_cf_top;

  }

  void scheduler_randws::resume(context_frame* cf) {
    common::verbose("Resume context frame [%p, %p) in the stack", cf, cf->parent_frame);
    context::resume(cf);
  }

  void scheduler_randws::resume(suspended_state ss) {
    common::verbose("Resume context frame [%p, %p) evacuated at %p",
                    ss.frame_base, reinterpret_cast<uintptr_t>(ss.frame_base) + ss.frame_size, ss.evacuation_ptr);

    // We pass the suspended thread states *by value* because the current local variables can be overwritten by the
    // new stack we will bring from remote nodes.
    context::jump_to_stack(ss.frame_base, [](void* allocator_, void* evacuation_ptr, void* frame_base, void* frame_size_) {
      common::remotable_resource& allocator  = *reinterpret_cast<common::remotable_resource*>(allocator_);
      std::size_t                 frame_size = reinterpret_cast<std::size_t>(frame_size_);
      common::remote_get(allocator,
                         reinterpret_cast<std::byte*>(frame_base),
                         reinterpret_cast<std::byte*>(evacuation_ptr),
                         frame_size);
      allocator.deallocate(evacuation_ptr, frame_size);

      context_frame* cf = reinterpret_cast<context_frame*>(frame_base);
      //context::clear_parent_frame(cf);
      context::resume(cf);
    }, &suspended_thread_allocator_, ss.evacuation_ptr, ss.frame_base, reinterpret_cast<void*>(ss.frame_size));
  }

  void scheduler_randws::resume_sched() {
    common::verbose("Resume scheduler context");
    context::resume(sched_cf_);
  }

  void scheduler_randws::resume_top() {
    common::verbose("Resume top context");
    context::resume(cf_top_);
  }

  scheduler_randws::suspended_state scheduler_randws::evacuate(context_frame* cf) {
    std::size_t cf_size = reinterpret_cast<uintptr_t>(cf->parent_frame) - reinterpret_cast<uintptr_t>(cf);
    void* evacuation_ptr = suspended_thread_allocator_.allocate(cf_size);
    std::memcpy(evacuation_ptr, cf, cf_size);

    common::verbose("Evacuate suspended thread context [%p, %p) to %p with rbp=%p, rsp=%p, rip=%p",
                    cf, cf->parent_frame, evacuation_ptr, cf->rbp, cf->rsp, cf->rip);

    return {cf, cf_size, evacuation_ptr, nullptr, {0, 0, 0, 0}};
  }

  template <typename Fn>
  void scheduler_randws::root_on_stack(Fn&& fn) {
    cf_top_ = stack_base_;
    std::size_t stack_size_bytes = reinterpret_cast<std::byte*>(stack_base_) -
                                   reinterpret_cast<std::byte*>(stack_.top());
    context::call_on_stack(stack_.top(), stack_size_bytes,
                           [](void* fn_, void*, void*, void*) {
      Fn fn = std::forward<Fn>(*reinterpret_cast<Fn*>(fn_)); // copy closure to the new stack frame
      fn();
    }, &fn, nullptr, nullptr, nullptr);
  }


  bool scheduler_randws::should_exit_sched_loop() {
    if (sched_loop_make_mpi_progress_option::value()) {
      common::mpi_make_progress();
    }


    //Todo: make sure failed workers dont mess with termination
    if (exit_request_mailbox_.pop()) {
      auto my_rank = common::topology::my_rank();
      auto n_ranks = common::topology::n_ranks();
      for (common::topology::rank_t i = common::next_pow2(n_ranks); i > 1; i /= 2) {
        if (my_rank % i == 0) {
          auto target_rank = my_rank + i / 2;
          if (target_rank < n_ranks) {
            exit_request_mailbox_.put(target_rank);
          }
        }
      }
      return true;
    }

 //   common::verbose("should_exit_sched_loop false");
    return false;
  }

  template <typename T>
  scheduler_randws::thread_retval<T> scheduler_randws::get_retval_remote(entry<T>* e) {
    if constexpr (std::is_trivially_copyable_v<T>) {
      return remote_get_value(fpool_->remote_bufs_allocator_, &e->value);
    } else {
      // TODO: Fix this ugly hack of avoiding object destruction by using checkout/checkin
      thread_retval<T> retval;
      remote_get(fpool_->remote_bufs_allocator_, reinterpret_cast<std::byte*>(&retval), reinterpret_cast<std::byte*>(&e->value), sizeof(thread_retval<T>));
      return retval;
    }
  }

  template <typename T>
  void scheduler_randws::put_retval_remote(entry<T>* ts, thread_retval<T>&& retval) {
    if constexpr (std::is_trivially_copyable_v<T>) {
      remote_put_value(fpool_->remote_bufs_allocator_, retval.value, &ts->value);
    } else {
      common::verbose("ERROR PUT RETVAL REMOTE BAD PATH");
      // TODO: Fix this ugly hack of avoiding object destruction by using checkout/checkin
      std::byte* retvalp = reinterpret_cast<std::byte*>(new (alloca(sizeof(thread_retval<T>))) thread_retval<T>{std::move(retval)});
      remote_put(fpool_->remote_bufs_allocator_, retvalp, reinterpret_cast<std::byte*>(&ts->value), sizeof(thread_retval<T>));
    }
  }

}
