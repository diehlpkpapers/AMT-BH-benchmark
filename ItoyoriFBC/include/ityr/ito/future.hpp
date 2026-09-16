#pragma once


#include <tuple>
#include "ityr/common/util.hpp"
#include "ityr/ito/future_header.hpp"
#include "ityr/ito/worker.hpp"
#include "ityr/ito/futurepool.hpp"

namespace ityr::ito {

    constexpr ityr::common::topology::rank_t PID_INVALID = -1;

    // Forward decl (defined at the end of this header): push a waiter state
    // onto an arbitrary future's suspended list (shared by
    // future<T>::push_suspended_state and wait_all's handoff).
    template <class T>
    bool push_suspended_state(ityr::ito::scheduler_randws::entry<T>* e,
                              ityr::ito::scheduler_randws::suspended_state ss);


    template <class T>
    inline future<T>::future() : e(nullptr)
    {
    }

    template <class T>
    inline future<T>::future(ityr::ito::scheduler_randws::entry<T>* e_ptr) : e(e_ptr)
    {
      increment_gc_counter(1);
    }

    template <class T>
    inline future<T>::~future()
    {
      increment_gc_counter(-1);
    }

    template <class T>
    future<T>& future<T>::operator=(future<T>&& other) {
      if (this->e == other.e) return *this;
      increment_gc_counter(-1);
      e = other.e;
      other.e = nullptr;
      return *this;
    }

    template <class T>
    future<T>& future<T>::operator=(const future<T>& other) {
      if (this->e == other.e) return *this;
      increment_gc_counter(-1);
      e = other.e;
      increment_gc_counter(1);
      return *this;
    }

    template <class T>
    inline future<T>::future(const future<T>& other) {
      e = other.e;
      increment_gc_counter(1);
    }

    template <class T>
    inline future<T>::future(future<T>&& other) {
      e = other.e;
      other.e = nullptr;
    }

    template <class T>
    inline future<T> future<T>::make()
    {
        auto& w = ityr::ito::worker::instance::get();
        return make(w);
    }

    template <class T>
    inline future<T> future<T>::make(ityr::ito::worker::worker& w)
    {
        auto ret = w.sched().fpool().get<T>();
        return ret;
    }

    template<class T>
    inline future<T> future<T>::make_filled(T result, size_t size, bool no_copy)
    {
      auto& w = ityr::ito::worker::instance::get();
      future<T> future = make(w);
      // Publish value/value_size through the RMA window (like the normal
      // completion path), in order, BEFORE the done flag: the consumer's
      // sync() reads these fields with remote_get_value, and plain stores
      // could be observed as stale (e.g. a reused entry's previous value_size)
      // while done already reads ready.
      if (no_copy) {
        size = size | (1UL << 63UL);
      }
      w.sched().fpool().fill(future, result, size);
      future.e->gc_counter = 2;
      return future;
    }

    template<class T>
    inline future<T> future<T>::make_filled_pair(T pair, bool no_copy)
    {
      auto& w = ityr::ito::worker::instance::get();
      future<T> future = make(w);
      auto size = pair.second;
      if (no_copy) {
        size = size | (1UL << 63UL);
      }
      w.sched().fpool().fill(future, pair, size);
      future.e->gc_counter = 2;
      return future;
    }

    template <class T>
    inline void future<T>::set(T& value, size_t value_size) {
        auto& w = ityr::ito::worker::instance::get();

        w.sched().fpool().fill(*this, value, value_size);
        std::vector<ityr::ito::scheduler::suspended_state> ses = pop_suspended_states();

        // some waiters are found
        // push the waiter tasks to the local task queue
        // we need to put the waiter tasks to the bottom of the queue so they do not corrupt the stack of non-evacuated tasks
        for (auto ss : ses) {
          if (ss.waitall_handoff != nullptr) {
            // wait_all waiter: do NOT resume yet — hand the state off to the
            // next still-pending future (or enqueue for the single final
            // resume once all of them are done).
            ss.waitall_handoff(ss);
          } else {
            w.sched().taskq().push_bottom(ss);
          }
        }
    }

    #ifdef ITYR_COPY_FUTURE_DATA
    static void* alloc_future_data(size_t numBytes) {
      auto& w = ityr::ito::worker::instance::get();
      return w.sched().fpool().remote_bufs_allocator_.allocate(numBytes);
    }

    static void dealloc_future_data(void *p, size_t numBytes) {
      auto& w = ityr::ito::worker::instance::get();
      auto& fpool = w.sched().fpool();
      // If p is a cached get() copy, the cache owns it:
      // free it and drop the cache entry. Otherwise p is a producer buffer; in
      // addition to freeing it, drop any cached copies that were made FROM
      // data within this buffer (incl. sub-range futures) — its consumers
      // have all joined by the time the app frees it.
      if (!fpool.release_cached_copy(p)) {
        fpool.purge_copies_of_original(p, numBytes);
        fpool.remote_bufs_allocator_.deallocate(p, numBytes);
      }
    }

    // Hand a cached get() copy over to the app without freeing it: the app
    // mutated/adopted this buffer and it outlives the future (e.g. the forces
    // bodies become the next step's live buffer). Safe only when the buffer is
    // not re-get()'d afterwards.
    [[maybe_unused]] static void adopt_future_data(void *p, size_t numBytes) {
      UNUSED(numBytes);
      auto& w = ityr::ito::worker::instance::get();
      w.sched().fpool().adopt_cached_copy(p);
    }

    // Drop every cached get() copy on this rank (freeing the buffers). Call at
    // a phase boundary, after all consumers of the phase's futures have joined
    // — this is what reclaims the per-rank copies that were made FROM remote
    // buffers on OTHER ranks (their originals are freed elsewhere, so the
    // purge-on-original path cannot see them). Adopted copies are already out
    // of the cache and survive.
    static void flush_copy_cache() {
      auto& w = ityr::ito::worker::instance::get();
      w.sched().fpool().clear_copy_cache();
    }
    #endif

    template <class T>
    void future<T>::increment_gc_counter(int value) {
      if (e != nullptr) {
        auto& w = ityr::ito::worker::instance::get();
        int old_value = remote_faa_value(w.sched().fpool().remote_bufs_allocator_, value, &e->gc_counter);
        int new_value = old_value + value;
        if (new_value == 0) {
          // Do NOT recycle the entry here. The F&O read-back that produced
          // new_value is not guaranteed to reflect the entry's true refcount
          // under concurrent F&Os. Returning the entry to the pool while it is
          // still referenced lets the chunk be reallocated (e.g. as a
          // suspended-list node), which corrupts the still-live future.
          // Entries are 32 B; leaking them is bounded and far cheaper than the
          // corruption.
        }
      }
    }

    template<class T>
    std::vector<ityr::ito::scheduler_randws::suspended_state> future<T>::pop_suspended_states() {
      auto& w = ityr::ito::worker::instance::get();
      std::vector<ityr::ito::scheduler_randws::suspended_state> states;
      ityr::ito::scheduler_randws::suspended_list* current_entry_ptr = remote_get_value(w.sched().fpool().remote_bufs_allocator_, &e->next);
      // CAS to mark e->next as already popped
      intptr_t expected = (intptr_t) current_entry_ptr;
      while (((intptr_t)(current_entry_ptr = (ityr::ito::scheduler_randws::suspended_list*) remote_cas_value(w.sched().fpool().remote_bufs_allocator_, (intptr_t) -1, expected, (intptr_t*)&e->next))) != expected) {
        expected = (intptr_t) current_entry_ptr;
      } // another thread pushed something
      while (current_entry_ptr) {
        ityr::ito::scheduler_randws::suspended_list current_entry = remote_get_value(w.sched().fpool().remote_bufs_allocator_, current_entry_ptr);
        states.push_back(current_entry.entry);
        w.sched().fpool().remote_bufs_allocator_.deallocate(current_entry_ptr, sizeof(ityr::ito::scheduler_randws::suspended_list));
        current_entry_ptr = current_entry.next;
      }
      return states;
    }

    template<class T>
    bool future<T>::push_suspended_state(ityr::ito::scheduler_randws::suspended_state ss) {
      return ityr::ito::push_suspended_state(this->e, ss);
    }

    void continue_computation() {
      ityr::common::verbose("continue computation\n");
      auto& w = ityr::ito::worker::instance::get();

      auto entry = w.sched().taskq().pop();

      if (entry.has_value()) {
        ityr::common::verbose("continue computation, queue non-empty\n");
        if (entry->evacuation_ptr == nullptr) {
          // a parent task is popped
          ityr::ito::context_frame* ctx = (ityr::ito::context_frame*)entry->frame_base;
          w.sched().resume(ctx);
        } else {
          // an evacuated task is popped
          w.sched().resume(ityr::ito::scheduler_randws::suspended_state(*entry));
        }
      } else {
        ityr::common::verbose("continue computation, queue empty\n");
        // move to the scheduler
        w.sched().resume_sched();
      }
    }

    template <class T>
    static void future_join_suspended(ityr::ito::scheduler_randws::suspended_state se, future<T>& f)
    {
        auto& w = ityr::ito::worker::instance::get();

        if (w.sched().fpool().sync_suspended(f, se)) {
          // return to the suspended thread again
          w.sched().resume(se);
        } else {
          continue_computation();
        }
    }

    template <class T>
    inline bool future<T>::is_ready() {
      auto& w = ityr::ito::worker::instance::get();
      return remote_get_value(w.sched().fpool().remote_bufs_allocator_, &e->done) != 0;
    }

    template <class T>
    inline void future<T>::wait() {
      if (is_ready()) return;

      auto& w = ityr::ito::worker::instance::get();
      w.sched().suspend([&w, this] (ityr::ito::context_frame* cf) {
        future_join_suspended(w.sched().evacuate(cf), *this);
      });
    }

    template <class T>
    inline T* future<T>::get_pointer() {
      ityr::common::verbose("future::get_pointer e=%p\n", this->e);
      wait();
      auto *e = this->e;
      auto& w = ityr::ito::worker::instance::get();
      return (T*)remote_get_value(w.sched().fpool().remote_bufs_allocator_, &e->value);
    }

    template <class T>
    inline std::size_t future<T>::size() {
      auto& w = ityr::ito::worker::instance::get();
      auto& alloc = w.sched().fpool().remote_bufs_allocator_;
      return remote_get_value(alloc, &e->value_size) & ~(1ull << 63);
    }

    template <class T>
    inline T future<T>::get(void *target_pointer)
    {
        ityr::common::verbose("future::get e=%p\n", this->e);
        auto *e = this->e;
        auto& w = ityr::ito::worker::instance::get();

        T value;
        if (!w.sched().fpool().sync(*this, &value, target_pointer)) {
            w.sched().suspend([&w, this] (ityr::ito::context_frame* cf) {
                future_join_suspended(w.sched().evacuate(cf), *this);
            });


            // worker can change after suspend
            auto& w1 = ityr::ito::worker::instance::get();

            w1.sched().fpool().sync_resume(e, &value, target_pointer);
        }

        ityr::common::verbose("future::get return val=%p\n", value);
        return value;
    }

    template <class T>
    inline void future<T>::discard()
    {
        auto& w = ityr::ito::worker::instance::get();
        w.sched().fpool().return_future_id(e);
        e = nullptr;
    }

    inline size_t index_of_size(size_t size)
    {
        return 64UL - static_cast<size_t>(__builtin_clzl(size - 1));
    }

    inline future_pool::future_pool(size_t buf_size) :
        remote_bufs_allocator_(buf_size)
    {
    }

    inline future_pool::~future_pool()
    {
    }

    inline void future_pool::initialize()
    {
    }

    inline void future_pool::finalize()
    {
      // Free any cached get() copies that the app never released/purged.
      clear_copy_cache();
    }

    template <class T>
    inline future<T> future_pool::get()
    {
        void *memory = remote_bufs_allocator_.allocate(sizeof(ityr::ito::scheduler_randws::entry<T>));
        memset(memory, 0, sizeof(ityr::ito::scheduler_randws::entry<T>));
        future<T> ret = future<T>((ityr::ito::scheduler_randws::entry<T>*) memory);

        ityr::common::verbose("%d allocated fut %p\n", ityr::common::topology::my_rank(), ret.e);
        return ret;
    }

    // ================= wait_all (single suspension for several futures) =================
    // Semantics: like calling wait() on each future, but the task
    // suspends/resumes EXACTLY ONCE. The evacuated state is attached to the
    // first not-ready future's suspended list; when that future completes, its
    // producer's set() calls waitall_handoff instead of resuming, which hands
    // the state off to the next still-pending future (one extra push). Only the
    // LAST completion enqueues the state → the single resume.
    //
    // Correctness: the state is popped only by the producer of the future it is
    // attached to, and that producer published done before popping — so a scan
    // for the first not-done entry automatically skips the completing future.
    // Push races (producer CAS'd e->next to -1 concurrently) make
    // push_suspended_state return false and the scan simply continues; if
    // nothing can be pushed, the task resumes and re-checks readiness (same
    // fallback as wait()).

    template <class T>
    inline bool push_suspended_state(ityr::ito::scheduler_randws::entry<T>* e,
                                     ityr::ito::scheduler_randws::suspended_state ss) {
      auto& w = ityr::ito::worker::instance::get();
      auto& allocator = w.sched().fpool().remote_bufs_allocator_;
      auto next_ptr = (ityr::ito::scheduler_randws::suspended_list*)remote_get_value(allocator, &e->next);
      if ((intptr_t)next_ptr == -1) {
        return false;
      }
      uintptr_t next = reinterpret_cast<uintptr_t>(next_ptr);
      auto new_entry_ptr = (ityr::ito::scheduler_randws::suspended_list*)allocator.allocate(sizeof(ityr::ito::scheduler_randws::suspended_list));
      new_entry_ptr->next = next_ptr;
      new_entry_ptr->entry = ss;
      // Publish the node with plain stores + cache_flush + win_sync instead of
      // one atomic op per field. This is the same publication mechanism fill()
      // uses for data buffers — zero MPI ops per suspend. The publication is
      // done BEFORE the linking CAS, so a reader that observes the node via
      // the CAS sees the published contents.
      ityr::common::cache_flush(new_entry_ptr, sizeof(ityr::ito::scheduler_randws::suspended_list));
      ityr::common::mpi_win_sync(allocator.win());
      uintptr_t new_entry = reinterpret_cast<uintptr_t>(new_entry_ptr);
      while ((next = remote_cas_value(allocator, new_entry, next, (uintptr_t*)&e->next)) != (uintptr_t)new_entry_ptr->next) {
        if ((intptr_t)next == -1) {
          allocator.deallocate(new_entry_ptr, sizeof(ityr::ito::scheduler_randws::suspended_list));
          return false;
        }
        new_entry_ptr->next = (ityr::ito::scheduler_randws::suspended_list*)next;
        ityr::common::cache_flush(new_entry_ptr, sizeof(ityr::ito::scheduler_randws::suspended_list));
        ityr::common::mpi_win_sync(allocator.win());
      };
      return true;
    }

    template <class F>
    struct future_value;
    template <class T>
    struct future_value<ityr::ito::future<T>> { using type = T; };

    // Check one pending future: done → keep scanning (false); not done → push
    // the state onto its list (true). A raced push (false) also continues the
    // scan — the future completed concurrently.
    template <class T>
    inline bool waitall_check_and_push(ityr::ito::scheduler_randws::entry<T>* e,
                                       ityr::ito::scheduler_randws::suspended_state& se) {
      auto& w = ityr::ito::worker::instance::get();
      auto& alloc = w.sched().fpool().remote_bufs_allocator_;
      if (remote_get_value(alloc, &e->done) != 0) return false;
      return push_suspended_state(e, se);
    }

    template <std::size_t I, class T0, class... Rest>
    inline bool waitall_scan(ityr::ito::scheduler_randws::suspended_state& se) {
      if (waitall_check_and_push((ityr::ito::scheduler_randws::entry<T0>*)se.waitall_entries[I], se))
        return true;
      return waitall_scan<I + 1, Rest...>(se);
    }
    template <std::size_t I>
    inline bool waitall_scan(ityr::ito::scheduler_randws::suspended_state& se) {
      (void)se;
      return false;
    }

    // Producer-side handler, called from set() for a wait_all waiter.
    template <class... Ts>
    inline void waitall_handoff_impl(ityr::ito::scheduler_randws::suspended_state& se) {
      if (!waitall_scan<0, Ts...>(se)) {
        // every pending future is done → single final resume
        auto& w = ityr::ito::worker::instance::get();
        w.sched().taskq().push_bottom(se);
      }
    }

    // Attach the state to the first not-ready future's list. Returns true if
    // the task stayed suspended; false means resume now (all done, or a raced
    // push — the resumed task re-checks readiness and re-suspends if needed).
    template <class F1, class... Fs>
    inline bool future_join_suspended_multi(ityr::ito::scheduler_randws::suspended_state& se,
                                            F1& f1, Fs&... fs) {
      auto& w = ityr::ito::worker::instance::get();
      auto& alloc = w.sched().fpool().remote_bufs_allocator_;
      bool pushed = false;
      if (remote_get_value(alloc, &f1.e->done) == 0) pushed = push_suspended_state(f1.e, se);
      (void)std::initializer_list<int>{([&] {
        if (!pushed && remote_get_value(alloc, &fs.e->done) == 0)
          pushed = push_suspended_state(fs.e, se);
      }(), 0)...};
      return pushed;
    }

    // Wait for all given futures with a single suspension (up to 5 futures).
    template <class F1, class... Fs>
    inline void wait_all(F1& f1, Fs&... fs) {
      static_assert(1 + sizeof...(Fs) <= 5, "wait_all supports at most 5 futures");
      if (f1.is_ready() && (fs.is_ready() && ...)) return;

      auto& w = ityr::ito::worker::instance::get();
      w.sched().suspend([&](ityr::ito::context_frame* cf) {
        auto se = w.sched().evacuate(cf);
        se.waitall_handoff =
            &waitall_handoff_impl<typename future_value<F1>::type,
                                  typename future_value<Fs>::type...>;
        std::size_t i = 0;
        se.waitall_entries[i++] = reinterpret_cast<intptr_t>(f1.e);
        ((se.waitall_entries[i++] = reinterpret_cast<intptr_t>(fs.e)), ...);
        se.waitall_entries[i] = 0;

        if (!future_join_suspended_multi(se, f1, fs...)) {
          w.sched().resume(se);
        } else {
          continue_computation();
        }
      });
    }

    // Range-based wait_all for same-typed futures in an array (e.g. the
    // children of a recursive divide-and-conquer): single suspension for the
    // whole group, up to 15 futures (waitall_entries[16]).
    //
    // Same correctness story as the variadic form: the evacuated state is
    // attached to the first not-ready future; its producer's set() hands the
    // state to the next still-pending future until the LAST completion does
    // the single resume. The producer is same-typed (array), so the handoff
    // scan uses one type instead of the variadic pack.
    template <class T>
    inline void waitall_handoff_impl_range(ityr::ito::scheduler_randws::suspended_state& se) {
      auto& w = ityr::ito::worker::instance::get();
      for (int i = 0; se.waitall_entries[i] != 0; ++i) {
        if (waitall_check_and_push((ityr::ito::scheduler_randws::entry<T>*)se.waitall_entries[i], se))
          return;
      }
      // every pending future is done → single final resume
      w.sched().taskq().push_bottom(se);
    }

    template <class T>
    inline bool future_join_suspended_range(ityr::ito::scheduler_randws::suspended_state& se,
                                            ityr::ito::future<T>* fs, std::size_t n) {
      auto& w = ityr::ito::worker::instance::get();
      auto& alloc = w.sched().fpool().remote_bufs_allocator_;
      for (std::size_t i = 0; i < n; ++i) {
        if (remote_get_value(alloc, &fs[i].e->done) == 0) {
          if (push_suspended_state(fs[i].e, se)) return true;
        }
      }
      return false;
    }

    template <class T>
    inline void wait_all(ityr::ito::future<T>* fs, std::size_t n) {
      ITYR_CHECK(n <= 15);   // waitall_entries[16] (1 slot for the 0 terminator)
      bool all_ready = true;
      for (std::size_t i = 0; i < n; ++i)
        if (!fs[i].is_ready()) { all_ready = false; break; }
      if (all_ready) return;

      auto& w = ityr::ito::worker::instance::get();
      w.sched().suspend([&](ityr::ito::context_frame* cf) {
        auto se = w.sched().evacuate(cf);
        se.waitall_handoff = &waitall_handoff_impl_range<T>;
        for (std::size_t i = 0; i < n; ++i)
          se.waitall_entries[i] = reinterpret_cast<intptr_t>(fs[i].e);
        se.waitall_entries[n] = 0;

        if (!future_join_suspended_range(se, fs, n)) {
          w.sched().resume(se);
        } else {
          continue_computation();
        }
      });
    }

    // Fetch the value of a future whose completion is guaranteed (i.e. after
    // wait_all): like future<T>::get() but skips the redundant done-check.
    // Data is copied/cached exactly as in get() (per-rank copy cache).
    template <class F>
    inline typename future_value<F>::type get_ready_value(F& f) {
      auto& w = ityr::ito::worker::instance::get();
      auto& fpool = w.sched().fpool();
      typename future_value<F>::type v;
      fpool.sync_resume(f.e, &v, NULL);
      return v;
    }

    // Get all given futures' values with a single suspension (up to 5 futures):
    // wait_all to guarantee completion, then fetch each value without the
    // per-future done-check that N separate get()s would do. Returns a tuple of
    // the values, in the order of the futures.
    template <class F1, class... Fs>
    inline auto get_all(F1& f1, Fs&... fs) {
      static_assert(1 + sizeof...(Fs) <= 5, "get_all supports at most 5 futures");
      wait_all(f1, fs...);
      return std::make_tuple(get_ready_value(f1), get_ready_value(fs)...);
    }

    // Fetch the raw value pointer of a future whose completion is guaranteed
    // (i.e. after wait_all): like future<T>::get_pointer() but for a batch —
    // returns the producer's stored pointer, NO copy, NO per-rank cache. The
    // caller must treat it like get_pointer() (e.g. the producer's buffer may
    // be remote; do not dereference unless locally accessible).
    template <class F>
    inline typename future_value<F>::type get_ready_pointer(F& f) {
      auto& w = ityr::ito::worker::instance::get();
      auto& alloc = w.sched().fpool().remote_bufs_allocator_;
      return (typename future_value<F>::type)remote_get_value(alloc, &f.e->value);
    }

    // Get all given futures' raw stored pointers with a single suspension
    // (up to 5 futures): wait_all to guarantee completion, then read each
    // future's value pointer directly (no copy/cache, like get_pointer()).
    // Returns a tuple of the pointers, in the order of the futures.
    template <class F1, class... Fs>
    inline auto get_all_pointer(F1& f1, Fs&... fs) {
      static_assert(1 + sizeof...(Fs) <= 5, "get_all_pointer supports at most 5 futures");
      wait_all(f1, fs...);
      return std::make_tuple(get_ready_pointer(f1), get_ready_pointer(fs)...);
    }
}
