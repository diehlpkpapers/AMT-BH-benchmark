#pragma once


/*
    This file contains the declarations of the classes future and future_pool.
*/

namespace ityr::ito {

#include <vector>
#include <cstdint>
#include <unordered_map>
#include "ityr/ito/sched/randws_header.hpp"
#include "ityr/ito/worker.hpp"

// This prototype declaration is only needed to satisfy the compiler.
namespace worker {
  class worker;
}

void continue_computation();

/**
 * @brief A class representing a future value with dependency tracking.
 * 
 * The `future` class template provides an interface for asynchronous computation results.
 * It allows for creation, assignment, waiting,
 * and retrieval of results, as well as checking readiness and discarding dependencies.
 * 
 * @tparam T      The type of the value held by the future.
 *
 * Features:
 * - Creation of empty or pre-filled futures.
 * - Support for dependency-aware waiting and result retrieval.
 * 
 * Usage:
 * - Use `make()` to create a future.
 * - Use `make_filled()` to create a ready future with a result.
 *  This can be used as a workoround for copying data on the heap between workers.
 * - Use `set()` to assign a value to the future.
 * - Use `wait()` or `get()` to synchronize and retrieve the result.
 * - Use `is_ready()` to check if the result is available.
 */

#ifdef ITYR_COPY_FUTURE_DATA
  static __attribute__((unused)) void* alloc_future_data(size_t numBytes);
  static __attribute__((unused)) void dealloc_future_data(void *p, size_t numBytes);
#endif
    template <class T>
    class future {
        friend class future_pool;
    private:

    public:
        ityr::ito::scheduler_randws::entry<T>* e;

        future();
        future(ityr::ito::scheduler_randws::entry<T> *e);
        future(const future<T>& other);
        future(future<T>&& other);
        ~future();

        static future<T> make();
        static future<T> make(ityr::ito::worker::worker& w);

        static future<T> make_filled(T result, size_t size=0, bool no_copy=false);
        static future<T> make_filled_pair(T pair, bool no_copy=false);

        future<T>& operator=(future<T>&& other);
        future<T>& operator=(const future<T>& other);

        bool push_suspended_state(ityr::ito::scheduler_randws::suspended_state ss);
        std::vector<ityr::ito::scheduler_randws::suspended_state> pop_suspended_states();

        void increment_gc_counter(int value);

        void set(T& value, [[maybe_unused]] size_t value_size=0);

        void wait();
        T get(void *target_pointer=NULL);

        bool is_ready();
        T* get_pointer();

        // Byte size of the produced value buffer (the no_copy flag bit is
        // masked out). Mirrors what future_pool::sync reads from e->value_size.
        std::size_t size();

        void discard();
    };

    inline size_t index_of_size(size_t size);


    /**
     * @brief Manages allocation, synchronization, and lifecycle of future objects.
     *
     * The future_pool class is responsible for efficiently managing a pool of future objects,
     * which represent asynchronous computation results. It handles allocation and recycling
     * of future entries, synchronization of results between tasks or threads, and memory
     * management for future data. The pool also provides mechanisms for filling, waiting,
     * and discarding futures, as well as for integrating with distributed or parallel
     * execution environments.
     *
     * Key features:
     * - Efficient allocation and reuse of future entries to minimize overhead.
     * - Synchronization primitives for waiting on and retrieving future results.
     * - Support for discarding and resetting futures to manage dependencies and memory.
     * - Integration with remote memory resources for distributed systems.
     * - Mechanisms for collective operations on futures.
     *
     * Typical usage:
     * - Use get() to allocate a new future from the pool.
     * - Use fill() to assign a value to a future and notify dependents.
     * - Use sync() or sync_resume() to wait for and retrieve results.
     * - Use initialize() and finalize() to manage the pool's lifecycle.
     */
    
    class future_pool /* : noncopyable */ {

        enum constants {
            MAX_ENTRY_BITS = 16,
        };

        int locally_freed_val_  = 417;
        int remotely_freed_val_ = 418;

    public:
        future_pool(size_t buf_size);
        ~future_pool();

        void initialize();
        void finalize();

        template <class T>
        future<T> get();

        template <class T>
        void fill(future<T>& f, T& value, size_t value_size=0);

        template <class T>
        bool sync(future<T>& f, T *value, void *target_pointer=NULL);

        template <class T>
        bool sync_suspended(future<T>& f,
                            ityr::ito::scheduler_randws::suspended_state se);

        template <class T>
        void sync_resume(ityr::ito::scheduler_randws::entry<T> *e, T *value, void *target_pointer=NULL);

        template <class T>
        void return_future_id(ityr::ito::scheduler_randws::entry<T> *e);

        void flush();

        template <class T>
        void flush_data(T* data, std::size_t data_size);

        // ---- per-rank cache of get() copies ----
        // A remote future's data is copied at most once per rank; every
        // consumer on this rank shares that copy (future data is write-once /
        // read-only by contract, so a copy never goes stale). Copies are freed
        // when the app deallocs the copy itself (release_cached_copy), when the
        // app deallocs the ORIGINAL buffer the copy was made from
        // (purge_copies_of_original — covers sub-range futures), or at
        // finalize(). adopt_cached_copy hands a copy over to the app without
        // freeing it (the app mutated/adopted it, e.g. the forces bodies).
        struct copy_cache_entry {
          void* copy;
          std::size_t copy_size;
          void* original;
          std::size_t original_size;
        };
        std::unordered_map<void*, copy_cache_entry> copy_cache_;

        void* get_cached_copy(void* e);
        void cache_copy(void* e, void* copy, std::size_t copy_size,
                        void* original, std::size_t original_size);
        bool release_cached_copy(void* p);
        void adopt_cached_copy(void* p);
        void purge_copies_of_original(void* p, std::size_t n);
        std::size_t cached_copy_entries() const;
        void clear_copy_cache();

        // DIAG (env ITYR_DEBUG_COPY_CACHE=1): print per-rank cache stats.
        std::uint64_t copy_cache_hits_ = 0;
        std::uint64_t copy_cache_misses_ = 0;

      ityr::common::remotable_resource remote_bufs_allocator_;
    };



} // end namespace
