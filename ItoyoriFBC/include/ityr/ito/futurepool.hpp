#include "ityr/ito/future_header.hpp"
#include "ityr/ito/sched/scheduler.hpp"

namespace ityr::ito {

  template<class T>
  inline void future_pool::fill(future<T>& f, T &value, size_t value_size) {
    auto *e = f.e;

    // ORDERING CONTRACT: a consumer sync()s on e->done (RMA F&O) and then
    // issues its MPI_Get of the entry fields and of the data buffer. For the
    // get to observe the data, the producer must make its plain stores to the
    // data buffer visible to remote RMA ops BEFORE the consumer can observe
    // done. That requires (1) writing back the dirty cache lines (the get's
    // DMA bypasses this rank's CPU cache; MPI_Win_sync alone is only a fence)
    // and (2) MPI_Win_sync, ALL BEFORE publishing done. Publishing done first
    // (as an earlier version did) lets the consumer's get race ahead of the
    // flush and read stale content.
    size_t data_size = value_size & (((unsigned long int) -1) >> 1UL);
    void* data_ptr = reinterpret_cast<void*>(value);
    if (data_size > 0 && remote_bufs_allocator_.is_locally_accessible(data_ptr)) {
      ityr::common::cache_flush(data_ptr, data_size);
    }
    ityr::common::mpi_win_sync(remote_bufs_allocator_.win());
    // Publish the entry fields through the RMA layer (MPI_Put + flush), never
    // via plain stores: a consumer's remote_get_value (MPI_Get) can observe
    // plain stores as stale (they may sit in the producer's CPU cache), and
    // the get is only ordered against RMA operations to the same target.
    ityr::common::remote_put_value(remote_bufs_allocator_, value, &e->value);
    ityr::common::remote_put_value(remote_bufs_allocator_, value_size, &e->value_size);
    // done LAST: the consumer's sync() reads it via F&O, and its subsequent
    // get of the data is ordered after this faa, hence after the publish.
    remote_faa_value(remote_bufs_allocator_, 1, &e->done);
  }

  inline void future_pool::flush() {
    ityr::common::mpi_win_sync(remote_bufs_allocator_.win());
  }

  template<class T>
  inline void future_pool::flush_data(T* data, size_t data_size) {
    void* p = reinterpret_cast<void*>(data);
    if (data_size > 0 && remote_bufs_allocator_.is_locally_accessible(p)) {
      ityr::common::cache_flush(p, data_size);
    }
    ityr::common::mpi_win_sync(remote_bufs_allocator_.win());
  }

  // ---- per-rank cache of get() copies ----
  inline void* future_pool::get_cached_copy(void* e) {
    auto it = copy_cache_.find(e);
    if (it != copy_cache_.end()) {
      copy_cache_hits_++;
      return it->second.copy;
    }
    copy_cache_misses_++;
    return nullptr;
  }

  inline void future_pool::cache_copy(void* e, void* copy, size_t copy_size,
                                      void* original, size_t original_size) {
    copy_cache_.emplace(e, copy_cache_entry{copy, copy_size, original, original_size});
  }

  inline bool future_pool::release_cached_copy(void* p) {
    for (auto it = copy_cache_.begin(); it != copy_cache_.end(); ++it) {
      if (it->second.copy == p) {
        remote_bufs_allocator_.deallocate(p, it->second.copy_size);
        copy_cache_.erase(it);
        return true;
      }
    }
    return false;
  }

  inline void future_pool::adopt_cached_copy(void* p) {
    for (auto it = copy_cache_.begin(); it != copy_cache_.end(); ++it) {
      if (it->second.copy == p) {
        copy_cache_.erase(it);
        return;
      }
    }
  }

  inline void future_pool::purge_copies_of_original(void* p, size_t n) {
    uintptr_t lo = reinterpret_cast<uintptr_t>(p);
    uintptr_t hi = lo + n;
    for (auto it = copy_cache_.begin(); it != copy_cache_.end();) {
      uintptr_t o = reinterpret_cast<uintptr_t>(it->second.original);
      if (lo <= o && o < hi) {
        remote_bufs_allocator_.deallocate(it->second.copy, it->second.copy_size);
        it = copy_cache_.erase(it);
      } else {
        ++it;
      }
    }
  }

  inline std::size_t future_pool::cached_copy_entries() const {
    return copy_cache_.size();
  }

  inline void future_pool::clear_copy_cache() {
    for (auto& kv : copy_cache_) {
      remote_bufs_allocator_.deallocate(kv.second.copy, kv.second.copy_size);
    }
    copy_cache_.clear();
    if (std::getenv("ITYR_DEBUG_COPY_CACHE")) {
      fprintf(stderr,
              "[ityrcc] rank=%d hits=%llu misses=%llu final_cache=%zu\n",
              ityr::common::topology::my_rank(),
              (unsigned long long)copy_cache_hits_,
              (unsigned long long)copy_cache_misses_,
              cached_copy_entries());
      fflush(stderr);
    }
  }

  template<class T>
  inline void future_pool::return_future_id(ityr::ito::scheduler_randws::entry<T> *e) {
    remote_bufs_allocator_.deallocate(e, sizeof(ityr::ito::scheduler_randws::entry<T>));
  }

  template<class T>
  inline bool future_pool::sync(future<T>& f, T *value, void *target_pointer) {
    auto *e = f.e;

    int flag = remote_get_value(remote_bufs_allocator_, &e->done);

    if (flag > 0) {
      // The target thread has already been completed

      asm volatile("":::"memory");
#ifdef ITYR_COPY_FUTURE_DATA
      *value = remote_get_value(remote_bufs_allocator_, &e->value);
      size_t value_size = remote_get_value(remote_bufs_allocator_, &e->value_size);

      bool no_copy = value_size & (1UL << 63UL);
      value_size = value_size & (((unsigned long int) -1) >> 1UL);
      if (value_size > 0 && !remote_bufs_allocator_.is_locally_accessible(*reinterpret_cast<int**>(value))) {
        // Remote data: reuse this rank's cached copy if this entry was already
        // read here (all consumers of a future share one copy), else copy once
        // and cache it. target_pointer bypasses the cache (caller wants the
        // data IN a specific buffer, e.g. a par_map concat).
        if (target_pointer == NULL) {
          if (void* cached = get_cached_copy(e)) {
            *reinterpret_cast<T**>(value) = reinterpret_cast<T*>(cached);
            return true;
          }
        }
        T *value_copy =
            target_pointer == NULL ? (T *) remote_bufs_allocator_.allocate(value_size) : (T *) target_pointer;
        if (!no_copy) {
          remote_get(remote_bufs_allocator_, (int *) value_copy, *reinterpret_cast<int **>(value),
                     value_size / sizeof(int));
        }
        void* original = *reinterpret_cast<void**>(value);
        *reinterpret_cast<T **>(value) = value_copy;
        if (target_pointer == NULL && !no_copy) {
          cache_copy(e, value_copy, value_size, original, value_size);
        }
      } else if (value_size > 0 && target_pointer != NULL && target_pointer != *reinterpret_cast<T**>(value)) {
      std::memmove(target_pointer, *reinterpret_cast<T**>(value), value_size);
      *reinterpret_cast<T**>(value) = (T*)target_pointer;
    }
#else
      *value = remote_get_value(remote_bufs_allocator_, &e->value);
#endif

      return true;
    } else {
      // The target thread can be still running, so suspend the current thread
      return false;
    }
  }

  template<class T>
  inline bool future_pool::sync_suspended(future<T>& f,
                                          ityr::ito::scheduler_randws::suspended_state se) {
    auto *e = f.e;

    if (remote_get_value(remote_bufs_allocator_, &e->done) == 0) {
      // the target thread is still running, so let the thread resume
      // the current thread when completed
      return !f.push_suspended_state(se);
    } else {
      // the target thread has already been completed
      return true;
    }
  }

  template<class T>
  inline void future_pool::sync_resume(ityr::ito::scheduler_randws::entry<T>* e, T *value, void *target_pointer) {
#ifdef ITYR_COPY_FUTURE_DATA
      *value = remote_get_value(remote_bufs_allocator_, &e->value);
      size_t value_size = remote_get_value(remote_bufs_allocator_, &e->value_size);

      bool no_copy = value_size & (1UL << 63UL);
      value_size = value_size & (((unsigned long int) -1) >> 1UL);
      if (value_size > 0 && !remote_bufs_allocator_.is_locally_accessible(*reinterpret_cast<int**>(value))) {
        // Remote data: reuse this rank's cached copy if this entry was already
        // read here, else copy once and cache it (see sync()).
        if (target_pointer == NULL) {
          if (void* cached = get_cached_copy(e)) {
            *reinterpret_cast<T**>(value) = reinterpret_cast<T*>(cached);
            return;
          }
        }
        T *value_copy = target_pointer == NULL ? (T *) remote_bufs_allocator_.allocate(value_size) : (T*)target_pointer;
        if (!no_copy) {
          remote_get(remote_bufs_allocator_, (int *) value_copy, *reinterpret_cast<int **>(value),
                     value_size / sizeof(int));
        }
        void* original = *reinterpret_cast<void**>(value);
        *reinterpret_cast<T**>(value) = value_copy;
        if (target_pointer == NULL && !no_copy) {
          cache_copy(e, value_copy, value_size, original, value_size);
        }
      } else if (value_size > 0 && target_pointer != NULL && target_pointer != *reinterpret_cast<T**>(value)) {
        std::memmove(target_pointer, *reinterpret_cast<T**>(value), value_size);
        *reinterpret_cast<T**>(value) = (T*)target_pointer;
      }
#else
      *value = remote_get_value(remote_bufs_allocator_, &e->value);
#endif
  }
}

namespace ityr::ito {
// Make plain stores made by THIS rank into window/future-data buffers visible
// to RMA GETs from other ranks (e.g. after in-place updates outside a task
// completion). Task completions (fill/make_filled) flush automatically.
inline void flush_window_data(void* data, std::size_t data_size) {
  auto& w = ityr::ito::worker::instance::get();
  w.sched().fpool().flush_data(data, data_size);
}

// RMA-ordered, SPMD-safe get of a buffer published through a future entry by a
// task on another rank (fill()/make_filled()). The caller is in the SPMD
// section (NOT inside a task), so this NEVER suspends or migrates: it spins on
// the entry's done flag via remote_get_value (RMA-ordered after the producer's
// flush + publish), then reads the published pointer, then (optionally) copies
// `bytes` of that buffer into `dest`. Returns the published pointer.
// This is the SPMD-side counterpart of future<T>::get_pointer()/get() and the
// building block for per-node replicas.
template <typename T>
T* spmd_get(const void* entry_ptr, T* dest = nullptr, size_t bytes = 0) {
  auto& alloc = worker::instance::get().sched().fpool().remote_bufs_allocator_;
  auto* e = (ityr::ito::scheduler_randws::entry<T*>*)entry_ptr;
  while (ityr::common::remote_get_value(alloc, &e->done) == 0) { }
  T* src = ityr::common::remote_get_value(alloc, &e->value);
  if (dest) ityr::common::remote_get(alloc, dest, src, bytes / sizeof(T));
  return src;
}
} // namespace ityr::ito
