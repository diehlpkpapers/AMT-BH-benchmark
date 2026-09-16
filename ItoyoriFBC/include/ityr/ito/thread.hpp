#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ito/worker.hpp"
#include "ityr/ito/sched/scheduler.hpp"
#include "ityr/ito/future_header.hpp"

namespace ityr::ito {


/**
 * @brief Represents a thread that operates on data of type T.
 * 
 * @tparam T The type of data the thread will process or manage.
 *
 * This class provides mechanisms for managing and executing threads
 * that work with objects of type T.
 *
 * Remember that thread is not an alias for worker but for task in this project
 */
template<typename T>
class thread {
public:
  thread();

  template <typename Fn, typename... Args>
  explicit thread(Fn&& fn, Args&&... args);

  thread(const thread&) = delete;
  thread& operator=(const thread&) = delete;

  thread(thread&& th) = default;
  thread& operator=(thread&& th) = default;

  template <typename Fn, typename... Args>
  void fork(Fn&& fn, Args&&... args);

  T join();

  future<T> getFuture() const {
    return fut;
  };

private:
  template <class F, class... Args>
  static void start(future<T> fut, F f, Args... args);

  future<T> fut;
};

  template<typename T>
  thread<T>::thread() {}

  template<typename T>
  template <typename Fn, typename... Args>
  thread<T>::thread(Fn&& fn, Args&&... args) {
    fut = future<T>::make();
    fork(start<Fn, Args...>, fut, std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  template <typename T, typename Fn, typename... Args>
  static auto spawn(Fn&& fn, Args&&... args) {
    if constexpr (std::is_void_v<T>) {
      // This block compiles only if T is void
      thread<void>(std::forward<Fn>(fn), std::forward<Args>(args)...);
      // Returns void implicitly
    } else {
      // This block compiles only if T is NOT void
      return thread<T>(std::forward<Fn>(fn), std::forward<Args>(args)...).getFuture();
    }
  }

  template<typename T>
  template <typename Fn, typename... Args>
  void thread<T>::fork(Fn&& fn, Args&&... args) {
    auto& w = worker::instance::get();
    ITYR_CHECK(!w.is_spmd());
    w.sched().fork(std::forward<Fn>(fn), std::forward<Args>(args)...);
  }


  template<typename T>
  T thread<T>::join() {
    return fut.get();
  }


  template<typename T>
  template <class F, class... Args>
  void thread<T>::start(future<T> fut, F f, Args... args) {
#ifdef ITYR_COPY_FUTURE_DATA
    auto ret = f(std::forward<Args>(args)...);
    if constexpr (std::is_same_v<decltype(ret), int>) {
      fut.set(ret);
    } else {
      fut.set(ret.first, ret.second);
    }
#else
    T value = f(std::forward<Args>(args)...);
    fut.set(value);
#endif
    continue_computation();
  }

  template<>
  class thread<void> {
  public:
    thread() {}

    template <typename Fn, typename... Args>
    explicit thread(Fn&& fn, Args&&... args) {
      fork(start<Fn, Args...>, std::forward<Fn>(fn), std::forward<Args>(args)...);
    }

   
    thread(const thread&) = delete;
    thread& operator=(const thread&) = delete;

    thread(thread&& th) = default;
    thread& operator=(thread&& th) = default;

    template <typename Fn, typename... Args>
    void fork(Fn&& fn, Args&&... args) {
      auto& w = worker::instance::get();
      ITYR_CHECK(!w.is_spmd());
      w.sched().fork(std::forward<Fn>(fn), std::forward<Args>(args)...);
    }


    void join() {
    }

  private:

    template <class F, class... Args>
    static void start(F f, Args... args) {
      f(args...);
      continue_computation();
    }
  };

}
