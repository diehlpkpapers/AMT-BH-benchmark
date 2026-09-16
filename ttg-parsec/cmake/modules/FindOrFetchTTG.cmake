# Pulls in TTG as a real CMake subproject (via FetchContent) rather than
# consuming a pre-built/installed copy, so that WE control its build
# options - CMAKE_BUILD_TYPE and TTG_ENABLE_TRACE most importantly - instead
# of inheriting whatever an ad hoc prior build happened to be configured
# with (the one this project originally linked was Debug + TTG_ENABLE_TRACE
# ON, neither of which is representative of the runtime's real overhead).
#
# This deliberately does NOT try to reuse the existing hand-installed
# PaRSEC/MADNESS from that prior build: PaRSEC's own task-scheduler code is
# a much likelier source of measured per-task overhead than TTG's thin C++
# layer, so a fair comparison needs PaRSEC (and MADNESS, which TTG's
# serialization layer always links) rebuilt at the SAME optimization level
# as TTG, not reused as-is. TTG's own FindOrFetchPARSEC/FindOrFetchMADNESS
# modules will fetch+build fresh copies, inheriting whatever
# CMAKE_BUILD_TYPE this project's top-level CMakeLists.txt sets.
#
# TTG_GIT_REPOSITORY/TTG_GIT_TAG pick the fork+branch to fetch: our
# aggregator fix (see aggregator.h's make_aggregator - TargetFn needs
# std::decay_t, same as EdgeT already gets, or AggregatorFactory stores a
# dangling reference to the caller's target-function closure) is committed
# there, not on upstream TESSEorg/ttg.

if (NOT TARGET ttg)
  find_package(ttg CONFIG QUIET)
endif()

if (TARGET ttg)
  message(STATUS "Found ttg CONFIG at ${ttg_CONFIG}")
else()
  set(TTG_GIT_REPOSITORY "git@github.com:devreal/ttg.git" CACHE STRING
    "TTG git repository to fetch")
  set(TTG_GIT_TAG "ttg-kernel-batching_aggregators" CACHE STRING
    "TTG git branch/tag/commit to fetch")

  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(TTG_EXAMPLES OFF CACHE BOOL "" FORCE)

  include(FetchContent)
  FetchContent_Declare(
    ttg
    GIT_REPOSITORY "${TTG_GIT_REPOSITORY}"
    GIT_TAG "${TTG_GIT_TAG}"
  )
  FetchContent_MakeAvailable(ttg)
endif()

if (NOT TARGET ttg)
  message(FATAL_ERROR "FindOrFetchTTG could not make the ttg target available")
endif()
