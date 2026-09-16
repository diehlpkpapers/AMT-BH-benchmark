cmake_minimum_required(VERSION 3.19)
project(pepc-ttg CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (NOT CMAKE_BUILD_TYPE)
  # Default to Release: this project exists partly to measure TTG/PaRSEC
  # per-task overhead, which an unoptimized build would badly distort.
  # Override with -DCMAKE_BUILD_TYPE=Debug if you actually want that.
  set(CMAKE_BUILD_TYPE Release)
endif()

# Reuse the already-cloned PaRSEC/MADNESS/etc. sources from the prior TTG
# build (same repo URLs/tags, just built with different flags there) so
# FetchContent only needs to reconfigure+rebuild them, not re-clone from
# GitHub.
set(FETCHCONTENT_BASE_DIR "/Users/joseph/src/ttg/ttg/build/_deps" CACHE PATH "")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")
include(FindOrFetchTTG)

# add_ttg_executable() is normally made available via ttg-config.cmake
# (install-time consumption). TTG is pulled in here as a live subproject
# instead (via add_subdirectory, under the hood, in FetchContent), so its
# own `list(APPEND CMAKE_MODULE_PATH ...)` only affected ITS OWN directory
# scope, not ours - add its modules dir here too, using the ttg_SOURCE_DIR
# FetchContent_MakeAvailable(ttg) set in our scope.
list(APPEND CMAKE_MODULE_PATH "${ttg_SOURCE_DIR}/cmake/modules")
include(AddTTGExecutable)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
