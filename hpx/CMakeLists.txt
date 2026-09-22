cmake_minimum_required(VERSION 3.16.3)
project(ParallelNBodySimulation VERSION 1.0 LANGUAGES CXX)

# Default to a Release build for performance if not specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif()

# core library
add_library(solar_system_lib STATIC
  src/linear_octree.cpp
  src/morton_keys.cpp
  src/bounding_box.cpp
  src/load_balancing.cpp
  src/exchange.cpp
  src/io.cpp
  src/traversal.cpp
  src/energy.cpp
  src/tinyxml2.cpp
)

target_compile_features(solar_system_lib PUBLIC cxx_std_17)
target_include_directories(solar_system_lib PUBLIC include)
target_compile_options(solar_system_lib PUBLIC
  # warnings  (all configs, GCC/Clang)
  $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
     -Wall -Wextra -Wpedantic>


  $<$<CXX_COMPILER_ID:GNU,Clang>:-include ${CMAKE_SOURCE_DIR}/include/fix_includes.h>

  # warnings  (all configs, MSVC)
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>



  # performance flags – **only in Release** and only for GCC/Clang
  $<$<AND:$<CONFIG:Release>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:
     -march=native -ffast-math>
)
target_compile_definitions(solar_system_lib PUBLIC _USE_MATH_DEFINES)

# HPX (provides both the loop-level parallelism and the distributed
# collectives that OpenMP/MPI provided on the mpi_reference branch)
find_package(HPX REQUIRED)
target_link_libraries(solar_system_lib PUBLIC HPX::hpx)

# main executable
add_executable(simulate src/main.cpp)
target_link_libraries(simulate PRIVATE solar_system_lib HPX::wrap_main)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Configured")