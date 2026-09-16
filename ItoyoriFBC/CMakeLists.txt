cmake_minimum_required(VERSION 3.16)
project(itoyori-barneshut CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED)

add_library(itoyori INTERFACE)
target_include_directories(itoyori INTERFACE
  ${MPI_CXX_INCLUDE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(itoyori INTERFACE ${MPI_CXX_LIBRARIES} rt)
target_compile_options(itoyori INTERFACE -Wall -Wextra -fno-stack-protector)
target_compile_definitions(itoyori INTERFACE ITYR_MAX_VERBOSE_LEVEL=0)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(itoyori INTERFACE -Wno-interference-size)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(itoyori INTERFACE -Wno-unused-private-field)
endif()

if(MPI_CXX_COMPILE_FLAGS)
  separate_arguments(MPI_CXX_COMPILE_OPTIONS UNIX_COMMAND "SHELL:${MPI_CXX_COMPILE_FLAGS}")
  target_compile_options(itoyori INTERFACE ${MPI_CXX_COMPILE_OPTIONS})
endif()
if(MPI_CXX_LINK_FLAGS)
  separate_arguments(MPI_CXX_LINK_OPTIONS UNIX_COMMAND "SHELL:${MPI_CXX_LINK_FLAGS}")
  target_link_options(itoyori INTERFACE ${MPI_CXX_LINK_OPTIONS})
endif()

include(CheckIncludeFileCXX)
check_include_file_cxx(numa.h HAS_NUMA)
if(HAS_NUMA)
  target_link_libraries(itoyori INTERFACE numa)
endif()
check_include_file_cxx(utofu.h HAS_UTOFU)
if(HAS_UTOFU)
  target_link_libraries(itoyori INTERFACE tofucom)
endif()

option(BARNESHUT_MARCH_NATIVE "Build with -march=native (build and run on the same CPU type)" ON)
option(BARNESHUT_AVX512 "Compile the AVX-512F backend (x86_64; Skylake-SP+)" OFF)
option(BARNESHUT_SVE "Compile the SVE backend (aarch64)" OFF)

add_executable(barneshut.out src/barneshut_ityr.cpp)
target_link_libraries(barneshut.out itoyori)
target_compile_definitions(barneshut.out PRIVATE ITYR_COPY_FUTURE_DATA=1)

# No FP contraction: the SIMD force kernel uses explicit mul/add intrinsics,
# so the scalar kernel must not be contracted to FMA either.
target_compile_options(barneshut.out PRIVATE -ffp-contract=off)

# ISA extensions: the SIMD backend is selected at runtime among the ones that
# are compiled in.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
  target_compile_options(barneshut.out PRIVATE -mavx2)
  if(BARNESHUT_AVX512)
    target_compile_options(barneshut.out PRIVATE -mavx512f)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  if(BARNESHUT_SVE)
    target_compile_options(barneshut.out PRIVATE -march=armv8-a+sve)
  endif()
endif()
if(BARNESHUT_MARCH_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(barneshut.out PRIVATE -march=native)
endif()
