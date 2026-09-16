# Pure-math check - no TTG runtime needed.
add_executable(test_force_kernel test_force_kernel.cc)
add_test(NAME test_force_kernel COMMAND test_force_kernel)

# Standalone tree-build perf probe - no TTG needed. Built with optimizations
# regardless of the project's (Debug, to match the linked TTG libs) build
# type, since this is a pure host-code benchmark that never links TTG.
add_executable(bench_tree bench_tree.cc "../src/tree.cc")
target_compile_options(bench_tree PRIVATE -O2)

# These exercise the full TTG pipeline (tree build + both TTG passes).
add_ttg_executable(agg_repro "agg_repro.cc"
  RUNTIMES "parsec" SINGLERANKONLY NOT_EXCLUDE_FROM_ALL)

# Ad hoc comparison probe against PEPC (see ../tools/gen_shared_ic.py,
# ../tools/parse_pepc_vtu.py, ../tools/compare_pepc.cc for the rest of the
# comparison workflow) - not a correctness test itself (nothing to assert
# against without a matching PEPC run), just built + smoke-run via the
# auto-generated test so its own build/runtime keeps getting checked.
add_ttg_executable(compare_pepc_theta "compare_pepc_theta.cc;../src/tree.cc;../src/io.cc"
  RUNTIMES "parsec" SINGLERANKONLY NOT_EXCLUDE_FROM_ALL
  TEST_CMDARGS "${CMAKE_SOURCE_DIR}/data/shared_ic_20.csv;0.5;0.0001;8")

add_ttg_executable(test_bh_vs_bruteforce "test_bh_vs_bruteforce.cc;../src/tree.cc"
  RUNTIMES "parsec" SINGLERANKONLY NOT_EXCLUDE_FROM_ALL)
add_test(NAME test_bh_vs_bruteforce
  COMMAND /usr/local/bin/mpirun -n 1 $<TARGET_FILE:test_bh_vs_bruteforce-parsec>)

add_ttg_executable(test_energy_conservation "test_energy_conservation.cc;../src/tree.cc"
  RUNTIMES "parsec" SINGLERANKONLY NOT_EXCLUDE_FROM_ALL)
add_test(NAME test_energy_conservation
  COMMAND /usr/local/bin/mpirun -n 1 $<TARGET_FILE:test_energy_conservation-parsec>)
