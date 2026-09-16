add_ttg_executable(hello "hello.cc"
  RUNTIMES "parsec" SINGLERANKONLY NOT_EXCLUDE_FROM_ALL)

add_ttg_executable(pepc-ttg "main.cc;tree.cc;io.cc"
  RUNTIMES "parsec" SINGLERANKONLY NOT_EXCLUDE_FROM_ALL
  TEST_CMDARGS "--input;${CMAKE_SOURCE_DIR}/data/two_body.csv;--steps;2;--energy-every;0")
