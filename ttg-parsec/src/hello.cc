#include <ttg.h>

int main(int argc, char** argv) {
  ttg::initialize(argc, argv, -1);

  ttg::Edge<int, void> seed;
  auto hello = ttg::make_tt(
      [](const int& n) { ttg::print("Hello TTG from pepc-ttg, n =", n); },
      ttg::edges(seed), ttg::edges(), "hello");

  ttg::make_graph_executable(hello.get());
  ttg::execute();
  if (ttg::default_execution_context().rank() == 0) {
    hello->template in<0>()->sendk(42);
  }
  ttg::fence();

  ttg::finalize();
  return 0;
}
