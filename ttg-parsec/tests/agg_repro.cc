#include <ttg.h>

#include <iostream>
#include <unordered_map>
#include <vector>

struct Payload {
  int value = 0;
  template <typename Archive>
  void serialize(Archive& ar) { ar & value; }
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) { ar & value; }
};

int main(int argc, char** argv) {
  ttg::initialize(argc, argv, -1);
  ttg::execute();

  // Exact structure of the tree that originally triggered the bug:
  //   0,1 -> 2 -> 4 -> 10 (root)
  //   3 -> 4
  //   5,8 -> 9 -> 10
  //   6,7 -> 8
  std::unordered_map<int, std::size_t> child_count = {{2, 2}, {4, 2}, {8, 2}, {9, 2}, {10, 2}};
  std::unordered_map<int, int> parent_of = {
      {0, 2}, {1, 2}, {2, 4}, {3, 4}, {4, 10}, {5, 9}, {6, 8}, {7, 8}, {8, 9}, {9, 10}};
  std::vector<int> leaves = {0, 1, 3, 5, 6, 7};
  int root = 10;

  ttg::Edge<int, void> seed_edge;
  ttg::Edge<int, Payload> raw_edge;

  auto seed_tt = ttg::make_tt(
      [&parent_of](const int& leaf_id, std::tuple<ttg::Out<int, Payload>>& outs) {
        ttg::send<0>(parent_of.at(leaf_id), Payload{leaf_id * 10}, outs);
      },
      ttg::edges(seed_edge), ttg::edges(raw_edge), "seed_leaf");

  auto target_fn = [&child_count](const int& key) -> std::size_t {
    std::size_t sz = child_count.at(key);
    std::cerr << "target_fn: &child_count=" << &child_count << " key=" << key << " -> " << sz
              << "\n";
    return sz;
  };
  auto agg_edge = ttg::make_aggregator(raw_edge, target_fn);

  auto combine_up_tt = ttg::make_tt(
      [&parent_of, root](const int& key, const ttg::Aggregator<Payload>& agg,
                          std::tuple<ttg::Out<int, Payload>>& outs) {
        int sum = 0;
        for (auto&& v : agg) sum += v.value;
        std::cerr << "combine_up fired for key=" << key << " agg.size()=" << agg.size()
                  << " sum=" << sum << "\n";
        if (key != root) {
          ttg::send<0>(parent_of.at(key), Payload{sum}, outs);
        }
      },
      ttg::edges(agg_edge), ttg::edges(raw_edge), "combine_up");

  ttg::make_graph_executable(seed_tt.get(), combine_up_tt.get());
  if (ttg::default_execution_context().rank() == 0) {
    for (int leaf : leaves) seed_tt->template in<0>()->sendk(leaf);
  }
  ttg::fence();
  ttg::finalize();
  return 0;
}
