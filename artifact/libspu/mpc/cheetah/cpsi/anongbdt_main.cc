#include <omp.h>

#include <boost/program_options.hpp>
#include <chrono>
#include <fstream>
#include <regex>
#include <string>
#include <map>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/link/context.h"
#include "yacl/link/test_util.h"

#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/cpsi/anon_xgb.h"
#include "libspu/mpc/cheetah/cpsi/histogram.h"
#include "libspu/mpc/cheetah/cpsi/main_io.h"

using namespace std;
using namespace spu::mpc::cheetah;

int main(int32_t argcp, char **argvp) {
  namespace po = boost::program_options;
  po::options_description allowed("Allowed options");
  std::string type;
  std::string alice_address, bob_address, input_path, output_path;
  int role, alice_port, bob_port, num_thread;
  double lambda;
  size_t depth, tree_num, log2_scale, max_bin, n=0, m=0;
  double positive_label, negative_label;
  std::vector<std::vector<double>> bin_bound;
  bool have_label;
  allowed.add_options()("help,h", "produce this message")
      ("role,r", po::value<decltype(role)>(&role)->required(),"Role of the node")
      ("alice-address,a", po::value<decltype(alice_address)>(&alice_address)->default_value("0.0.0.0"), "IP address of Alice")
      ("bob-address,b", po::value<decltype(bob_address)>(&bob_address)->default_value("0.0.0.0"), "IP address of Bob")
      ("alice-port,x", po::value<decltype(alice_port)>(&alice_port)->default_value(18000), "Port of the Alice")
      ("bob-port,y", po::value<decltype(bob_port)>(&bob_port)->default_value(19000), "Port of the Bob")
      ("input-csv-file,i", po::value<decltype(input_path)>(&input_path)->default_value(""), "Path and name of input file")
      ("num-sample,n", po::value<decltype(n)>(&n)->default_value(10000), "Number of samples (for synthetic data)")
      ("num-feature,m", po::value<decltype(m)>(&m)->default_value(10), "Number of features (for synthetic data)")
      ("max-bin,B", po::value<decltype(max_bin)>(&max_bin)->default_value(8), "Max bin: 4,8,16")
      ("output-model-file,o", po::value<decltype(output_path)>(&output_path)->default_value(""), 
        "Path and name of output model file")
      ("num-thread,t", po::value<decltype(num_thread)>(&num_thread)->default_value(1), "Number of threads")
      ("depth,d", po::value<decltype(depth)>(&depth)->default_value(5), "Depth of a tree")
      ("tree-num,T", po::value<decltype(tree_num)>(&tree_num)->default_value(1), "Number of trees")
      ("lambda,l", po::value<decltype(lambda)>(&lambda)->default_value(0.001),
      "Lambda")
      ("log2-scale,s", po::value<decltype(log2_scale)>(&log2_scale)->default_value(20), "Log2 scale")
      ("positive-label,P", po::value<decltype(positive_label)>(&positive_label)->default_value(1.0), "Positive label")
      ("negative-label,N", po::value<decltype(negative_label)>(&negative_label)->default_value(0.0), "Negative label")
      ("have-label,H", po::value<decltype(have_label)>(&have_label)->default_value(false), "Have label?");
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argcp, argvp, allowed), vm);
    po::notify(vm);
  } catch (po::error const &e) {
    if (!vm.count("help")) {
      std::cout << e.what() << std::endl;
      std::cout << allowed << std::endl;
      return 1;
    }
  }

  if (vm.count("help")) {
    std::cout << allowed << "\n";
    return 1;
  }

  std::cout << "Role: " << role << ", input_file: " << input_path
            << ", output_model: " << output_path 
            << ", num_thread = " << num_thread << std::endl;
  vector<string> parties;
  for (int i = 0; i < num_thread; i++) {
    parties.emplace_back(alice_address + ":" + to_string(alice_port) + "," +
                         bob_address + ":" + to_string(bob_port));
    alice_port++;
    bob_port++;
  }

  std::vector<double> labels;
  vector<string> ids;
  vector<vector<uint8_t>> split_bools;
  vector<std::tuple<int, double, double>> split_values;

  if (input_path == "") {
    std::cout << "generate random input" << std::endl;
    SPU_ENFORCE(n > 0, "number of samples should larger than 0");
    SPU_ENFORCE(m > 0, "number of features should larger than 0");
    ids = createItems(n, role);
    if (role == 0) {
      labels = createLabels(n);
    }
    split_bools = createSplitBools(n, m*max_bin, max_bin);
  } else {
    auto split_info =
      read_file_and_gen_split_bools(input_path, bin_bound, labels, ids, max_bin, have_label, positive_label, negative_label);
    split_values = std::get<0>(split_info);
    split_bools = std::get<1>(split_info);
  }

  size_t cols = split_bools[0].size();

  printf("id_size = %zu, num_feature = %zu, bucket_num = %zu, depth = %zu, lambda = %f, tree_num = %zu \n",
         ids.size(), m, max_bin, depth, lambda, tree_num);

  depth = depth - 1;

  auto time_start = std::chrono::high_resolution_clock::now();

  vector<shared_ptr<AnonXGB>> hg_ptr(num_thread);
  auto ctxs = create_context(num_thread, parties, role);
  hg_ptr = initialize(ctxs, ids, labels, cols, log2_scale);

  vector<std::pair<vector<vector<size_t>>, vector<uint64_t>>> model;
  for (size_t t = 1; t <= tree_num; t++) {
    std::cout << "================= Training tree " << t << " ====================" << std::endl;
    auto current_model = gen_tree_parallel(hg_ptr, split_bools, lambda, depth, (t != 1) && (t == tree_num));
    model.push_back(current_model);
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
      time_end - time_start);
  auto time_duration = time_diff.count();
  size_t total_bytes = 0;
  for (int i = 0; i < num_thread; i++) {
    total_bytes += (hg_ptr[i]->get_link_context()->GetStats()->sent_bytes +
                    hg_ptr[i]->get_link_context()->GetStats()->recv_bytes);
  }

  cout << "================= Evaluation Result ====================" << endl;
  std::cout << "AnonGBDT training takes " << time_duration
            << " ms, com bytes = " << total_bytes << " B" << std::endl;
  cout << "========================================================" << endl;

  if (input_path != "" && output_path != "") {
    write_file(output_path, tree_num, depth, log2_scale, split_values, model);
    return 0;
  }
}
