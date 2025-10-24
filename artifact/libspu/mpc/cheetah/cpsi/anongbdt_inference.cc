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

void print_tree(vector<pair<vector<vector<tuple<int, double, double>>>,vector<uint64_t>>> &model,
                size_t &num_tree,
                size_t &depth,
                size_t &num_feature,
                size_t &log2_scale){
  printf("num_tree = %zu, depth = %zu, num_feature = %zu, log2_scale = %zu \n",
         num_tree, depth, num_feature, log2_scale);
  for (size_t t = 0; t < num_tree; t++) {
    std::cout << t << "-th tree:" << std::endl;
    std::vector<std::vector<std::tuple<int, double, double>>> &split_one_tree = model[t].first;
    std::vector<uint64_t> &weight = model[t].second;
    for (size_t d = 0; d < depth; d++) {
      std::cout << d << "-th level: ";
      for (int node_ind = 0; node_ind < (1 << d); node_ind++) {
        int split_feature = std::get<0>(split_one_tree[d][node_ind]);
        double split_bin_lower = std::get<1>(split_one_tree[d][node_ind]);
        double split_bin_upper = std::get<2>(split_one_tree[d][node_ind]);
        std::cout << split_feature << ", [" << split_bin_lower << ", " << split_bin_upper << "); ";
      }
      std::cout << std::endl;
    }
    for (int node_ind = 0; node_ind < (1 << depth); node_ind++){
      std::cout << weight[node_ind] << " ";
    }
    std::cout << std::endl;
  }

}


int main(int32_t argcp, char **argvp) {
  namespace po = boost::program_options;
  po::options_description allowed("Allowed options");
  std::string type;
  std::string alice_address, bob_address, input_path, model_path, output_file;
  int role, alice_port, bob_port, num_thread;
  size_t depth, num_tree, log2_scale, num_feature;
  bool have_label;
  std::vector<std::vector<double>> bin_bound;
  allowed.add_options()("help,h", "produce this message")
      ("role,r", po::value<decltype(role)>(&role)->required(),"Role of the node")
      ("alice-address,a", po::value<decltype(alice_address)>(&alice_address)->default_value("0.0.0.0"), 
        "IP address of the Alice")
      ("bob-address,b", po::value<decltype(bob_address)>(&bob_address)->default_value("0.0.0.0"), "IP address of the Bob")
      ("alice-port,x", po::value<decltype(alice_port)>(&alice_port)->default_value(18000), "Port of the Alice")
      ("bob-port,y", po::value<decltype(bob_port)>(&bob_port)->default_value(19000), "Port of the Bob")
      ("input-csv-file,i", po::value<decltype(input_path)>(&input_path)->required(), "Path and name of input file")
      ("model-file,m", po::value<decltype(model_path)>(&model_path)->required(), "Path and name of model file")
      ("num-thread,t", po::value<decltype(num_thread)>(&num_thread)->default_value(1), "Number of threads")
      ("log2-scale,s", po::value<decltype(log2_scale)>(&log2_scale)->default_value(20), "Log2 scale")
      ("output-file,o", po::value<decltype(output_file)>(&output_file)->default_value(""), "Outpu File")
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
            << ", model: " << model_path << std::endl;
  vector<string> parties;
  for (int i = 0; i < num_thread; i++) {
    parties.emplace_back(alice_address + ":" + to_string(alice_port) + "," +
                         bob_address + ":" + to_string(bob_port));
    alice_port++;
    bob_port++;
  }
  
  std::vector<std::pair<std::vector<std::vector<std::tuple<int, double, double>>>,std::vector<uint64_t>>> model;
  read_model(model_path, model, num_tree, depth, num_feature, log2_scale);
  print_tree(model, num_tree, depth, num_feature, log2_scale);
  std::vector<double> labels;
  std::vector<std::vector<double>> feats;
  vector<string> ids;
  read_data_file(input_path, labels, feats, ids, have_label);
  auto time_start = std::chrono::high_resolution_clock::now();

  auto psi_role = role==0?psi::PsiRoleType::Sender : psi::PsiRoleType::Receiver;
  vector<shared_ptr<AnonXGB>> hg_ptr(num_thread);
  auto ctxs = create_context(num_thread, parties, role);
  hg_ptr = initialize_inference(ctxs, ids, psi_role, num_feature, log2_scale);
  auto score = infer_parallel(hg_ptr, num_tree, depth, model, feats, false);

  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
      time_end - time_start);
  auto time_duration = time_diff.count();
  size_t total_bytes = 0;
  for (int i = 0; i < num_thread; i++) {
    total_bytes = hg_ptr[i]->get_link_context()->GetStats()->sent_bytes +
                  hg_ptr[i]->get_link_context()->GetStats()->recv_bytes;
  }

  std::cout << "AnonGBDT infer takes " << time_duration
            << " us, com bytes = " << total_bytes << std::endl;

  if (role==0) {
    write_score(output_file, score);
  }
}
