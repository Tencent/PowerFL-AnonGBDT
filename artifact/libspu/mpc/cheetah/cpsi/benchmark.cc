#include <omp.h>

#include <boost/program_options.hpp>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <regex>
#include <string>
#include <map>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/link/context.h"
#include "yacl/link/test_util.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/cpsi/anon_xgb.h"

using namespace std;
using namespace spu::mpc::cheetah;
using namespace spu;
using namespace seal;

const uint64_t SEED = 0x123456789ABCDEF0;

std::vector<std::string> createItems(size_t n) {
  std::mt19937_64 rng(SEED);
  std::uniform_int_distribution<uint64_t> dist;

  vector<std::string> ret(n);
  for (size_t i = 0; i < n; ++i) {
    ret[i] = std::to_string(dist(rng)) + std::to_string(i*2);
  }

  return ret;
}

vector<uint64_t> genArithShares(size_t n, double scale, int role) {
  vector<uint64_t> res(n);

  std::mt19937_64 rng(SEED);
  std::uniform_int_distribution<uint64_t> dist;

  for (size_t i = 0; i < n; i++) {
    double val = 1.0 + i / (n + 0.0);
    auto vv = static_cast<int64_t>(std::round(val * scale));
    if (role == 0) {
        res[i] = dist(rng);
    } else {
        res[i] = static_cast<uint64_t>(vv) - dist(rng);
    }
  }

  return res;
}

vector<uint8_t> genBoolShares1(size_t n, int role) {
  vector<uint8_t> res(n);

  std::mt19937_64 rng(SEED);
  std::uniform_int_distribution<uint8_t> dist;
  
  for (size_t i = 0; i < n; i++) {
    uint64_t val = (i + 1) & 0x1;
    if (role == 0) {
        res[i] = dist(rng) & 0x1;
    } else {
        res[i] = (val ^ dist(rng)) & 0x1;
    }
  }

  return res;
}

vector<double> genDouble(size_t n) {
  vector<double> res(n);

  std::mt19937_64 rng(SEED);
  std::uniform_real_distribution<double> dist;

  for (size_t i = 0; i < n; i++) {
    res[i] = dist(rng);
  }

  return res;
}

vector<vector<int>> genIndexVec(size_t n, size_t m) {
  vector<vector<int>> res;
  res.reserve(m);
  SPU_ENFORCE(m < n, "m should be less than n");

  std::mt19937_64 rng(SEED);
  std::uniform_int_distribution<int> dist;

  size_t eachLen = std::min(m, (n) / 8);

  vector<int> sequence(n);
  std::iota(sequence.begin(), sequence.end(), 0);

  for (size_t i = 0; i < m; i++) {
    vector<int> sample;
    sample.reserve(eachLen);
    std::sample(sequence.begin(), sequence.end(), std::back_inserter(sample), eachLen, rng);
    res.push_back(std::move(sample));
  }

  return res;
}

vector<std::shared_ptr<yacl::link::Context>> create_context(
    size_t num_thread, const vector<std::string> &parties, int my_rank) {
  SPU_ENFORCE(num_thread <= parties.size());
  vector<std::shared_ptr<yacl::link::Context>> ctxs(num_thread);
  for (size_t tid = 0; tid < num_thread; tid++) {
    yacl::link::ContextDesc lctx_desc;
    std::vector<std::string> hosts = absl::StrSplit(parties[tid], ',');
    for (size_t i = 0; i < hosts.size(); i++) {
      const auto id =
          "party" + std::to_string(i) + "-tid-" + std::to_string(tid);
      lctx_desc.parties.push_back({id, hosts[i]});
    }
    ctxs[tid] = yacl::link::FactoryBrpc().CreateContext(lctx_desc, my_rank);
    ctxs[tid]->ConnectToMesh();
    ctxs[tid]->SetRecvTimeout(3UL * 60 * 1000);
  }
  return ctxs;
}

void fastpacklwes(size_t n, size_t m){
  
    seal::EncryptionParameters parms(seal::scheme_type::ckks);

    uint64_t lwe_coeff_count = 4096;
    uint64_t coeff_count = 8192;

    parms.set_poly_modulus_degree(lwe_coeff_count);
    parms.set_coeff_modulus(
        seal::CoeffModulus::Create(lwe_coeff_count, {60, 49}));
    seal::SEALContext context0(parms);

    parms.set_poly_modulus_degree(coeff_count);
    parms.set_coeff_modulus(
        seal::CoeffModulus::Create(coeff_count, {60, 49, 60}));
    seal::SEALContext context1(parms);

    auto ckks_ptr0 = make_shared<CKKSParams>(context0, false, false);
    auto ckks_ptr1 = make_shared<CKKSParams>(context1, false, true);

    vector<uint64_t> lwe_sk;
    seal::KSwitchKeys switch_keys;
    ckks_ptr0->get_secret_key(lwe_sk);
    ckks_ptr1->gen_switch_keys(lwe_sk, switch_keys);

    double scale = std::pow(2, 35);

    auto vals = genDouble(n);
    vector<vector<int>> index_vec = genIndexVec(n, m);

    vector<Ciphertext> enc_vec;
    for (size_t i = 0; i < n; i += lwe_coeff_count) {
      vector<double> vec(lwe_coeff_count, 0.0);

      std::copy_n(vals.data() + i, 
      min(size_t(lwe_coeff_count), n - i), 
      vec.data());

      Ciphertext enc_tmp;
      ckks_ptr0->encrypt_non_ntt(vec, enc_tmp, scale);
      enc_vec.push_back(enc_tmp);
    }

    vector<vector<uint64_t>> coeffs;
    vector<uint64_t> values;
    ckks_ptr0->sum(enc_vec, index_vec, coeffs, values);
    uint64_t lwe_mod = ckks_ptr0->get_coeff_modulus(0);

    auto time_start = chrono::high_resolution_clock::now();
    Ciphertext enc;
    ckks_ptr1->lwes_to_rlwe(coeffs, values, lwe_mod, scale, switch_keys, enc);

    
    auto time_end = chrono::high_resolution_clock::now();
    auto time_diff =
      chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
    cout << "================= Evaluation Result ====================" << endl;
    cout << "fastpacklwes done [" << time_diff.count() << " milliseconds]" << endl;
    cout << "========================================================" << endl;
}

void bois(size_t n, size_t m, vector<std::shared_ptr<yacl::link::Context>> &ctxs, int role) {
  
    std::vector<std::string> items = createItems(n);
    std::shared_ptr<VoleCPsi> cpsi_ptr = nullptr;
    std::vector<uint8_t> bools;
    if (role == 0) {
      cpsi_ptr = make_shared<VoleCPsi>(ctxs[0], psi::PsiRoleType::Sender);
      bools = cpsi_ptr->RunCPsi(items);
    } else {
      cpsi_ptr = make_shared<VoleCPsi>(ctxs[0], psi::PsiRoleType::Receiver);
      bools = cpsi_ptr->RunCPsi(items);
    }

    size_t bgn_bytes = cpsi_ptr->get_link_context()->GetStats()->sent_bytes +
                  cpsi_ptr->get_link_context()->GetStats()->recv_bytes;
    auto time_start = std::chrono::high_resolution_clock::now();
    if (role == 0) {
      vector<uint64_t> values(n);
      for (size_t i = 0; i < n; i++) {
        values[i] = i;
      }
      auto shares = cpsi_ptr->OpprfShareSender(values, true);
    } else {
      auto shares = cpsi_ptr->OpprfShareReceiver(true);
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
        time_end - time_start);
    size_t end_bytes = cpsi_ptr->get_link_context()->GetStats()->sent_bytes +
              cpsi_ptr->get_link_context()->GetStats()->recv_bytes;
    cout << "================= Evaluation Result ====================" << endl;
    std::cout << " bois done [" << time_diff.count() << " milliseconds]"
              << " [" << (end_bytes - bgn_bytes) << " B]" << std::endl;
    cout << "========================================================" << endl;
}

int main(int32_t argcp, char **argvp) {
  namespace po = boost::program_options;
  po::options_description allowed("Allowed options");
  std::string op;
  std::string alice_address, bob_address;
  int role, alice_port, bob_port, num_thread;
  size_t log2_scale, n, m;

  allowed.add_options()("help,h", "produce this message")
      ("role,r", po::value<decltype(role)>(&role)->default_value(0),"Role of the node")
      ("alice-address,a", po::value<decltype(alice_address)>(&alice_address)->default_value("0.0.0.0"), 
        "IP address of the Alice")
      ("bob-address,b", po::value<decltype(bob_address)>(&bob_address)->default_value("0.0.0.0"), "IP address of the Bob")
      ("alice-port,x", po::value<decltype(alice_port)>(&alice_port)->default_value(18000), "Port of the Alice")
      ("bob-port,y", po::value<decltype(bob_port)>(&bob_port)->default_value(19000), "Port of the Bob")
      ("cryptographic-primitive,p", po::value<decltype(op)>(&op)->required(), 
        "Cryptographic primitive, including argmax, greater, mux, bois, binmatvec, sigmoid, fastpacklwes")
      ("log2-scale,s", po::value<decltype(log2_scale)>(&log2_scale)->default_value(20), "Log2 scale")
      ("num-sample,n", po::value<decltype(n)>(&n)->default_value(100000), "Number of samples")
      ("num-feature,m", po::value<decltype(m)>(&m)->default_value(10), "Number of features");
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

  num_thread = 1;

  std::cout << "Role: " << role << ", num_thread = " << num_thread << std::endl;
  vector<string> parties;
  for (int i = 0; i < num_thread; i++) {
    parties.emplace_back(alice_address + ":" + to_string(alice_port) + "," +
                         bob_address + ":" + to_string(bob_port));
    alice_port++;
    bob_port++;
  }

  double scale = std::pow(2, log2_scale);

  printf("n = %zu, m = %zu, \n", n, m);

  if (op == "fastpacklwes") {
    fastpacklwes(n, m);
    return 0;
  }

  auto ctxs = create_context(num_thread, parties, role);

  if (op == "bois") {
    bois(n, m, ctxs, role);
    return 0;
  }

  vector<shared_ptr<AnonXGB>> hg_ptr(num_thread);
  for (int i = 0; i < num_thread; i++) {
    if (role == 0) {
      hg_ptr[i] = std::make_shared<AnonXGB>(ctxs[i], psi::PsiRoleType::Sender, log2_scale);
    } else {
      hg_ptr[i] = std::make_shared<AnonXGB>(ctxs[i], psi::PsiRoleType::Receiver, log2_scale);  
    }
  }

  vector<uint64_t> ashr = genArithShares(n, scale, role);
  vector<uint64_t> bshr = genArithShares(n, scale, role);
  vector<uint8_t> mshr = genBoolShares1(n, role);

  auto time_start = std::chrono::high_resolution_clock::now();

  if (op == "argmax") {
    auto res = hg_ptr[0]->arg_max(ashr, m);
  } else if (op == "mux") {
    auto res = hg_ptr[0]->mux(mshr, ashr, bshr);
  } else if (op == "greater") {
    vector<uint64_t> diff(n);
    for (size_t i = 0; i < n; i++) {
      diff[i] = ashr[i] - bshr[i];
    }
    auto res = hg_ptr[0]->msb(diff);
  } else if (op == "sigmoid") {
    auto res = hg_ptr[0]->sigmoid(ashr, 2);
  } else if (op == "binmatvec") {
    vector<vector<uint8_t>> splitBools(n);
    for (size_t i = 0; i < n; i++) {
      splitBools[i].resize(m);
      for (size_t j = 0; j < m; j++) {
          splitBools[i][j] = (i ^ j) & 0x1;
      }
    }
    if (role == 0) {
      auto com_ptr = make_shared<HistoGram>(ctxs[0], psi::PsiRoleType::Sender);
      com_ptr->initialize(true);
      com_ptr->ComputeHistoGram(ashr, bshr, scale, splitBools);
    } else {
      auto com_ptr = make_shared<HistoGram>(ctxs[0], psi::PsiRoleType::Receiver);
      com_ptr->initialize(true);
      com_ptr->ComputeHistoGram(ashr, bshr, scale, splitBools);
    }
  } else {
    std::cout << "Invalid op" << std::endl;
    return 1;
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
      time_end - time_start);
  auto time_duration = time_diff.count();
  size_t total_bytes = 0;
  for (int i = 0; i < num_thread; i++) {
    total_bytes = hg_ptr[i]->get_link_context()->GetStats()->sent_bytes +
                  hg_ptr[i]->get_link_context()->GetStats()->recv_bytes;
  }
  
  cout << "================= Evaluation Result ====================" << endl;
  std::cout << op << " done [" << time_duration
            << " milliseconds], [" << total_bytes << " B]" << std::endl;
  cout << "========================================================" << endl;
}
