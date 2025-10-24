#include <cstddef>
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


using namespace std;
using namespace spu::mpc::cheetah;

inline bool 
read_lines_to_strings(fstream &fin, vector<vector<string>> &records, char sep) {
  string line;
  while (getline(fin, line)) {
    vector<string> row;

    std::regex reg(",+");
    std::sregex_token_iterator iter(line.begin(), line.end(), reg, -1);
    std::sregex_token_iterator end;

    while (iter != end) {
      row.push_back(*iter);
      iter++;
    }

    records.push_back(row);
  }
  return true;
}

// from squirrel
inline std::vector<double> binning(const vector<double> &x, size_t max_bin) {
  SPU_ENFORCE(not x.empty(), "can not quantize empty input");

  const size_t nsamples = x.size();
  std::vector<double> sorted_x(x.data(), x.data() + nsamples);
  std::sort(sorted_x.begin(), sorted_x.end());

  std::vector<double> bin_thresholds;
  std::vector<double> value_category;
  max_bin = max_bin - 1;
  bin_thresholds.push_back(-std::numeric_limits<double>::infinity());
  size_t remain_samples = nsamples;
  size_t expected_idx = (remain_samples + max_bin - 1) / max_bin;
  bool fast_skip = false;
  double last_value = std::numeric_limits<double>::max();
  size_t idx = 0;
  while (idx < nsamples) {
    double v = sorted_x[idx];
    bool is_diff = !(std::abs(v - last_value) < 1e-10);

    if (!fast_skip && is_diff) {
      if (value_category.size() <= max_bin) {
        value_category.push_back(v);
      } else {
        fast_skip = true;
      }
      last_value = v;
    }

    if (idx >= expected_idx && is_diff) {
      bin_thresholds.push_back(v);
      size_t nn = bin_thresholds.size();
      if (nn + 1 == max_bin) {
        break;
      }
      remain_samples = nsamples - idx;
      size_t expected_bin_count = (remain_samples + max_bin - nn - 1) / (max_bin - nn);
      expected_idx = idx + expected_bin_count;
      last_value = v;
    }

    if (!fast_skip || idx >= expected_idx) {
      idx += 1;
    } else {
      idx = expected_idx;
    }
  }

  // if (value_category.size() <= max_bin) {
  //   // Use category as split point.
  //   // NOTE: bin_thresholds = value_category[1:]
  //   bin_thresholds =
  //       std::vector<double>(value_category.begin() + 1, value_category.end());
  // }

  // add `infty` for test samples that might larger than the train samples
  bin_thresholds.push_back(std::numeric_limits<double>::infinity());

  SPU_ENFORCE(bin_thresholds.size() <= max_bin);

  return bin_thresholds;
}


uint8_t get_split_bool(double feat, double split_value) {
  if (feat < (split_value - 0.000001)) {
    return 1;
  } else {
    return 0;
  }
}

inline void gen_split_bools(
  std::vector<std::vector<double>> &bin_bound, 
  vector<vector<double>> &feats, 
  vector<vector<uint8_t>> &split_bools, 
  vector<tuple<int, double, double>> &split_values
) {
  size_t rows = feats[0].size();
  size_t cols = feats.size();

  auto selected_bin_bound = bin_bound;
  size_t split_size = 0;
  for (size_t j = 0; j < cols; j++) {
    split_size += selected_bin_bound[j].size() - 1;
    for (size_t k = 0; k < selected_bin_bound[j].size() - 1; k++) {
      auto current_bin_value = make_tuple(j, selected_bin_bound[j][k], selected_bin_bound[j][k+1]);
      // std::cout << j << ", (" << selected_bin_bound[j][k] << ", " << selected_bin_bound[j][k+1] << "); ";
      split_values.emplace_back(std::move(current_bin_value));
    }
    // std::cout << std::endl;
  }

  
  for (size_t i = 0; i < rows; i++) {
    split_bools[i].assign(split_size, 0);
  }
  
  size_t bin_bgn_ind = 0, bin_end_ind = 0;
  for (size_t j = 0; j < cols; j++) {
    if (j != 0) 
      bin_bgn_ind += selected_bin_bound[j-1].size() - 1;
    bin_end_ind += (selected_bin_bound[j].size() - 1);
    for (size_t i = 0; i < rows; i++) {
      for (size_t k = bin_bgn_ind; k < bin_end_ind; k++) {
        split_bools[i][k] = get_split_bool(feats[j][i], get<2>(split_values[k]));
      }
    }
  }
  // for (size_t i = 0; i < std::min((size_t)10, rows); i++){
  //   int pre_feature_ind = 0;
  //   for(size_t j = 0; j < split_size; j++) {
  //     if (get<0>(split_values[j]) != pre_feature_ind) {
  //       std::cout << "| ";
  //       pre_feature_ind++;
  //     }
  //     std::cout << (int)split_bools[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
}


void handle_label_and_ids(
  vector<vector<string>> &records,
  std::vector<double> &labels,
  std::vector<string> &ids, 
  bool have_label, 
  double positive_label,
  double negative_label
) {
  for (size_t i = 0; i < records.size(); i++) {
    vector<double> rows;
    std::string id = records[i][0];

    if (have_label) {
      double label = std::stod(records[i][1]);
      if (abs(label - negative_label)<0.000001) {
        labels.push_back(0);
      } else {
        if (abs(label - positive_label)<0.000001) {
          labels.push_back(1);
        } else {
          throw logic_error("label is wrong");
        }
      }
    }
    ids.push_back(id);
  }
}

inline pair<vector<tuple<int,double,double>>, vector<vector<uint8_t>>>
read_file_and_gen_split_bools(const std::string &path,
                              std::vector<std::vector<double>> &bin_bound,
                              std::vector<double> &labels,
                              std::vector<string> &ids, size_t max_bin,
                              bool have_label=false, double positive_label=1,
                              double negative_label=0) {
  fstream fin;
  fin.open(path.c_str(), ios::in);
  if (!fin.is_open()) {
    SPU_THROW("open file " + path + " failed");
  }

  vector<vector<string>> records;
  string line;
  if (!read_lines_to_strings(fin, records, ',')) {
    for (auto &record : records) {
      record.clear();
      vector<string>().swap(record);
    }
    records.clear();
    vector<vector<string>>().swap(records);
  };
  fin.close();

  vector<vector<double>> feats;
  handle_label_and_ids(records, labels, ids, have_label, positive_label, negative_label);

  for (size_t i = have_label?2:1; i < records[0].size(); i++) {
    vector<double> records_one_col;
    for (size_t j = 0; j < records.size(); j++) {
      records_one_col.push_back(std::stod(records[j][i]));
    }
    std::vector<double> bin_bound_one_feature = binning(records_one_col, max_bin);
    bin_bound.push_back(bin_bound_one_feature);
    feats.push_back(records_one_col);
  }

  size_t rows = feats[0].size();
  size_t cols = feats.size();

  std::cout << "#samples: " << rows << ", #features: " << cols << std::endl;
  if (have_label) {
    std::cout << "positive label: " << positive_label << ", negative label: " << negative_label << std::endl;
  } 
  std::cout << "bin bounds with max_bin = " << max_bin << std::endl; 
  
  vector<tuple<int, double, double>> split_values;
  vector<vector<uint8_t>> split_bools(rows);
  gen_split_bools(bin_bound, feats, split_bools, split_values);
  return make_pair(split_values, split_bools);
}

inline void write_file(const string &path, const int &tree_num, 
                const int &depth, const int &log2_scale,
                const std::vector<std::tuple<int, double, double>> &split_values,
                const std::vector<std::pair<std::vector<std::vector<size_t>>,
                                std::vector<uint64_t>>> &res) {
  ofstream fout;
  fout.open(path.c_str(), ios::out);
  if (!fout.is_open()) {
    SPU_THROW("open file " + path + " failed");
  }
  int num_feature = std::get<0>(split_values[split_values.size()-1]) + 1;
  fout <<  tree_num << " " << depth+1 << " " <<  num_feature << " " << log2_scale << std::endl;
  for (int t = 0; t < tree_num; t++) {
    for (size_t i = 0; i < res[t].first.size(); i++) {
      for (size_t j = 0; j < res[t].first[i].size(); j++){
        auto pos = static_cast<int>(res[t].first[i][j]);
        if (pos >= 0) {
          fout << std::get<0>(split_values[pos]) << ", [" << std::get<1>(split_values[pos]) << ", "
              << std::get<2>(split_values[pos]) << ")";
        } else {
          fout << "*, *";
        }
        if (j < res[t].first[i].size()-1) {
          fout << "; ";
        }
      }
      fout << std::endl;
    }
    for (auto wt : res[t].second) {
      fout << wt << " ";
    }
    fout << std::endl;
  }

  fout.close();
}


const uint64_t SEED = 0x123456789ABCDEF0;

inline std::vector<std::string> createItems(size_t n, int role) {
  vector<std::string> ret(n);
  for (size_t i = 0; i < n; ++i) {
    ret[i] = std::to_string(i*(role+1));
  }

  return ret;
}

std::vector<double> createLabels(size_t n) {
  std::mt19937_64 rng(SEED);
  std::uniform_int_distribution<int> dist(0, 1);

  vector<double> ret(n);
  for (size_t i = 0; i < n; ++i) {
    ret[i] = dist(rng);
  }

  return ret;
}

inline vector<vector<uint8_t>> createSplitBools(size_t rows, size_t cols,
                                      size_t iv_splits_num) {
  SPU_ENFORCE(rows >= iv_splits_num && (cols % iv_splits_num) == 0,
              "{} >= {}, {} % {} == 0", rows, iv_splits_num, cols,
              iv_splits_num);

  vector<uint64_t> pos(rows);
  for (size_t i = 0; i < iv_splits_num; i++) {
    pos[i] = i;
  }

  for (size_t i = iv_splits_num; i < rows; i++) {
    uint64_t idx = yacl::crypto::SecureRandU64() % iv_splits_num;
    pos[i] = pos[idx];
  }

  vector<vector<uint8_t>> split_bools(rows);
  for (auto &bools : split_bools) {
    bools.assign(cols, 0);
  }

  for (size_t cnt = 0; cnt < (cols / iv_splits_num); cnt++) {
    // random permutate pos
    std::shuffle(pos.begin(), pos.end(), std::default_random_engine());

    for (size_t i = 0; i < rows; i++) {
      split_bools[i][cnt * iv_splits_num + pos[i]] = 1;
    }
  }

  for (size_t j = 0; j < cols; j++) {
    uint64_t cnt = 0;
    for (size_t i = 0; i < rows; i++) {
      if (split_bools[i][j]) {
        cnt = cnt + 1;
      }
    }
    SPU_ENFORCE(cnt != 0);
  }

  return split_bools;
}

std::vector<std::string> split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}



void read_data_file(const std::string &path,
                    std::vector<double> &labels,
                    std::vector<std::vector<double>> &feats,
                    std::vector<string> &ids,
                    bool have_label=false) {
  fstream fin;
  fin.open(path.c_str(), ios::in);
  if (!fin.is_open()) {
    SPU_THROW("open file " + path + " failed");
  }

  vector<vector<string>> records;
  string line;
  if (!read_lines_to_strings(fin, records, ',')) {
    for (auto &record : records) {
      record.clear();
      vector<string>().swap(record);
    }
    records.clear();
    vector<vector<string>>().swap(records);
  };
  fin.close();

  for (size_t i = 0; i < records.size(); i++) {
    vector<double> rows;
    std::string id = records[i][0];

    if (have_label) {
      labels.push_back(std::stod(records[i][1]));
    }
    ids.push_back(id);
    vector<double> records_one_sample;
    for (size_t j = have_label?2:1; j < records[0].size(); j++) {
      records_one_sample.emplace_back(std::stod(records[i][j]));
    }
    feats.push_back(records_one_sample);
  }
  std::cout << "#samples: " << feats.size() << ", #features: " << feats[0].size() << std::endl;
}


void read_model(std::string &model_path,
                std::vector<std::pair<std::vector<std::vector<std::tuple<int, double, double>>>,std::vector<uint64_t>>> &model,
                size_t &num_tree,
                size_t &depth,
                size_t &num_feature,
                size_t &log2_scale) {
  std::ifstream infile(model_path);
  std::string line;
  {
    std::getline(infile, line);
    std::istringstream iss(line);
    iss >> num_tree >> depth >> num_feature >> log2_scale;
  }
  depth--;
  for (size_t t = 0; t < num_tree; t++) {
    std::vector<std::vector<std::tuple<int, double, double>>> split_one_tree;
    std::vector<uint64_t> weight((1 << depth));
    for (size_t d = 0; d < depth; d++) {
      std::vector<std::tuple<int, double, double>> split_one_level;
      std::getline(infile, line);
      line.erase(
        std::remove_if(
            line.begin(), line.end(),
            [] (char c) { return (c == '[' || c == ')' || c == ' '); }
        ),
        line.end()
      );
      auto splits_str = split(line, ";");
      for (int node_ind = 0; node_ind < (1 << d); node_ind++) {
        auto split_str = split(splits_str[node_ind],",");
        int split_feature=-1;
        double split_bin_lower = -1;
        double split_bin_upper = -1;
        if (!(split_str[0] == "*")) {
          split_feature = std::stoi(split_str[0]);
          split_bin_lower = std::stod(split_str[1]);
          split_bin_upper = std::stod(split_str[2]);
        }
        split_one_level.emplace_back(
          make_tuple(split_feature, split_bin_lower, split_bin_upper)
        );
      }
      split_one_tree.emplace_back(split_one_level);
    }
    std::getline(infile, line);
    std::istringstream iss(line);
    for (int node_ind = 0; node_ind < (1 << depth); node_ind++){
      iss >> weight[node_ind];
    }
    model.emplace_back(make_pair(split_one_tree, weight));
  }
}

void write_score(const string &path, const std::vector<double> &score) {
  ofstream fout;
  fout.open(path.c_str(), ios::out);
  if (!fout.is_open()) {
    SPU_THROW("open file " + path + " failed");
  }
  for (size_t i = 0; i < score.size(); i++) {
    fout << score[i] << endl;
  }
  fout.close();
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
