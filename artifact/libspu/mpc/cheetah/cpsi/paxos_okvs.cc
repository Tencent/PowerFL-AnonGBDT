
#include "libspu/mpc/cheetah/cpsi/paxos_okvs.h"

#include "yacl/crypto/rand/rand.h"

#include "psi/rr22/okvs/paxos.h"
//#include "openssl/rand.h"
//#include <unordered_set>

namespace spu::mpc::cheetah {
#define INVALID_NUM 0xffffffffffffffff

//    using namespace std;
//
//    Paxos::Paxos(size_t n, uint64_t seed):num_bins_(get_num_bins(n)) {
//        ext_cols_ = get_ext_cols(n);
//        if (ext_cols_ > 128) {
//            throw invalid_argument("input size is too large");
//        }
//    }
//
//    std::vector<size_t> Paxos::SimpleIndex(uint128_t item_hash) const {
//        SPU_ENFORCE(hash_num_ == 3, "only support 3-cuckoo hash");
//
//        psi::CuckooIndex::HashRoom hash_room(item_hash);
//
//        std::vector<size_t> hash_bin_idx;
//
//        size_t bin_idx0 = hash_room.GetHash(0);
//        hash_bin_idx.push_back(bin_idx0 % num_bins_);
//
//        size_t bin_idx1 = hash_room.GetHash(1);
//        size_t hash_bin_idx1 = bin_idx1 % num_bins_;
//        if (hash_bin_idx[0] != hash_bin_idx1) {
//            hash_bin_idx.push_back(hash_bin_idx1);
//        }
//
//        size_t bin_idx2 = hash_room.GetHash(2);
//        size_t hash_bin_idx2 = bin_idx2 % num_bins_;
//
//        for (size_t i = 0; i < hash_bin_idx.size(); i++) {
//            if (hash_bin_idx[i] == hash_bin_idx2) {
//                return hash_bin_idx;
//            }
//        }
//
//        hash_bin_idx.push_back(hash_bin_idx2);
//        return hash_bin_idx;
//    }
//
//    vector<uint64_t> Paxos::solve(const vector<pair<BitVector128, uint64_t>>&
//    inputs) const {
//        size_t rows = inputs.size();
//        size_t cols = ext_cols_;
//
//        vector<BitVector128> matrix(rows);
//        for (size_t i = 0; i < rows; i++) {
//            matrix[i] = inputs[i].first;
//        }
//
//        vector<uint64_t> values(rows);
//        for (size_t i = 0; i < rows; i++) {
//            values[i] = inputs[i].second;
//        }
//
//        size_t pivot = 0;
//        vector<size_t> deps_cols;
//
//        for (size_t j = 0; j < cols; j++) {
//            size_t rowJ = INVALID_NUM;
//            for (size_t i = pivot; i < rows; i++) {
//                if (locate(matrix[i], j) && rowJ == INVALID_NUM) {
//                    rowJ = i;
//                }
//            }
//
//            if (rowJ == INVALID_NUM) {
//                deps_cols.push_back(j);
//            } else {
//                if (rowJ != pivot) {
//                    swap(matrix[rowJ], matrix[pivot]);
//                    swap(values[rowJ], values[pivot]);
//                }
//
//                for (size_t i = 0; i < rows; i++) {
//                    if (i != pivot && locate(matrix[i], j)) {
//                        matrix[i] = matrix[i] ^ matrix[pivot];
//                        values[i] = values[i] ^ values[pivot];
//                    }
//                }
//
//                pivot = pivot + 1;
//            }
//        }
//
//        if (cols - rows != deps_cols.size()) {
//            throw runtime_error("bit matrix in garbled cuckoo table is not
//            full rank.");
//        }
//
//        vector<uint64_t> res(cols);
//        for (auto j : deps_cols) {
//            res[j] = yacl::crypto::SecureRandU64();
//
//            for (size_t i = 0; i < rows; i++) {
//                if (locate(matrix[i], j)) {
//                    values[i] = values[i] ^ res[j];
//                }
//            }
//        }
//
//        size_t pos = 0;
//        for (size_t j = 0; j < cols; j++) {
//            if (res[j] == 0) {
//                res[j] = values[pos];
//                pos = pos + 1;
//            }
//        }
//
//        return res;
//    }
//
//    void Paxos::cuckooHashTest(const std::vector<vector<size_t>>&
//    input_index_arr) {
//        vector<vector<size_t>> gc_table(num_bins_);
//
//        stack_buf_index_.clear();
//        stash_buf_index_.clear();
//
//        for (size_t i = 0; i < input_index_arr.size(); i++) {
//            for (auto j: input_index_arr[i]) {
//                gc_table[j].push_back(i);
//            }
//        }
//
//        vector<bool> visited(input_index_arr.size());
//
//        vector<size_t> queue_buf;
//        for (auto &elem : gc_table) {
//            if (elem.size() == 1) {
//                size_t i = elem[0];
//
//                if (!visited[i]) {
//                    queue_buf.push_back(i);
//                    visited[i] = true;
//                }
//            }
//        }
//
//        while (!queue_buf.empty()) {
//            for (auto i : queue_buf) {
//                for (auto j : input_index_arr[i]) {
//                    auto it = std::lower_bound(gc_table[j].begin(),
//                    gc_table[j].end(), i); if (it != gc_table[j].end()) {
//                        gc_table[j].erase(it);
//                    }
//                }
//
//                stack_buf_index_.push_back(i);
//            }
//
//            vector<size_t> tmp_buf;
//            for (auto i : queue_buf) {
//                for (auto j : input_index_arr[i]) {
//                    if (gc_table[j].size() == 1) {
//                        size_t i_star = gc_table[j][0];
//
//                        if (!visited[i_star]) {
//                            tmp_buf.push_back(i_star);
//                            visited[i_star] = true;
//                        }
//                    }
//                }
//            }
//
//            queue_buf = tmp_buf;
//        }
//
//        for (size_t i = 0; i < visited.size(); i++) {
//            if (!visited[i]) {
//                stash_buf_index_.push_back(i);
//            }
//        }
//    }
//
//    void Paxos::fill(const vector<pair<uint64_t, vector<size_t>>> &stack,
//    vector<uint64_t> &res) {
//        size_t top = stack.size();
//        while (top > 0) {
//            auto value = stack[top - 1].first;
//            auto idx_arr = stack[top-1].second;
//
//            size_t empty_slot = INVALID_NUM;
//
//            uint64_t share = 0;
//            for (auto j : idx_arr) {
//                if (res[j] == 0) {
//                    if (empty_slot == INVALID_NUM) {
//                        empty_slot = j;
//                    } else {
//                        res[j] = yacl::crypto::SecureRandU64();
//                        share = share ^ res[j];
//                    }
//                } else{
//                    share = share ^ res[j];
//                }
//            }
//
//            if (empty_slot == INVALID_NUM) {
//                throw runtime_error("collision happen in garbled cuckoo
//                table");
//            } else {
//                res[empty_slot] = share ^ value;
//            }
//
//            top = top - 1;
//        }
//
//        for (auto &vv : res) {
//            if (vv == 0) {
//                vv = yacl::crypto::SecureRandU64();
//            }
//        }
//    }
//
//    vector<uint64_t> Paxos::encode(const vector<pair<string, uint64_t>>
//    &inputs) {
//
////        auto time_start = chrono::high_resolution_clock::now();
//
//        vector<pair<uint128_t, BitVector128>> hash_inputs(inputs.size());
//
//        for (size_t i = 0; i < inputs.size(); i++) {
//            auto hash_bytes = yacl::crypto::Blake3(inputs[i].first);
//
//            uint128_t high = 0;
//            BitVector128 low = 0;
//            memcpy(&high, hash_bytes.data(), sizeof(uint128_t));
//            memcpy(&low, hash_bytes.data() + sizeof(uint128_t),
//            sizeof(BitVector128));
//
//            hash_inputs[i] = make_pair(high, low);
//        }
//
//        vector<vector<size_t>> input_index_arr(hash_inputs.size());
//        for (size_t i = 0; i < hash_inputs.size(); i++) {
//            input_index_arr[i] = SimpleIndex(hash_inputs[i].first);
//        }
//
//        cuckooHashTest(input_index_arr);
////        auto time_end = chrono::high_resolution_clock::now();
////        auto time_diff =
///chrono::duration_cast<chrono::microseconds>(time_end - time_start); / cout <<
///"cuckoo test [" << time_diff.count() << " microseconds]" << endl;
//
//        vector<uint64_t> res0(num_bins_, 0);
//        vector<uint64_t> res1(ext_cols_, 0);
//
//        if (stash_buf_index_.empty()) {
////            std::cout << "stash为空" << std::endl;
////            for (size_t j = 0; j < ext_cols_; j++) {
////                res1[j] = yacl::crypto::SecureRandU64();
////            }
//            auto buf = yacl::crypto::SecureRandBytes(res1.size() *
//            sizeof(uint64_t)); memcpy(reinterpret_cast<uint8_t
//            *>(res1.data()), buf.data(), buf.size());
//        } else {
////            std::cout << "stash不为空" << std::endl;
//
//            for (auto idx : stash_buf_index_) {
//                for (auto j : input_index_arr[idx]) {
//                    if (res0[j] == 0) {
//                        res0[j] = yacl::crypto::SecureRandU64();
//                    }
//                }
//            }
//
//            vector<pair<BitVector128, uint64_t>> stash_buf;
//            for (auto idx : stash_buf_index_) {
//                uint64_t value = inputs[idx].second;
//                for (auto j : input_index_arr[idx]) {
//                    value = value ^ res0[j];
//                }
//
//                pair<BitVector128, uint64_t> vv =
//                make_pair(hash_inputs[idx].second, value);
//                stash_buf.push_back(vv);
//            }
//
////            time_start = chrono::high_resolution_clock::now();
//            res1 = solve(stash_buf);
////            time_end = chrono::high_resolution_clock::now();
////            time_diff = chrono::duration_cast<chrono::microseconds>(time_end
///- time_start); /            cout << "solve linear equation [" <<
///time_diff.count() << " microseconds]" << endl;
//        }
//
////        time_start = chrono::high_resolution_clock::now();
//        vector<pair<uint64_t, vector<size_t>>> stack;
//        for (auto i : stack_buf_index_) {
//            auto bitvec = compute_bit_vec(hash_inputs[i].second);
//
//            uint64_t value = inputs[i].second;
//            for (size_t j = 0; j < ext_cols_; j++) {
//                if (locate(bitvec, j)) {
//                    value = value ^ res1[j];
//                }
//            }
//
//            pair<uint64_t, vector<size_t>> vv = make_pair(value,
//            input_index_arr[i]); stack.push_back(vv);
//        }
//
//        fill(stack, res0);
////        time_end = chrono::high_resolution_clock::now();
////        time_diff = chrono::duration_cast<chrono::microseconds>(time_end -
///time_start); /        cout << "fill [" << time_diff.count() << "
///microseconds]" << endl;
//
//        vector<uint64_t> res(num_bins_ + ext_cols_);
//        copy_n(res0.data(), res0.size(), res.data());
//        copy_n(res1.data(), res1.size(), res.data() + num_bins_);
//
//        return res;
//    }
//
//    uint64_t Paxos::decode(const vector<uint64_t>& gc_table, const string
//    &item) {
//        if (gc_table.empty()) {
//            throw invalid_argument("empty garbled cuckoo table");
//        }
//
//        if (gc_table.size() != (num_bins_ + ext_cols_)) {
//            throw invalid_argument("invalid length of garbled cuckoo table");
//        }
//
//        auto hash_bytes = yacl::crypto::Blake3(item);
//        uint128_t high = 0;
//        BitVector128 low = 0;
//        memcpy(&high, hash_bytes.data(), sizeof(uint128_t));
//        memcpy(&low, hash_bytes.data() + sizeof(uint128_t),
//        sizeof(BitVector128));
//
//        auto index_vec = SimpleIndex(high);
//        auto bit_vec = compute_bit_vec(low);
//
//        uint64_t recovered = 0;
//        for (auto j: index_vec) {
//            recovered = recovered ^ gc_table[j];
//        }
//        for (size_t j = 0; j < ext_cols_; j++) {
//            if (locate(bit_vec, j)) {
//                recovered = recovered ^ gc_table[j + num_bins_];
//            }
//        }
//
//        return recovered;
//    }
//
//    vector<uint64_t> Paxos::decode(const vector<uint64_t> &gc_table, const
//    vector<string> &items) {
//        vector<uint64_t> res(items.size());
//        for (size_t i = 0; i < items.size(); i++) {
//            res[i] = decode(gc_table, items[i]);
//        }
//
//        return res;
//    }

PaxosOKVS::PaxosOKVS(size_t n, uint64_t seed) {
  paxos_ptr_ = std::make_shared<psi::rr22::okvs::Paxos<uint64_t>>();
  paxos_ptr_->Init(n, w_, s_, psi::rr22::okvs::PaxosParam::DenseType::Binary,
                   yacl::MakeUint128(0, seed));
}

std::vector<uint64_t> PaxosOKVS::encode(
    const std::vector<std::pair<std::string, uint64_t>> &inputs) {
  std::vector<uint128_t> keys(inputs.size());
  std::vector<uint128_t> values(inputs.size());

  for (size_t i = 0; i < inputs.size(); i++) {
    keys[i] = yacl::crypto::Blake3_128(inputs[i].first);
    values[i] = inputs[i].second;
  }

  std::vector<uint128_t> gc_table(paxos_ptr_->size());

  paxos_ptr_->SetInput(absl::MakeSpan(keys));
  paxos_ptr_->Encode(absl::MakeSpan(values), absl::MakeSpan(gc_table));

  std::vector<uint64_t> ret(paxos_ptr_->size());
  std::transform(gc_table.begin(), gc_table.end(), ret.begin(),
                 [&](uint128_t vv) { return vv; });

  return ret;
}

std::vector<uint64_t> PaxosOKVS::decode(const std::vector<uint64_t> &gc_table,
                                        const std::vector<std::string> &items) {
  std::vector<uint128_t> keys(items.size());
  for (size_t i = 0; i < items.size(); i++) {
    keys[i] = yacl::crypto::Blake3_128(items[i]);
  }

  SPU_ENFORCE(gc_table.size() == paxos_ptr_->size());
  std::vector<uint128_t> table(gc_table.size());
  std::transform(gc_table.begin(), gc_table.end(), table.begin(),
                 [&](uint64_t vv) { return vv; });

  std::vector<uint128_t> values(items.size());
  paxos_ptr_->Decode(absl::MakeSpan(keys), absl::MakeSpan(values),
                     absl::MakeSpan(table));

  std::vector<uint64_t> ret(values.size());
  std::transform(values.begin(), values.end(), ret.begin(),
                 [&](uint128_t vv) { return vv; });

  return ret;
}
}  // namespace spu::mpc::cheetah