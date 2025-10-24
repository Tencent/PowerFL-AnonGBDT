

#ifndef SPU_PAXOS_OKVS_H
#define SPU_PAXOS_OKVS_H

#pragma once

#include <vector>

#include "yacl/crypto/hash/hash_utils.h"
#include "yacl/utils/parallel.h"

#include "libspu/core/prelude.h"

#include "psi/rr22/okvs/paxos.h"
#include "psi/utils/cuckoo_index.h"

namespace libpsi = psi;
namespace spu::mpc::cheetah {

//    using BitVector128 = uint128_t;
//
//    class Paxos {
//        size_t num_bins_;
//        size_t ext_cols_;
//        const size_t hash_num_ = 3;
//        const size_t lambda_ = 40;
//        std::vector<size_t> stash_buf_index_;
//        std::vector<size_t> stack_buf_index_;
//
//        bool locate(BitVector128 vv, size_t j) const {
//            return (vv >> j) & 0x1;
//        };
//
//    public:
//        explicit Paxos(size_t n, uint64_t seed=0);
//
//        size_t get_num_bins(size_t n) const {
//            return n < 1500 ? (n << 1) : static_cast<int>(ceil(1.3 * n));
//        }
//
//        size_t get_ext_cols(size_t n) const {
//            auto logn = static_cast<size_t>(std::ceil(std::log2(n)));
//            return logn + lambda_;
//        }
//
//        size_t length() const {
//            return num_bins_ + ext_cols_;
//        }
//
//        BitVector128 compute_bit_vec(BitVector128 item_hash) const {
//            return item_hash & ((1ULL << ext_cols_) - 1);
//        }
//
//        std::vector<size_t> SimpleIndex(uint128_t item_hash) const;
//
//        std::vector<uint64_t> solve(const std::vector<std::pair<BitVector128,
//        uint64_t>>& inputs) const;
//
//        void cuckooHashTest(const std::vector<std::vector<size_t>>&
//        input_index_arr);
//
//        void fill(const std::vector<std::pair<uint64_t, std::vector<size_t>>>
//        &stack, std::vector<uint64_t> &res);
//
//        std::vector<uint64_t> encode(const std::vector<std::pair<std::string,
//        uint64_t>> &inputs);
//
//        uint64_t decode(const std::vector<uint64_t>& gc_table, const
//        std::string &item);
//
//        std::vector<uint64_t> decode(const std::vector<uint64_t> &gc_table,
//        const std::vector<std::string> &items);
//    };

class PaxosOKVS {
  public:
  explicit PaxosOKVS(size_t n, uint64_t seed = 0);

  std::vector<uint64_t> encode(
      const std::vector<std::pair<std::string, uint64_t>> &inputs);

  std::vector<uint64_t> decode(const std::vector<uint64_t> &gc_table,
                               const std::vector<std::string> &items);

  size_t length() const { return paxos_ptr_->size(); }

  private:
  size_t w_ = 3;
  size_t s_ = 40;

  std::shared_ptr<psi::rr22::okvs::Paxos<uint64_t>> paxos_ptr_;
};
};  // namespace spu::mpc::cheetah

#endif  // SPU_PAXOS_OKVS_H
