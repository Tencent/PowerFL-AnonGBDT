
#include "libspu/mpc/cheetah/cpsi/vole_cpsi.h"

#include <future>

#include "absl/container/flat_hash_set.h"
#include "openssl/rand.h"
#include "yacl/crypto/hash/hash_utils.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/utils/parallel.h"

#include "libspu/mpc/cheetah/cpsi/cuckoo_hash.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

#include "psi/utils/serialize.h"

#include "psi/utils/serializable.pb.h"

namespace spu::mpc::cheetah {
using namespace std;

namespace {

constexpr size_t kDefaultHashNum = 3;
constexpr double scaleFactor = 1.3;

constexpr size_t kVoleCPsiBatchSize = 4096;
constexpr size_t kVoleCPsiLogBatchSize = kVoleCPsiBatchSize * 100;

//        constexpr size_t kMaxCompareBytes = 13;

// BC22 section 2.3
// H(i, vi) = H(i, wi-delta*ui)
inline uint64_t BaRKOPRFHash(size_t bin_idx,
                             const psi::bc22::WolverineVoleFieldType &value) {
  std::string hash_input(
      sizeof(psi::bc22::WolverineVoleFieldType) + sizeof(size_t), '\0');
  std::memcpy(hash_input.data(), &bin_idx, sizeof(size_t));
  std::memcpy(hash_input.data() + sizeof(size_t), &value,
              sizeof(psi::bc22::WolverineVoleFieldType));
  auto hash_res = yacl::crypto::Blake3(hash_input);

  uint64_t res = 0;
  std::memcpy(&res, hash_res.data(), sizeof(uint64_t));
  return res;
}
}  // namespace

VoleCPsi::VoleCPsi(std::shared_ptr<yacl::link::Context> link_ctx,
                   psi::PsiRoleType role)
    : link_ctx_(std::move(link_ctx)),
      role_(role),
      batch_size_(kVoleCPsiBatchSize) {
  auto conn = std::make_shared<Communicator>(link_ctx_);
  base_ot_ =
      std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::YACL_Ferret);
  // vole_ = std::make_unique<psi::WolverineVole>(role_, link_ctx_);
  vole_ = std::make_shared<SilentVole>(link_ctx_, role_);
}

VoleCPsi::VoleCPsi(std::shared_ptr<yacl::link::Context> link_ctx,
                   std::shared_ptr<cheetah::BasicOTProtocols> base_ot,
                   psi::PsiRoleType role)
    : link_ctx_(std::move(link_ctx)),
      base_ot_(std::move(base_ot)),
      role_(role),
      batch_size_(kVoleCPsiBatchSize) {
  // vole_ = std::make_unique<psi::WolverineVole>(role_, link_ctx_);
  vole_ = std::make_shared<SilentVole>(link_ctx_, role_);
}

void VoleCPsi::ExchangeSetSize(size_t self_items_size) {
  size_t input_size = self_items_size;

  link_ctx_->SendAsyncThrottled(
      link_ctx_->NextRank(), psi::utils::SerializeSize(input_size),
      fmt::format("CPSI:SELF_SIZE={}", self_items_size));

  size_t peer_size = psi::utils::DeserializeSize(
      link_ctx_->Recv(link_ctx_->NextRank(), fmt::format("CPSI:PEER_SIZE")));

  peer_items_num_ = peer_size;
}

std::vector<uint8_t> VoleCPsi::RunCPsi(absl::Span<const std::string> items) {
  ExchangeSetSize(items.size());

  size_t compare_bytes_size =
      psi::utils::GetCompareBytesLength(items.size(), peer_items_num_);

  // SPDLOG_INFO("self size:{}, peer size:{} compare_bytes_size:{}", items.size(),
              // peer_items_num_, compare_bytes_size);

  if (role_ == psi::PsiRoleType::Sender) {
    cuckoo_options_ = psi::CuckooIndex::Options(
        {peer_items_num_, 0, kDefaultHashNum, scaleFactor});
    // sender alice

    // call mBaRK-OPRF
    oprfs_ = RunmBaRKOprfSender(items, compare_bytes_size);

    // simple bins
    OkvsEncode(oprfs_);

    SPDLOG_INFO("finish okvs encode and begin equal 64");

    // equal64
    masks_ = equal64(bin_values_);
    return masks_;

  } else if (role_ == psi::PsiRoleType::Receiver) {
    cuckoo_options_ = psi::CuckooIndex::Options(
        {items.size(), 0, kDefaultHashNum, scaleFactor});

    // receiver bob

    // call mBaRK-OPRF
    oprfs_ = RunmBaRKOprfReceiver(items, compare_bytes_size);

    // cuckoo bins
    OkvsDecode(oprfs_);
    // SPDLOG_INFO("finish okvs decode and begin equal 64");

    // equal64
    masks_ = equal64(bin_values_);
    return masks_;
  } else {
    SPU_THROW("wrong psi role type: {}", static_cast<int>(role_));
  }
}

std::vector<uint64_t> VoleCPsi::RunOprf(const vector<std::string> &items) {
  ExchangeSetSize(items.size());

  size_t compare_bytes_size =
      psi::utils::GetCompareBytesLength(items.size(), peer_items_num_);

  // SPDLOG_INFO("self size:{}, peer size:{} compare_bytes_size:{}", items.size(),
              // peer_items_num_, compare_bytes_size);

  if (role_ == psi::PsiRoleType::Sender) {
    cuckoo_options_ = psi::CuckooIndex::Options(
        {peer_items_num_, 0, kDefaultHashNum, scaleFactor});
    // sender alice

    // call mBaRK-OPRF
    std::vector<uint64_t> oprfs = RunmBaRKOprfSender(items, compare_bytes_size);

    // simple bins
    //            OkvsEncode(oprfs);
    //
    //            return bin_values_;

    //            // equal64
    //            auto bools = equal64(bin_values_);
    return oprfs;

  } else if (role_ == psi::PsiRoleType::Receiver) {
    cuckoo_options_ = psi::CuckooIndex::Options(
        {items.size(), 0, kDefaultHashNum, scaleFactor});

    // receiver bob

    // call mBaRK-OPRF
    std::vector<uint64_t> oprf_encode_vec =
        RunmBaRKOprfReceiver(items, compare_bytes_size);

    // cuckoo bins
    //            OkvsDecode(oprf_encode_vec);
    //
    //            return bin_values_;
    //
    //            // equal64
    //            auto bools = equal64(bin_values_);

    return oprf_encode_vec;
  } else {
    SPU_THROW("wrong psi role type: {}", static_cast<int>(role_));
  }
}

void VoleCPsi::OpprfSharePrevProc() {
  if (okvs_ptr_ == nullptr) {
    SPU_THROW("okvs pointer is not initialized");
  }

  if (bin_values_.empty() || oprfs_.empty()) {
    SPU_THROW("empty bin-values or oprf-values");
  }

  updateBinValues();
}

std::vector<uint64_t> VoleCPsi::OpprfSharePostProc(
    const std::vector<uint64_t> &res, bool bool_share) {
  SPU_ENFORCE(res.size() == masks_.size());

  if (bool_share) {
    return bmultiplexer(res, masks_);
  } else {
    return multiplexer(res, masks_);
  }
}

yacl::Buffer VoleCPsi::OpprfShareSendProc(std::vector<uint64_t> &data,
                                          bool bool_share) {
  uint64_t num_hash = cuckoo_options_.num_hash;
  SPU_ENFORCE(data.size() * num_hash == oprfs_.size(),
              "incorrect input data length or invalid oprf-values");

  std::vector<std::pair<std::string, uint64_t>> key_values;
  auto res = yacl::crypto::RandVec<uint64_t>(bins_.size(), true);

  for (size_t bin_idx = 0; bin_idx < bins_.size(); bin_idx++) {
    uint64_t rvalue = res[bin_idx];
    uint64_t bin_rand = bin_values_[bin_idx];

    for (const auto &elem : bins_[bin_idx]) {
      size_t item_idx = elem.InputIdx();
      size_t hash_idx = elem.HashIdx();
      uint64_t vv = oprfs_[item_idx * num_hash + hash_idx];

      psi::bc22::WolverineVoleFieldType expand_vv = bin_rand;
      expand_vv = (expand_vv << 64) | vv;
      vv = BaRKOPRFHash(bin_idx, expand_vv);

      std::string key(sizeof(uint64_t), '\0');
      std::memcpy(reinterpret_cast<uint8_t *>(key.data()), &vv,
                  sizeof(uint64_t));
      uint64_t value =
          bool_share ? (data[item_idx] ^ rvalue) : (data[item_idx] - rvalue);

      key_values.emplace_back(key, value);
    }
  }

  SPU_ENFORCE(key_values.size() <= oprfs_.size());

  size_t remain_size = oprfs_.size() - key_values.size();
  if (remain_size > 0) {
    auto tmp_rand = yacl::crypto::RandVec<uint64_t>(remain_size * 2, true);
    for (size_t i = 0; i < remain_size; i++) {
      key_values.emplace_back(std::to_string(tmp_rand[i * 2]),
                              tmp_rand[i * 2 + 1]);
    }
  }

  std::vector<uint64_t> gc_table = okvs_ptr_->encode(key_values);

  yacl::Buffer gc_table_buffer(gc_table.size() * sizeof(uint64_t));
  for (size_t i = 0; i < gc_table.size(); i++) {
    std::memcpy(reinterpret_cast<uint8_t *>(gc_table_buffer.data()) +
                    i * sizeof(uint64_t),
                &(gc_table[i]), sizeof(uint64_t));
  }

  data.resize(res.size());
  std::copy_n(res.data(), res.size(), data.data());

  return gc_table_buffer;
}

std::vector<uint64_t> VoleCPsi::OpprfShareRecvProc(
    yacl::Buffer &gc_table_buffer) {
  SPU_ENFORCE(gc_table_buffer.size() ==
              static_cast<int64_t>(okvs_ptr_->length() * sizeof(uint64_t)));

  std::vector<uint64_t> gc_table;
  for (int64_t i = 0; i < gc_table_buffer.size(); i += sizeof(uint64_t)) {
    uint64_t vv = 0;
    std::memcpy(&vv, reinterpret_cast<uint8_t *>(gc_table_buffer.data()) + i,
                sizeof(uint64_t));
    gc_table.push_back(vv);
  }

  vector<size_t> pos_arr;
  vector<string> items;
  auto res = yacl::crypto::RandVec<uint64_t>(bins_.size(), true);

  for (size_t bin_idx = 0; bin_idx < bins_.size(); bin_idx++) {
    // uint64_t value = yacl::crypto::SecureRandU64();
    uint64_t bin_rand = bin_values_[bin_idx];

    const auto &bin = bins_[bin_idx];
    if (!bin.empty()) {
      size_t item_idx = bin[0].InputIdx();
      uint64_t vv = oprfs_[item_idx];

      psi::bc22::WolverineVoleFieldType expand_vv = bin_rand;
      expand_vv = (expand_vv << 64) | vv;
      vv = BaRKOPRFHash(bin_idx, expand_vv);

      std::string item(sizeof(uint64_t), '\0');
      std::memcpy(reinterpret_cast<uint8_t *>(item.data()), &vv,
                  sizeof(uint64_t));

      //                value = okvs_ptr_->decode(gc_table, item);
      items.push_back(item);
      pos_arr.push_back(bin_idx);
    }
    // res[bin_idx] = value;
  }

  auto values = okvs_ptr_->decode(gc_table, items);
  for (size_t i = 0; i < values.size(); i++) {
    res[pos_arr[i]] = values[i];
  }

  return res;
}

std::vector<uint64_t> VoleCPsi::OpprfShareSender(const vector<uint64_t> &data,
                                                 bool bool_share) {
  OpprfSharePrevProc();

  std::vector<uint64_t> res(data.begin(), data.end());

  yacl::Buffer gc_table_buffer = OpprfShareSendProc(res, bool_share);

  link_ctx_->SendAsyncThrottled(
      link_ctx_->NextRank(), gc_table_buffer,
      fmt::format("send opprf-share-table of size {}", gc_table_buffer.size()));

  return OpprfSharePostProc(res, bool_share);
}

std::vector<uint64_t> VoleCPsi::OpprfShareReceiver(bool bool_share) {
  OpprfSharePrevProc();

  yacl::Buffer gc_table_buffer = link_ctx_->Recv(
      link_ctx_->NextRank(), fmt::format("recv opprf-share-table"));

  std::vector<uint64_t> res = OpprfShareRecvProc(gc_table_buffer);

  return OpprfSharePostProc(res, bool_share);
}

// mBaRK-OPRF
// std::vector<uint64_t> VoleCPsi::RunmBaRKOprfSender(absl::Span<const
// std::string> items,
//                                             size_t compare_bytes_size) {

//     printf("begin run mBaRKOprf: ");
//     link_ctx_->PrintStats();

//     cheetah::SimpleHashTable simple_table(cuckoo_options_);

//     psi::CuckooIndex cuckoo_index(cuckoo_options_);

//     // insert simple hash table
//     std::future<void> table_thread = std::async([&] {
//         SPDLOG_INFO("begin insert simple hash table");

//         simple_table.Insert(items);

//         SPDLOG_INFO("after insert simple hash table");
//     });

//     uint64_t vole_count_needed = cuckoo_options_.NumBins();

//     // vole extension
//     SPDLOG_INFO("begin pcg vole extension");
//     // wi = delta * ui + vi
//     // alice : delta vi
//     // bob : ui || wi as one __uint128_t
//     std::vector<psi::bc22::WolverineVoleFieldType> vole_blocks =
//             vole_->Extend(vole_count_needed);

//     psi::bc22::WolverineVoleFieldType delta = vole_->Delta();
//     SPDLOG_INFO("after pcg vole extension");

//     yacl::crypto::SilentVoleSender
//     vole_send(yacl::crypto::CodeType::Silver11);

//     printf("after vole: ");
//     link_ctx_->PrintStats();

//     const size_t coeff_byte_size = sizeof(psi::bc22::WolverineVoleFieldType);

//     // sender alice
//     // recv masked polynomial coeff
//     size_t recv_bin_idx = 0;
//     size_t bins_num = cuckoo_options_.NumBins();
//     SPDLOG_INFO("cuckoo_options_.NumBins: {}", bins_num);

//     SPDLOG_INFO("begin recv receiver's masked coeff");

//     std::vector<psi::bc22::WolverineVoleFieldType> masked_coeffs(bins_num);

//     yacl::Buffer masked_coeff_buffer = link_ctx_->Recv(
//                 link_ctx_->NextRank(), fmt::format("recv {} bin",
//                 recv_bin_idx));
//     SPU_ENFORCE((masked_coeff_buffer.size() % coeff_byte_size) == 0);

//     size_t num_batch_bin = masked_coeff_buffer.size() / coeff_byte_size;

//     std::memcpy(&masked_coeffs[recv_bin_idx],
//                 reinterpret_cast<uint8_t *>(masked_coeff_buffer.data()),
//                 masked_coeff_buffer.size());

//     recv_bin_idx += num_batch_bin;
//     // every kPcgPsiLogBatchSize bin print percentage
//     if (recv_bin_idx % kVoleCPsiLogBatchSize == 0) {
//         SPDLOG_INFO(
//                 "recv receiver's masked coeff, recv_bin_idx: {} Bins_Num:{} "
//                 "percentage:{}",
//                 recv_bin_idx, bins_num, (double)recv_bin_idx / bins_num);
//     }

//     SPU_ENFORCE(recv_bin_idx == bins_num, "{} != {}", recv_bin_idx,
//     bins_num); SPDLOG_INFO("after recv receiver's masked coeff, recv_bin_idx:
//     {}",
//                 recv_bin_idx);

//     table_thread.get();

//     bins_ = simple_table.bins();

//     items_hash_low64_ = simple_table.GetItemsHashLow64();

//     SPDLOG_INFO("role:{} items:{} bins size: {},items_hash_low64 size: {}",
//                 (role_ == psi::PsiRoleType::Sender) ? "sender" : "receiver",
//                 items.size(), bins_.size(), items_hash_low64_.size());

//     size_t num_hash = cuckoo_options_.num_hash;
//     std::vector<uint64_t> oprfs(items.size() * num_hash, 0);
//     std::vector<bool> oprfs_bool(items.size() * num_hash, true);
//     size_t oprf_empty_counts =  items.size() * num_hash;

//     // compute sender's oprf
//     for (size_t bin_idx = 0; bin_idx < bins_.size(); ++bin_idx) {
//         // delta * masked_coeff_i + v_i
//         // = delta * coeff_i + delta * u_i + v_i
//         // = delta * coeff_i + w_i
//         psi::bc22::WolverineVoleFieldType masked_coeff =
//         masked_coeffs[bin_idx]; psi::bc22::WolverineVoleFieldType tmp =
//         mod(delta * masked_coeff, pr); psi::bc22::WolverineVoleFieldType
//         oprf_key = mod(vole_blocks[bin_idx] + tmp, pr);

//         for (auto &candidate : bins_[bin_idx]) {
//             size_t item_idx = candidate.InputIdx();
//             size_t hash_idx = candidate.HashIdx();

//             uint64_t test_hash = items_hash_low64_[item_idx];
//             absl::string_view item_hash_str = absl::string_view(
//                     reinterpret_cast<const char *>(&test_hash),
//                     sizeof(uint64_t));

//             // delta * Poly(x) + wi
//             psi::bc22::WolverineVoleFieldType block_x = 0;
//             std::memcpy(&block_x, item_hash_str.data(),
//             item_hash_str.length()); block_x = mod(block_x, pr);

//             block_x = mod(block_x * delta, pr);
//             psi::bc22::WolverineVoleFieldType eval = mod(oprf_key + pr -
//             block_x, pr);

//             uint64_t hash_res = BaRKOPRFHash(bin_idx, eval);
//             auto oprfs_index = item_idx * num_hash + hash_idx;
//             oprfs[oprfs_index] = hash_res;
//             if (oprfs_bool[oprfs_index]) {
//                 oprfs_bool[oprfs_index] = false;
//                 oprf_empty_counts--;
//             }
//         }
//     }

//     if (oprf_empty_counts > 0) {
//         auto rand_oprfs =  yacl::crypto::RandVec<uint64_t>(oprf_empty_counts,
//         true);

//         for (auto & oprf : oprfs) {
//             if (oprf == 0) {
//                 oprf = rand_oprfs[oprf_empty_counts - 1];
//                 oprf_empty_counts--;
//             }
//         }
//     }

//     SPDLOG_INFO("after compute sender's oprf");

//     printf("after run mBaRKOprf: ");
//     link_ctx_->PrintStats();

//     return oprfs;
// }

//     std::vector<uint64_t> VoleCPsi::RunmBaRKOprfReceiver(
//             absl::Span<const std::string> items, size_t compare_bytes_size) {
// //        psi::WolverineVole vole(role_, link_ctx_);

//         cheetah::CuckooHashTable cuckoo_table(cuckoo_options_);

//         uint64_t vole_count_needed = cuckoo_options_.NumBins();

//         std::future<void> table_thread = std::async([&] {
//             SPDLOG_INFO("begin insert hash table");

//             cuckoo_table.Insert(items);

//             SPDLOG_INFO("after insert hash table");
//         });

//         // emp vole extension

//         SPDLOG_INFO("begin pcg vole extension");
//         // wi = delta * ui + vi
//         // alice : delta vi
//         // bob : ui || wi as one __uint128_t
//         std::vector<psi::bc22::WolverineVoleFieldType> vole_blocks =
//                 vole_->Extend(vole_count_needed);
//         SPDLOG_INFO("after pcg vole extension");

//         table_thread.get();

//         bins_ = cuckoo_table.bins();

//         items_hash_low64_ = cuckoo_table.GetItemsHashLow64();

//         SPDLOG_INFO("role:{} items:{} bins size: {}, items_hash_low64 size:
//         {}",
//                     (role_ == psi::PsiRoleType::Sender) ? "sender" :
//                     "receiver", items.size(), bins_.size(),
//                     items_hash_low64_.size());

//         const size_t coeff_byte_size =
//         sizeof(psi::bc22::WolverineVoleFieldType);

//         // receiver bob
//         // send mask
//         std::vector<uint64_t> oprf_encode_vec(items.size());

//         SPDLOG_INFO("begin compute and send receiver's masked coeff");

//         batch_size_ = vole_count_needed;
//         yacl::Buffer masked_coeff_buffer(coeff_byte_size *
//         vole_count_needed);

//         std::vector<std::string> oprf_blocks_batch(vole_count_needed);
//         auto bin_values = yacl::crypto::RandVec<uint64_t>(vole_count_needed,
//         true); for (size_t j = 0; j < vole_count_needed; ++j) {
//             size_t bin_idx = j;

//             uint64_t bin_value = bin_values[j];
//             if (!bins_[bin_idx].empty()) {
//                 bin_value = items_hash_low64_[bins_[bin_idx][0].InputIdx()];
//             }

//             std::string_view bin_data = absl::string_view(
//                     reinterpret_cast<const char *>(&bin_value),
//                     sizeof(uint64_t)
//             );

//             psi::bc22::WolverineVoleFieldType block_x = 0;
//             std::memcpy(&block_x, bin_data.data(), sizeof(uint64_t));
//             block_x = mod(block_x, pr);

//             // use vole mask polynomial coefficient
//             // coeff_i + ui
//             psi::bc22::WolverineVoleFieldType coeff_block = mod(block_x +
//             (vole_blocks[bin_idx] >> 64), pr);

//             size_t pos = j * coeff_byte_size;

//             // copy to masked_coeff send buffer
//             std::memcpy(
//                     reinterpret_cast<uint8_t *>(masked_coeff_buffer.data()) +
//                     pos, &coeff_block, coeff_byte_size);

//             if (!bins_[bin_idx].empty()) {
//                 // get vi, i: bin index
//                 psi::bc22::WolverineVoleFieldType coeff_vole =
//                 vole_blocks[bin_idx] & 0xFFFFFFFFFFFFFFFFLL;

//                 // compute oprf, H(i, wi), i: bin index
//                 uint64_t hash_res = BaRKOPRFHash(bin_idx, coeff_vole);

//                 size_t item_index = bins_[bin_idx][0].InputIdx();
//                 oprf_encode_vec[item_index] = hash_res;
//             }
//         }

//         link_ctx_->SendAsyncThrottled(
//                 link_ctx_->NextRank(), masked_coeff_buffer,
//                 fmt::format("send {} bin", vole_count_needed));

//         SPDLOG_INFO("after send receiver's masked coeff");

//         return oprf_encode_vec;
//     }

std::vector<uint64_t> VoleCPsi::RunmBaRKOprfSender(
    absl::Span<const std::string> items, size_t compare_bytes_size) {
  printf("begin run mBaRKOprf: ");
  link_ctx_->PrintStats();

  cheetah::SimpleHashTable simple_table(cuckoo_options_);

  psi::CuckooIndex cuckoo_index(cuckoo_options_);

  // insert simple hash table
  std::future<void> table_thread = std::async([&] {
    SPDLOG_INFO("begin insert simple hash table");

    simple_table.Insert(items);

    SPDLOG_INFO("after insert simple hash table");
  });

  uint64_t vole_count_needed = cuckoo_options_.NumBins();

  // vole extension
  // SPDLOG_INFO("begin pcg vole extension");
  // wi = delta * ui + vi
  // alice : delta vi
  // bob : ui || wi as one __uint128_t
  std::vector<uint64_t> v = vole_->Send(vole_count_needed);
  auto delta_gf64 = Galois64(vole_->Delta());
  // SPDLOG_INFO("after pcg vole extension");

  yacl::crypto::SilentVoleSender vole_send(yacl::crypto::CodeType::Silver11);

  printf("after vole: ");
  link_ctx_->PrintStats();

  const size_t coeff_byte_size = sizeof(uint64_t);

  // sender alice
  // recv masked polynomial coeff
  size_t recv_bin_idx = 0;
  size_t bins_num = cuckoo_options_.NumBins();
  // SPDLOG_INFO("cuckoo_options_.NumBins: {}", bins_num);

  // SPDLOG_INFO("begin recv receiver's masked coeff");

  std::vector<uint64_t> masked_coeffs(bins_num);

  yacl::Buffer masked_coeff_buffer = link_ctx_->Recv(
      link_ctx_->NextRank(), fmt::format("recv {} bin", recv_bin_idx));
  SPU_ENFORCE((masked_coeff_buffer.size() % coeff_byte_size) == 0);

  size_t num_batch_bin = masked_coeff_buffer.size() / coeff_byte_size;

  std::memcpy(&masked_coeffs[recv_bin_idx],
              reinterpret_cast<uint8_t *>(masked_coeff_buffer.data()),
              masked_coeff_buffer.size());

  recv_bin_idx += num_batch_bin;
  // every kPcgPsiLogBatchSize bin print percentage
  if (recv_bin_idx % kVoleCPsiLogBatchSize == 0) {
    SPDLOG_INFO(
        "recv receiver's masked coeff, recv_bin_idx: {} Bins_Num:{} "
        "percentage:{}",
        recv_bin_idx, bins_num, (double)recv_bin_idx / bins_num);
  }

  SPU_ENFORCE(recv_bin_idx == bins_num, "{} != {}", recv_bin_idx, bins_num);
  SPDLOG_INFO("after recv receiver's masked coeff, recv_bin_idx: {}",
              recv_bin_idx);

  table_thread.get();

  bins_ = simple_table.bins();

  items_hash_low64_ = simple_table.GetItemsHashLow64();

  SPDLOG_INFO("role:{} items:{} bins size: {},items_hash_low64 size: {}",
              (role_ == psi::PsiRoleType::Sender) ? "sender" : "receiver",
              items.size(), bins_.size(), items_hash_low64_.size());

  size_t num_hash = cuckoo_options_.num_hash;
  std::vector<uint64_t> oprfs(items.size() * num_hash, 0);

  size_t oprf_empty_counts = items.size() * num_hash;
  std::vector<bool> oprfs_bool(items.size() * num_hash, true);

  // compute sender's oprf
  for (size_t bin_idx = 0; bin_idx < bins_.size(); ++bin_idx) {
    // delta * masked_coeff_i ^ v_i
    // = delta * coeff_i ^ delta * u_i ^ v_i
    // = delta * coeff_i ^ w_i
    uint64_t tmp =
        (delta_gf64 * Galois64(masked_coeffs[bin_idx])).get<uint64_t>()[0];
    uint64_t oprf_key = v[bin_idx] ^ tmp;

    for (auto candidate : bins_[bin_idx]) {
      size_t item_idx = candidate.InputIdx();
      size_t hash_idx = candidate.HashIdx();

      // delta * Poly(x) + wi
      uint64_t block_x = items_hash_low64_[item_idx];
      block_x = (Galois64(block_x) * delta_gf64).get<uint64_t>()[0];
      uint64_t eval = oprf_key ^ block_x;
      auto oprfs_index = item_idx * num_hash + hash_idx;

      oprfs[oprfs_index] = BaRKOPRFHash(bin_idx, eval);
      if (oprfs_bool[oprfs_index]) {
        oprfs_bool[oprfs_index] = false;
        oprf_empty_counts--;
      }
    }
  }

  if (oprf_empty_counts > 0) {
    auto rand_oprfs = yacl::crypto::RandVec<uint64_t>(oprf_empty_counts);

    for (auto &oprf : oprfs) {
      if (oprf == 0) {
        oprf = rand_oprfs[oprf_empty_counts - 1];
        oprf_empty_counts--;
      }
    }
  }

  // SPDLOG_INFO("after compute sender's oprf");

  // printf("after run mBaRKOprf: ");
  // link_ctx_->PrintStats();

  return oprfs;
}

std::vector<uint64_t> VoleCPsi::RunmBaRKOprfReceiver(
    absl::Span<const std::string> items, size_t compare_bytes_size) {
  //        psi::WolverineVole vole(role_, link_ctx_);

  cheetah::CuckooHashTable cuckoo_table(cuckoo_options_);

  uint64_t vole_count_needed = cuckoo_options_.NumBins();

  std::future<void> table_thread = std::async([&] {
    // SPDLOG_INFO("begin insert hash table");

    cuckoo_table.Insert(items);

    // SPDLOG_INFO("after insert hash table");
  });

  // emp vole extension

  // SPDLOG_INFO("begin pcg vole extension");
  // wi = delta * ui ^ vi
  // alice : delta vi
  // bob : ui || wi as one __uint128_t
  std::array<std::vector<uint64_t>, 2> ss = vole_->Recv(vole_count_needed);
  auto u = ss[0];
  auto w = ss[1];

  // SPDLOG_INFO("after pcg vole extension");

  table_thread.get();

  bins_ = cuckoo_table.bins();

  items_hash_low64_ = cuckoo_table.GetItemsHashLow64();

  // SPDLOG_INFO("role:{} items:{} bins size: {}, items_hash_low64 size: {}",
  //             (role_ == psi::PsiRoleType::Sender) ? "sender" : "receiver",
  //             items.size(), bins_.size(), items_hash_low64_.size());

  const size_t coeff_byte_size = sizeof(uint64_t);

  // receiver bob
  // send mask
  std::vector<uint64_t> oprf_encode_vec(items.size());

  // SPDLOG_INFO("begin compute and send receiver's masked coeff");

  batch_size_ = vole_count_needed;
  size_t current_batch_size = vole_count_needed;
  yacl::Buffer masked_coeff_buffer(coeff_byte_size * vole_count_needed);

  auto bin_values = yacl::crypto::RandVec<uint64_t>(vole_count_needed, true);

  std::vector<std::string> oprf_blocks_batch(vole_count_needed);
  for (size_t j = 0; j < vole_count_needed; ++j) {
    size_t bin_idx = j;

    uint64_t &bin_value = bin_values[j];
    if (!bins_[bin_idx].empty()) {
      bin_value = items_hash_low64_[bins_[bin_idx][0].InputIdx()];
    }

    // use vole mask polynomial coefficient
    // coeff_i ^ ui
    uint64_t coeff_block = bin_value ^ u[bin_idx];

    size_t pos = j * coeff_byte_size;

    // copy to masked_coeff send buffer
    std::memcpy(reinterpret_cast<uint8_t *>(masked_coeff_buffer.data()) + pos,
                &coeff_block, coeff_byte_size);

    if (!bins_[bin_idx].empty()) {
      // get vi, i: bin index
      psi::bc22::WolverineVoleFieldType coeff_vole = w[bin_idx];

      // compute oprf, H(i, wi), i: bin index
      uint64_t hash_res = BaRKOPRFHash(bin_idx, coeff_vole);

      size_t item_index = bins_[bin_idx][0].InputIdx();
      oprf_encode_vec[item_index] = hash_res;
    }
  }

  link_ctx_->SendAsyncThrottled(link_ctx_->NextRank(), masked_coeff_buffer,
                                fmt::format("send {} bin", current_batch_size));

  // SPDLOG_INFO("after send receiver's masked coeff");

  return oprf_encode_vec;
}

void VoleCPsi::OkvsEncode(const std::vector<uint64_t> &oprfs) {
  std::vector<std::pair<std::string, uint64_t>> key_values;

  bin_values_.resize(bins_.size());
  yacl::crypto::FillRand((char *)&bin_values_[0],
                         bin_values_.size() * sizeof(uint64_t));

  for (size_t i = 0; i < bins_.size(); i++) {
    auto &bin = bins_[i];
    // const auto &bin: bins_) {
    for (const auto &elem : bin) {
      size_t item_idx = elem.InputIdx();
      size_t hash_idx = elem.HashIdx();

      std::string key(sizeof(uint64_t), '\0');
      std::memcpy(reinterpret_cast<uint8_t *>(key.data()),
                  &oprfs[item_idx * cuckoo_options_.num_hash + hash_idx],
                  sizeof(uint64_t));
      // std::pair<std::string, uint64_t> tuples = std::pair(key, value);
      key_values.emplace_back(key, bin_values_[i]);
    }
  }

  SPU_ENFORCE(key_values.size() <= oprfs.size());
  size_t remain_size = oprfs.size() - key_values.size();
  if (remain_size > 0) {
    auto tmp_rand = yacl::crypto::RandVec<uint64_t>(remain_size * 2);
    for (size_t i = 0; i < remain_size; i++) {
      key_values.emplace_back(std::to_string(tmp_rand[i * 2]),
                              tmp_rand[i * 2 + 1]);
    }
  }
  // while (key_values.size() < oprfs.size()) {
  //     std::string key(sizeof(uint64_t), '\0');
  //     RAND_bytes(reinterpret_cast<uint8_t *>(key.data()), sizeof(uint64_t));
  //     uint64_t value = yacl::crypto::SecureRandU64();

  //     // std::pair<std::string, uint64_t> tuples = std::pair(key, value);
  //     key_values.emplace_back(key, value);
  // }

  okvs_ptr_ = std::make_unique<cheetah::PaxosOKVS>(key_values.size());
  std::vector<uint64_t> gc_table = okvs_ptr_->encode(key_values);

  yacl::Buffer gc_table_buffer(gc_table.data(),
                               gc_table.size() * sizeof(uint64_t));
  // for (size_t i = 0; i < gc_table.size(); i++) {
  //     std::memcpy(reinterpret_cast<uint8_t *>(gc_table_buffer.data()) + i *
  //     sizeof(uint64_t),
  //                 &(gc_table[i]), sizeof(uint64_t));
  // }

  link_ctx_->SendAsyncThrottled(
      link_ctx_->NextRank(), gc_table_buffer,
      fmt::format("send garbled table of size {}", gc_table_buffer.size()));
}

void VoleCPsi::OkvsDecode(const std::vector<uint64_t> &oprfs) {
  okvs_ptr_ = std::make_unique<cheetah::PaxosOKVS>(peer_items_num_ *
                                                   cuckoo_options_.num_hash);

  yacl::Buffer gc_table_buffer =
      link_ctx_->Recv(link_ctx_->NextRank(), fmt::format("recv garbled table"));
  SPU_ENFORCE(gc_table_buffer.size() ==
              static_cast<int64_t>(okvs_ptr_->length() * sizeof(uint64_t)));

  std::vector<uint64_t> gc_table;
  for (int64_t i = 0; i < gc_table_buffer.size(); i += sizeof(uint64_t)) {
    void *vv = reinterpret_cast<uint8_t *>(gc_table_buffer.data()) + i;
    // std::memcpy(&vv, reinterpret_cast<uint8_t *>(gc_table_buffer.data()) + i,
    // sizeof(uint64_t));
    gc_table.emplace_back(*(uint64_t *)vv);
  }

  std::vector<size_t> pos_arr;
  std::vector<std::string> items;
  bin_values_.resize(bins_.size());
  //        for (auto &bin: bins_) {
  yacl::crypto::FillRand((char *)&bin_values_[0],
                         bin_values_.size() * sizeof(uint64_t));
  for (size_t i = 0; i < bins_.size(); i++) {
    const auto &bin = bins_[i];

    if (!bin.empty()) {
      size_t item_idx = bin[0].InputIdx();
      uint64_t vv = oprfs[item_idx];

      std::string item(sizeof(uint64_t), '\0');
      std::memcpy(reinterpret_cast<uint8_t *>(item.data()), &vv,
                  sizeof(uint64_t));

      items.emplace_back(std::move(item));
      pos_arr.push_back(i);
    }
  }

  auto values = okvs_ptr_->decode(gc_table, items);
  for (size_t i = 0; i < pos_arr.size(); i++) {
    bin_values_[pos_arr[i]] = values[i];
  }
}

std::vector<uint8_t> VoleCPsi::equal64(const std::vector<uint64_t> &arr) {
  // equal64
  size_t size = arr.size();
  NdArrayRef rv(makeType<RingTy>(FieldType::FM64),
                {static_cast<int64_t>(size)});

  DISPATCH_ALL_FIELDS(FieldType::FM64, "", [&]() {
    auto data = NdArrayView<ring2k_t>(rv);
    for (int64_t i = 0; i < data.numel(); ++i) {
      data[i] = arr[i];
    }
  });

  EqualProtocol eq_prot(base_ot_);
  NdArrayRef res = eq_prot.Compute(rv);

  auto *res_ptr = res.data<uint64_t>();
  vector<uint8_t> bools(arr.size());
  for (size_t i = 0; i < bools.size(); i++) {
    bools[i] = res_ptr[i];
  }

  return bools;
}

void VoleCPsi::updateBinValues() {
  if (masks_.empty()) {
    SPU_THROW("VoleCPsi::updateBinValues: mask is empty");
  }

  int64_t size = masks_.size();
  auto rands = yacl::crypto::RandVec<uint64_t>(size, true);

  vector<uint8_t> bools(size);
  std::copy_n(masks_.data(), masks_.size(), bools.data());

  if (role_ == psi::PsiRoleType::Sender) {
    transform(bools.begin(), bools.end(), bools.begin(),
              [&](uint8_t x) { return x ^ 1; });
  }

  bin_values_ = bmultiplexer(rands, bools);
}

std::vector<uint64_t> VoleCPsi::bmultiplexer(const std::vector<uint64_t> &arr,
                                             const std::vector<uint8_t> &bools,
                                             int zeros) {
  if (arr.size() != bools.size()) {
    SPU_THROW("VoleCPsi::bmultiplexer: input length should be equal");
  }

  int64_t size = arr.size();

  vector<uint64_t> rand0(size);
  vector<uint64_t> rand1(size);
  vector<uint64_t> recv(size);

  vector<uint8_t> sel(size);
  pforeach(0, size, [&](int64_t i) { sel[i] = bools[i] & 0x1; });

  auto sender = base_ot_->GetSenderCOT();
  auto receiver = base_ot_->GetReceiverCOT();

  if (zeros < 0) {
    // Compute (b0 ^ b1) ? x1 : 0
    if (role_ == psi::PsiRoleType::Sender) {
      for (const auto vv : arr) {
        SPU_ENFORCE(vv == 0, "input should be zeros");
      }
      receiver->RecvRMCC(absl::MakeSpan(sel), absl::MakeSpan(recv));

      yacl::Buffer buf(size * sizeof(uint64_t));
      buf = link_ctx_->Recv(link_ctx_->NextRank(),
                            fmt::format("recv correlated msg"));
      SPU_ENFORCE(buf.size() == static_cast<int64_t>(size * sizeof(uint64_t)));

      std::memcpy(reinterpret_cast<uint8_t *>(rand1.data()),
                  reinterpret_cast<uint8_t *>(buf.data()), buf.size());

      pforeach(0, size,
               [&](int64_t i) { recv[i] = recv[i] ^ (sel[i] * rand1[i]); });
    } else {
      sender->SendRMCC(absl::MakeSpan(rand0), absl::MakeSpan(rand1));
      sender->Flush();

      pforeach(0, size,
               [&](int64_t i) { rand1[i] = rand1[i] ^ rand0[i] ^ arr[i]; });

      yacl::Buffer buf(rand1.data(), rand1.size() * sizeof(uint64_t));
      link_ctx_->SendAsyncThrottled(
          link_ctx_->NextRank(), buf,
          fmt::format("send correlated msg of size {}", buf.size()));

      pforeach(0, size,
               [&](int64_t i) { recv[i] = rand0[i] ^ (arr[i] * sel[i]); });
    }

  } else if (zeros > 0) {
    // Compute (b0 ^ b1) ? x0 : 0
    if (role_ == psi::PsiRoleType::Sender) {
      sender->SendRMCC(absl::MakeSpan(rand0), absl::MakeSpan(rand1));
      sender->Flush();

      pforeach(0, size,
               [&](int64_t i) { rand1[i] = rand1[i] ^ rand0[i] ^ arr[i]; });

      yacl::Buffer buf(rand1.data(), rand1.size() * sizeof(uint64_t));
      link_ctx_->SendAsyncThrottled(
          link_ctx_->NextRank(), buf,
          fmt::format("send correlated msg of size {}", buf.size()));

      pforeach(0, size,
               [&](int64_t i) { recv[i] = rand0[i] ^ (arr[i] * sel[i]); });
    } else {
      for (const auto vv : arr) {
        SPU_ENFORCE(vv == 0, "input should be zeros");
      }
      receiver->RecvRMCC(absl::MakeSpan(sel), absl::MakeSpan(recv));

      yacl::Buffer buf(size * sizeof(uint64_t));
      buf = link_ctx_->Recv(link_ctx_->NextRank(),
                            fmt::format("recv correlated msg"));
      SPU_ENFORCE(buf.size() == static_cast<int64_t>(size * sizeof(uint64_t)));

      std::memcpy(reinterpret_cast<uint8_t *>(rand1.data()),
                  reinterpret_cast<uint8_t *>(buf.data()), buf.size());

      pforeach(0, size,
               [&](int64_t i) { recv[i] = recv[i] ^ (sel[i] * rand1[i]); });
    }
  } else {
    if (role_ == psi::PsiRoleType::Sender) {
      sender->SendRMCC(absl::MakeSpan(rand0), absl::MakeSpan(rand1));
      sender->Flush();

      receiver->RecvRMCC(absl::MakeSpan(sel), absl::MakeSpan(recv));
    } else {
      receiver->RecvRMCC(absl::MakeSpan(sel), absl::MakeSpan(recv));

      sender->SendRMCC(absl::MakeSpan(rand0), absl::MakeSpan(rand1));
      sender->Flush();
    }

    vector<uint64_t> corr(size);
    pforeach(0, size,
             [&](int64_t i) { corr[i] = rand1[i] ^ rand0[i] ^ arr[i]; });

    if (role_ == psi::PsiRoleType::Sender) {
      yacl::Buffer buf(corr.data(), corr.size() * sizeof(uint64_t));
      link_ctx_->SendAsyncThrottled(
          link_ctx_->NextRank(), buf,
          fmt::format("send correlated msg of size {}", buf.size()));

      buf = link_ctx_->Recv(link_ctx_->NextRank(),
                            fmt::format("recv correlated msg"));
      SPU_ENFORCE(buf.size() == static_cast<int64_t>(size * sizeof(uint64_t)));

      std::memcpy(reinterpret_cast<uint8_t *>(rand1.data()),
                  reinterpret_cast<uint8_t *>(buf.data()), buf.size());
    } else {
      yacl::Buffer buf = link_ctx_->Recv(link_ctx_->NextRank(),
                                         fmt::format("recv correlated msg"));
      SPU_ENFORCE(buf.size() == static_cast<int64_t>(size * sizeof(uint64_t)));

      std::memcpy(reinterpret_cast<uint8_t *>(rand1.data()),
                  reinterpret_cast<uint8_t *>(buf.data()), buf.size());

      std::memcpy(reinterpret_cast<uint8_t *>(buf.data()),
                  reinterpret_cast<uint8_t *>(corr.data()), buf.size());
      link_ctx_->SendAsyncThrottled(
          link_ctx_->NextRank(), buf,
          fmt::format("send correlated msg of size {}", buf.size()));
    }

    pforeach(0, size, [&](int64_t i) {
      recv[i] = (sel[i] * arr[i]) ^ rand0[i] ^ (sel[i] * rand1[i]) ^ recv[i];
    });
  }

  return recv;
}

std::vector<uint64_t> VoleCPsi::multiplexer(const std::vector<uint64_t> &arr,
                                            const std::vector<uint8_t> &bools,
                                            int zeros) {
  if (arr.size() != bools.size()) {
    SPU_THROW("VoleCPsi::multiplexer: input length should be equal");
  }

  int64_t size = arr.size();
  vector<uint64_t> corr(size);
  vector<uint64_t> recv(size);

  vector<uint8_t> sel(size);
  pforeach(0, size, [&](int64_t i) { sel[i] = bools[i] & 0x1; });

  auto sender = base_ot_->GetSenderCOT();
  auto receiver = base_ot_->GetReceiverCOT();

  if (zeros < 0) {
    if (role_ == psi::PsiRoleType::Sender) {
      for (const auto vv : arr) {
        SPU_ENFORCE(vv == 0, "input should be zeros");
      }
      receiver->RecvCAMCC(absl::MakeSpan(sel), absl::MakeSpan(recv));
    } else {
      // Compute (b0 ^ b1) ? x1 : 0
      pforeach(0, size,
               [&](int64_t i) { corr[i] = arr[i] - (2 * arr[i] * sel[i]); });

      sender->SendCAMCC(absl::MakeSpan(corr), absl::MakeSpan(recv));
      sender->Flush();

      pforeach(0, size,
               [&](int64_t i) { recv[i] = (arr[i] * sel[i]) - recv[i]; });
    }
  } else if (zeros > 0) {
    if (role_ == psi::PsiRoleType::Sender) {
      // Compute (b0 ^ b1) ? x0 : 0
      pforeach(0, size,
               [&](int64_t i) { corr[i] = arr[i] - (2 * arr[i] * sel[i]); });

      sender->SendCAMCC(absl::MakeSpan(corr), absl::MakeSpan(recv));
      sender->Flush();

      pforeach(0, size,
               [&](int64_t i) { recv[i] = (arr[i] * sel[i]) - recv[i]; });
    } else {
      for (const auto vv : arr) {
        SPU_ENFORCE(vv == 0, "input should be zeros");
      }
      receiver->RecvCAMCC(absl::MakeSpan(sel), absl::MakeSpan(recv));
    }
  } else {
    // Compute (b0 ^ b1) ? (x0 + x1) : 0
    pforeach(0, size, [&](int64_t i) { corr[i] = arr[i] * (1 - 2 * sel[i]); });

    vector<uint64_t> value(size);
    if (role_ == psi::PsiRoleType::Sender) {
      sender->SendCAMCC(absl::MakeSpan(corr), absl::MakeSpan(recv));
      sender->Flush();

      receiver->RecvCAMCC(absl::MakeSpan(sel), absl::MakeSpan(value));

    } else {
      receiver->RecvCAMCC(absl::MakeSpan(sel), absl::MakeSpan(value));

      sender->SendCAMCC(absl::MakeSpan(corr), absl::MakeSpan(recv));
      sender->Flush();
    }

    pforeach(0, size, [&](int64_t i) {
      recv[i] = (arr[i] * sel[i]) - recv[i] + value[i];
    });
  }

  return recv;
}

SilentVole::SilentVole(std::shared_ptr<yacl::link::Context> link_ctx,
                       psi::PsiRoleType role)
    : role_(role), link_ctx_(link_ctx) {
  if (role_ == psi::PsiRoleType::Sender) {
    vole_send_ = std::make_shared<yacl::crypto::SilentVoleSender>(code_type_);
  } else {
    vole_recv_ = std::make_shared<yacl::crypto::SilentVoleReceiver>(code_type_);
  }
}

std::vector<uint64_t> SilentVole::Send(size_t n) {
  if (vole_send_ == nullptr) {
    vole_send_ = std::make_shared<yacl::crypto::SilentVoleSender>(code_type_);
  }

  std::vector<uint64_t> w(n);
  vole_send_->Send(link_ctx_, absl::MakeSpan(w));
  delta_ = vole_send_->GetDelta64();

  return w;
}

std::array<std::vector<uint64_t>, 2> SilentVole::Recv(size_t n) {
  if (vole_recv_ == nullptr) {
    vole_recv_ = std::make_shared<yacl::crypto::SilentVoleReceiver>(code_type_);
  }

  std::vector<uint64_t> u(n);
  std::vector<uint64_t> v(n);
  vole_recv_->Recv(link_ctx_, absl::MakeSpan(u), absl::MakeSpan(v));

  return {u, v};
}

}  // namespace spu::mpc::cheetah
