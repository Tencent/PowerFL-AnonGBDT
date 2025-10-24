
#include "libspu/mpc/cheetah/cpsi/cuckoo_hash.h"

#include <set>
#include <utility>

#include "absl/strings/escaping.h"
#include "spdlog/spdlog.h"
#include "yacl/utils/parallel.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/hash/hash_utils.h"

#include "libspu/core/prelude.h"

namespace spu::mpc::cheetah {
constexpr size_t kDefaultHashNum = 3;

std::array<uint64_t, 4> computeHash(yacl::ByteContainerView item) {
  std::array<uint8_t, 32> hashs = yacl::crypto::Sha256(item);
  std::array<uint64_t, 4> hash_values;
  std::memcpy(hash_values.data(), hashs.data(), 4 * sizeof(uint64_t));
  return hash_values;
}

std::vector<uint64_t> GetBinIdx(const psi::CuckooIndex::Options &options,
                                const std::array<uint64_t, 4> &item_hashs) {
  SPU_ENFORCE(options.num_hash == kDefaultHashNum);
  size_t num_bins = options.NumBins();

  std::vector<uint64_t> hash_bin_idx(options.num_hash);

  hash_bin_idx[0] = item_hashs[0] % num_bins;
  hash_bin_idx[1] = item_hashs[1] % num_bins;
  hash_bin_idx[2] = item_hashs[2] % num_bins;

  return hash_bin_idx;
}

psi::CuckooIndex::Options GetCuckooHashOption(size_t hash_num,
                                              size_t items_size) {
  psi::CuckooIndex::Options options;

  options.num_input = items_size;
  options.num_stash = 0;
  options.num_hash = hash_num;
  options.max_try_count = 512;

  SPU_ENFORCE(hash_num == kDefaultHashNum, "just support 3 hash");

  if (hash_num == kDefaultHashNum) {
    options.scale_factor = 1.3;
  } else {
    SPU_THROW("unsupported");
  }
  return options;
}

CuckooHashTable::CuckooHashTable(psi::CuckooIndex::Options options,
                                 uint128_t seed)
    : options_(options), seed_(seed), gen_(yacl::crypto::SecureRandU64()) {
  size_t table_size = options_.NumBins();
  bins_.resize(table_size);

  uniform_hash_idx_ =
      std::uniform_int_distribution<uint32_t>(0, options_.num_hash - 1);
}

void CuckooHashTable::Insert(yacl::ByteContainerView item_data,
                             size_t input_offset) {
  psi::CuckooIndex::Bin candidate;
  candidate.set_encoded(input_offset);

  int64_t level = options_.max_try_count;
  size_t bin_idx;
  while ((level--) != 0) {
    size_t rand_hash_idx = uniform_hash_idx_(gen_);

    for (uint32_t i = 0; i < options_.num_hash; i++) {
      size_t hash_idx = (rand_hash_idx + i) % options_.num_hash;

      bin_idx = hashes_[candidate.InputIdx()][hash_idx];

      if (bins_[bin_idx].empty()) {
        uint64_t next_candid =
            psi::CuckooIndex::Bin::Encode(candidate.InputIdx(), hash_idx);
        candidate.set_encoded(next_candid);

        bins_[bin_idx].push_back(candidate);
        inserted_items_++;

        return;
      }
    }

    // random select bin_idx, swap candidate
    rand_hash_idx = uniform_hash_idx_(gen_);

    bin_idx = hashes_[candidate.InputIdx()][rand_hash_idx];

    uint64_t next_candid =
        psi::CuckooIndex::Bin::Encode(candidate.InputIdx(), rand_hash_idx);
    candidate.set_encoded(next_candid);

    candidate =
        psi::CuckooIndex::Bin(bins_[bin_idx][0].Swap(candidate.encoded()));
  }

  SPU_THROW(
      "Error insert, level:{} insert item_data:{}", level,
      absl::BytesToHexString(absl::string_view(
          reinterpret_cast<const char *>(item_data.data()), item_data.size())));
}

void CuckooHashTable::Insert(yacl::ByteContainerView item) {
  auto item_hashs = computeHash(item);

  size_t input_offset = hashes_.size();

  // hash_bin_idx
  std::vector<uint64_t> hash_bin_idx =
      GetBinIdx(options_, item_hashs);
  hashes_.push_back(hash_bin_idx);

  items_hash_low64_.push_back(item_hashs[3]);

  Insert(item, input_offset);
}

void CuckooHashTable::Insert(absl::Span<const std::string> items) {
  size_t input_offset = items_hash_low64_.size();
  items_hash_low64_.resize(input_offset + items.size());
  hashes_.resize(input_offset + items.size());

  for (size_t i = 0; i < items.size(); ++i) {
    auto item_hashs = computeHash(items[i]);

    items_hash_low64_[input_offset + i] = item_hashs[3];
    hashes_[input_offset + i] = GetBinIdx(options_, item_hashs);
  }

  for (size_t i = 0; i < items.size(); ++i) {
    Insert(items[i], input_offset + i);
  }
}

SimpleHashTable::SimpleHashTable(psi::CuckooIndex::Options options,
                                 uint128_t seed)
    : options_(options), seed_(seed) {
  size_t table_size = options_.NumBins();

  bins_.resize(table_size);
}

void SimpleHashTable::Insert(yacl::ByteContainerView item_data,
                             const std::vector<uint64_t> &hash_bin_idx) {
  psi::CuckooIndex::Bin candidate;
  candidate.set_encoded(inserted_items_);

  SPU_ENFORCE(options_.num_hash == 3 && hash_bin_idx.size() == 3,
              "only support 3-cuckoo hash");
  std::vector<size_t> dhash_bin_idx;
  std::vector<size_t> hash_index;

  dhash_bin_idx.push_back(hash_bin_idx[0]);
  hash_index.push_back(0);

  if (hash_bin_idx[1] != dhash_bin_idx[0]) {
    dhash_bin_idx.push_back(hash_bin_idx[1]);
    hash_index.push_back(1);
  }

  if (std::find(dhash_bin_idx.begin(), dhash_bin_idx.end(), hash_bin_idx[2]) ==
      dhash_bin_idx.end()) {
    dhash_bin_idx.push_back(hash_bin_idx[2]);
    hash_index.push_back(2);
  }

  //        std::set<size_t> idx_set(hash_bin_idx.begin(), hash_bin_idx.end());
  //        std::vector<size_t> dhash_bin_idx(idx_set.begin(), idx_set.end());
  //        std::vector<size_t> hash_index;
  //
  //        if (dhash_bin_idx.size() < hash_bin_idx.size()) {
  //            for (const auto bin_index: dhash_bin_idx) {
  //                for (size_t i = 0; i < hash_bin_idx.size(); i++) {
  //                    if (bin_index == hash_bin_idx[i]) {
  //                        hash_index.push_back(i);
  //                        break;
  //                    }
  //                }
  //            }
  //        } else{
  //            hash_index.resize(options_.num_hash);
  //            std::iota(hash_index.begin(), hash_index.end(), 0);
  //        }

  for (size_t i = 0; i < hash_index.size(); i++) {
    size_t hash_id = hash_index[i];
    size_t bin_idx = dhash_bin_idx[i];
    uint64_t next_candid =
        psi::CuckooIndex::Bin::Encode(candidate.InputIdx(), hash_id);
    candidate.set_encoded(next_candid);
    bins_[bin_idx].push_back(candidate);
  }

  inserted_items_++;
}

void SimpleHashTable::Insert(yacl::ByteContainerView item) {
  auto item_hashs = computeHash(item);

  std::vector<uint64_t> hash_bin_idx;

  items_hash_low64_.push_back(item_hashs[3]);

  hash_bin_idx = GetBinIdx(options_, item_hashs);

  Insert(item, hash_bin_idx);
}

void SimpleHashTable::Insert(absl::Span<const std::string> items) {
  size_t input_offset = items_hash_low64_.size();
  items_hash_low64_.resize(input_offset + items.size());

  std::vector<std::vector<uint64_t>> hash_bin_idx(items.size());

  for (size_t i = 0; i < items.size(); ++i) {
    auto item_hashs = computeHash(items[i]);

    hash_bin_idx[i] = GetBinIdx(options_, item_hashs);

    items_hash_low64_[input_offset + i] = item_hashs[3];
  }

  for (size_t i = 0; i < items.size(); ++i) {
    Insert(items[i], hash_bin_idx[i]);
  }
}
};  // namespace spu::mpc::cheetah
