

#ifndef SPU_CUCKOO_HASH_H
#define SPU_CUCKOO_HASH_H

#pragma once
#include <memory>

#include "psi/bc22/generalized_cuckoo_hash.h"
namespace spu::mpc::cheetah {
    class CuckooHashTable : public psi::bc22::IPsiHashTable {
    public:
        explicit CuckooHashTable(psi::CuckooIndex::Options options, uint128_t seed = 0);

        void Insert(yacl::ByteContainerView item_data, size_t input_offset);
        void Insert(yacl::ByteContainerView item);
        void Insert(absl::Span<const std::string> items) override;

        const std::vector<std::vector<psi::CuckooIndex::Bin>> &bins() const {
            return bins_;
        }

        const std::vector<uint64_t> &GetItemsHashLow64() const {
            return items_hash_low64_;
        }

        int64_t GetItemIndex(size_t index) const {
            if (bins_[index].empty()) {
                return -1;
            } else if (bins_[index].size() == 1){
                return bins_[index][0].InputIdx();
            } else {
                throw yacl::RuntimeError("bin size should be no more than 1");
            }
        }

        uint64_t GetHashIndex(size_t index) const {
            if (bins_[index].empty()) {
                return static_cast<uint64_t>(-1);
            } else if (bins_[index].size() == 1){
                return bins_[index][0].HashIdx();
            } else {
                throw yacl::RuntimeError("bin size should be no more than 1");
            }
        }

        uint64_t GetItemHashLow64(size_t index) {
            if (bins_[index].empty()) {
                return 0;
            } else if (bins_[index].size() == 1){
                return items_hash_low64_[bins_[index][0].InputIdx()];
            } else {
                throw yacl::RuntimeError("bin size should be no more than 1");
            }
        }

        // Returns the current fill rate of the hash table and stash.
        inline double FillRate() const noexcept {
            return static_cast<double>(inserted_items_) /
                   (static_cast<double>(options_.NumBins()));
        }

        const psi::CuckooIndex::Options &GetCuckooOptions() const { return options_; }

    protected:
        psi::CuckooIndex::Options options_;
        // max data number per bin
        uint128_t seed_;
        std::vector<std::vector<psi::CuckooIndex::Bin>> bins_;
        std::vector<std::vector<uint64_t>> hashes_;
        std::vector<uint64_t> items_hash_low64_;
        size_t inserted_items_ = 0;

        // Randomness source for location function sampling.
        std::mt19937_64 gen_;

        std::uniform_int_distribution<std::uint32_t> uniform_hash_idx_;
    };

    class SimpleHashTable : public psi::bc22::IPsiHashTable {
    public:
        explicit SimpleHashTable(psi::CuckooIndex::Options options, uint128_t seed = 0);

        void Insert(yacl::ByteContainerView item_data,
                    const std::vector<uint64_t> &hash_bin_idx);
        void Insert(yacl::ByteContainerView item);
        void Insert(absl::Span<const std::string> items) override;

        const std::vector<std::vector<psi::CuckooIndex::Bin>> &bins() const {
            return bins_;
        }

        const std::vector<uint64_t> &GetItemsHashLow64() const {
            return items_hash_low64_;
        }

        const psi::CuckooIndex::Options &GetCuckooOptions() const { return options_; }

        std::vector<uint64_t> GetItemIndex(size_t index) const {
            std::vector<uint64_t> res;
            if (!bins_[index].empty()) {
                for (auto j : bins_[index]) {
                    res.push_back(j.InputIdx());
                }
            }
            return res;
        }

        std::vector<uint64_t> GetHashIndex(size_t index) const {
            std::vector<uint64_t> res;
            if (bins_[index].empty()) {
                res.push_back(static_cast<uint64_t>(-1));
            } else {
                for (auto j : bins_[index]) {
                    res.push_back(j.HashIdx());
                }
            }
            return res;
        }

        std::vector<uint64_t> GetItemHashLow64(size_t index) const {
            std::vector<uint64_t> res;
            if (!bins_[index].empty()) {
                for (auto j : bins_[index]) {
                    res.push_back(items_hash_low64_[j.InputIdx()]);
                }
            }
            return res;
        }

    protected:
        psi::CuckooIndex::Options options_;
        uint128_t seed_;
        std::vector<std::vector<psi::CuckooIndex::Bin>> bins_;
        std::vector<uint64_t> items_hash_low64_;
        size_t inserted_items_ = 0;
    };
};

#endif //SPU_CUCKOO_HASH_H
