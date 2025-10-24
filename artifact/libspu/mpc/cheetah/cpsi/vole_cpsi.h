

#ifndef SPU_VOLE_CPSI_H
#define SPU_VOLE_CPSI_H

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "paxos_okvs.h"
#include "yacl/base/int128.h"
#include "yacl/kernels/algorithms/silent_vole.h"
#include "yacl/link/context.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/nonlinear/equal_prot.h"
#include "libspu/mpc/cheetah/okvs/galois.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"

#include "psi/bc22/emp_vole.h"
#include "psi/bc22/generalized_cuckoo_hash.h"
#include "psi/utils/communication.h"

namespace spu::mpc::cheetah {

class SilentVole {
  public:
  SilentVole(std::shared_ptr<yacl::link::Context> link_ctx,
             psi::PsiRoleType role);

  std::vector<uint64_t> Send(size_t n);

  uint64_t Delta() const { return delta_; }

  std::array<std::vector<uint64_t>, 2> Recv(size_t n);

  private:
  yacl::crypto::CodeType code_type_ = yacl::crypto::CodeType::Silver11;
  psi::PsiRoleType role_;
  std::shared_ptr<yacl::link::Context> link_ctx_;
  std::shared_ptr<yacl::crypto::SilentVoleSender> vole_send_;
  std::shared_ptr<yacl::crypto::SilentVoleReceiver> vole_recv_;

  uint64_t delta_ = 0;
};

class VoleCPsi {
  public:
  VoleCPsi(std::shared_ptr<yacl::link::Context> link_ctx,
           psi::PsiRoleType role);

  VoleCPsi(std::shared_ptr<yacl::link::Context> link_ctx,
           std::shared_ptr<cheetah::BasicOTProtocols> base_ot,
           psi::PsiRoleType role);

  std::shared_ptr<yacl::link::Context> get_link_context() const {
    return link_ctx_;
  }

  std::vector<uint8_t> RunCPsi(absl::Span<const std::string> items);

  std::vector<uint64_t> RunOprf(const std::vector<std::string> &items);

  std::vector<uint64_t> getItemIndex(size_t index) const {
    std::vector<uint64_t> res;
    for (auto &bin : bins_[index]) {
      res.push_back(bin.InputIdx());
    }
    return res;
  }

  template <typename T>
  std::vector<T> adjustOrder(const std::vector<T> &items) const {
    std::vector<T> bin_values(bins_.size(), 0);
    for (size_t idx = 0; idx < bins_.size(); idx++) {
      auto item_idx = getItemIndex(idx);
      SPU_ENFORCE(item_idx.size() <= 1, "only support cuckoo hash bin");
      if (!item_idx.empty()) {
        bin_values[idx] = items[item_idx[0]];
      }
    }
    return bin_values;
  }

  uint64_t getNumBin() const { return bins_.size(); }

  psi::PsiRoleType getRole() const { return role_; }

  std::vector<uint64_t> OpprfShareSender(const std::vector<uint64_t> &data,
                                         bool boolShare = true);

  std::vector<uint64_t> OpprfShareReceiver(bool boolShare = true);

  std::vector<uint8_t> equal64(const std::vector<uint64_t> &arr);

  std::vector<uint64_t> bmultiplexer(const std::vector<uint64_t> &arr,
                                     const std::vector<uint8_t> &bools,
                                     int zeros = 0);

  std::vector<uint64_t> multiplexer(const std::vector<uint64_t> &arr,
                                    const std::vector<uint8_t> &bools,
                                    int zeros = 0);

  void OpprfSharePrevProc();

  std::vector<uint64_t> OpprfSharePostProc(const std::vector<uint64_t> &res,
                                           bool bool_share);

  yacl::Buffer OpprfShareSendProc(std::vector<uint64_t> &data, bool bool_share);

  std::vector<uint64_t> OpprfShareRecvProc(yacl::Buffer &gc_table_buffer);

  private:
  // exchange items number, compute compare bytes size
  void ExchangeSetSize(size_t self_item_num);

  // mBaRK-OPRF  sender/receiver
  std::vector<uint64_t> RunmBaRKOprfSender(absl::Span<const std::string> items,
                                           size_t compare_bytes_size);

  std::vector<uint64_t> RunmBaRKOprfReceiver(
      absl::Span<const std::string> items, size_t compare_bytes_size);

  void OkvsEncode(const std::vector<uint64_t> &oprfs);

  void OkvsDecode(const std::vector<uint64_t> &oprfs);

  void updateBinValues();

  // vole
  // std::unique_ptr<psi::WolverineVole> vole_;
  std::shared_ptr<SilentVole> vole_;

  // cuckoo_options
  psi::CuckooIndex::Options cuckoo_options_;

  std::vector<std::vector<psi::CuckooIndex::Bin>> bins_;
  std::vector<uint64_t> items_hash_low64_;

  std::vector<uint64_t> bin_values_;

  std::unique_ptr<cheetah::PaxosOKVS> okvs_ptr_;

  // Provides the link for the rank world.
  std::shared_ptr<yacl::link::Context> link_ctx_;

  std::shared_ptr<cheetah::BasicOTProtocols> base_ot_;

  std::vector<uint64_t> oprfs_;

  std::vector<uint8_t> masks_;

  // psi role sender/receiver
  psi::PsiRoleType role_;

  // batch send/recv data size
  size_t batch_size_;

  // peer's item size
  size_t peer_items_num_ = 0;
};
};  // namespace spu::mpc::cheetah

#endif  // SPU_VOLE_CPSI_H
