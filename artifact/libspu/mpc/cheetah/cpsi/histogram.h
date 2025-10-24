

#ifndef SPU_HISTOGRAM_H
#define SPU_HISTOGRAM_H

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "yacl/link/context.h"
#include "psi/utils/communication.h"
#include "libspu/core/type_util.h"

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/rlwe/fast_pack.h"

#include "libspu/mpc/cheetah/nonlinear/compare_prot.h"
#include "libspu/mpc/cheetah/nonlinear/truncate_prot.h"
#include "libspu/mpc/cheetah/arith/cheetah_mul.h"

namespace libpsi = psi;

namespace spu::mpc::cheetah {

    class HistoGram {
    public:

        HistoGram(std::shared_ptr<yacl::link::Context> link_ctx, psi::PsiRoleType role);

        std::shared_ptr<yacl::link::Context> get_link_context() const {
            return link_ctx_;
        }

        void initialize(bool use_ckks = true);

        void set_ckks_params(std::shared_ptr<HistoGram> &histogram_ptr);

        yacl::Buffer serialize(const std::vector<std::vector<uint8_t>> &bytes, bool sameSize = false);

        std::vector<std::vector<uint8_t>> deserialze(yacl::Buffer &buf, size_t sameSize = 0);

        double ComputeHistoGram(std::vector<uint64_t> &gshares, std::vector<uint64_t> &hshares, double scale,
                                const std::vector<std::vector<uint8_t>> &splitBools);

        double ComputeHistoGram(std::vector<std::vector<uint64_t>> &gshare_vec, 
                                std::vector<std::vector<uint64_t>> &hshare_vec, double scale,
                                const std::vector<std::vector<uint8_t>> &splitBools);

        void getSplitBools(const std::vector<std::vector<double>> &features, size_t numBuckets,
                           std::vector<std::vector<uint8_t>> &splitBools, std::vector<std::vector<double>> &splitPoints);


        yacl::Buffer CKKSEncrypt(const std::vector<std::vector<uint64_t>> &gshare_vec,
                                 const std::vector<std::vector<uint64_t>> &hshare_vec,
                                 double scale);

        double HomCompHistoGram(std::vector<std::vector<uint64_t>> &gshare_vec, std::vector<std::vector<uint64_t>> &hshare_vec,
                                yacl::Buffer &buf, double scale,
                                const std::vector<std::vector<uint8_t>> &split_bools);

        double PostProcess(std::vector<std::vector<uint64_t>> &gshare_vec, std::vector<std::vector<uint64_t>> &hshare_vec,
                           yacl::Buffer &buf, size_t cols);

        psi::PsiRoleType getRole() const {
            return role_;
        }

    private:

        std::vector<uint64_t> uint64_to_lwe_mod(const std::vector<uint64_t> &value, uint64_t lwe_mod);

        double CKKSHistoGram(std::vector<uint64_t> &gshares, std::vector<uint64_t> &hshares, double scale,
                             const std::vector<std::vector<uint8_t>> &splitBools);

        double CKKSHistoGram(std::vector<std::vector<uint64_t>> &gshare_vec, 
                             std::vector<std::vector<uint64_t>> &hshare_vec, double scale,
                             const std::vector<std::vector<uint8_t>> &splitBools);


        // Provides the link for the rank world.
        std::shared_ptr<yacl::link::Context> link_ctx_;

//        std::shared_ptr<BasicOTProtocols> base_ot_;
//        std::shared_ptr<CompareProtocol> comp_prot_;
//        std::shared_ptr<TruncateProtocol> trunc_prot_;
//        std::shared_ptr<CheetahMul> mul_ptr_;

        // psi role sender/receiver
        psi::PsiRoleType role_;

        std::shared_ptr<CKKSParams> ckks_ptr0_ = nullptr;
        std::shared_ptr<CKKSParams> ckks_ptr1_ = nullptr;
        seal::KSwitchKeys switch_keys_;

        bool use_ckks_ = true;
    };
};

#endif //SPU_HISTOGRAM_H
