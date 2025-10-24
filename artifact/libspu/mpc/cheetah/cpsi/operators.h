

#ifndef SPU_OPERATORS_H
#define SPU_OPERATORS_H

#include "yacl/link/context.h"
#include "psi/utils/communication.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/core/context.h"
#include <memory>

namespace libpsi = psi;
namespace spu::mpc::cheetah {
    class Operators {

    public :
        Operators(const std::shared_ptr<yacl::link::Context> &link_ctx, psi::PsiRoleType role);

        std::shared_ptr<yacl::link::Context> get_link_context() const {
            return kectx_ptr_.get()->lctx();
        }

        NdArrayRef from(const std::vector<uint64_t> &aa, size_t cols = 0);

        NdArrayRef from(const std::vector<uint8_t> &aa, size_t cols = 0);

        std::vector<uint64_t> to(const NdArrayRef &aa);

        NdArrayRef from(const std::vector<uint64_t> &aa, const Shape &shape);

        NdArrayRef from(uint64_t aa, const Shape &shape);

        NdArrayRef encodeD(const std::vector<double> &aa, const Shape &shape, size_t log2_scale);

        NdArrayRef encodeD(double aa, const Shape &shape, size_t log2_scale);

        NdArrayRef AddAA(const NdArrayRef &aa, const NdArrayRef &bb);

        NdArrayRef AddAP(const NdArrayRef &aa, const NdArrayRef &bb);

        NdArrayRef SubAA(const NdArrayRef &aa, const NdArrayRef &bb);

        NdArrayRef SubPA(const NdArrayRef &aa, const NdArrayRef &bb);

        NdArrayRef MulAA(const NdArrayRef &aa, const NdArrayRef &bb);

        NdArrayRef MulAP(const NdArrayRef &aa, const NdArrayRef &bb);

        NdArrayRef SquareA(const NdArrayRef &aa);

        NdArrayRef TruncA(const NdArrayRef &aa, size_t shift, SignType sign);

        NdArrayRef NegA(const NdArrayRef &aa);

        NdArrayRef AbsA(const NdArrayRef &aa);

        NdArrayRef MultiplexA(const NdArrayRef &bb, const NdArrayRef &aa);

        NdArrayRef MuxA(const NdArrayRef &mm, const NdArrayRef &aa, const NdArrayRef &bb);

        NdArrayRef A2B(const NdArrayRef &aa);

        NdArrayRef B2A(const NdArrayRef &bb);

        NdArrayRef asBShares(const NdArrayRef &aa, size_t packed_bits = 0);

        NdArrayRef asAShares(const NdArrayRef &aa);

        NdArrayRef MulAB(const NdArrayRef &aa, const NdArrayRef &bb, size_t shift, bool out_of_range = false);

        NdArrayRef LShiftB(const NdArrayRef &bb, size_t shift);

        NdArrayRef RShiftB(const NdArrayRef &bb, size_t shift);

        NdArrayRef AndBB(const NdArrayRef &b0, const NdArrayRef &b1);

        NdArrayRef AndBP(const NdArrayRef &b0, const NdArrayRef &b1);

        NdArrayRef OrBB(const NdArrayRef &b0, const NdArrayRef &b1);

        NdArrayRef XorBB(const NdArrayRef &b0, const NdArrayRef &b1);

        NdArrayRef XorBP(const NdArrayRef &b0, const NdArrayRef &b1);

        NdArrayRef MSB(const NdArrayRef &aa, size_t nbits=64);

        NdArrayRef MSBSigmoid(const NdArrayRef &aa, uint64_t low_seg, uint64_t high_seg, size_t skip_bits, size_t nbits=64);

        NdArrayRef HighestOneBit(const NdArrayRef &b_abs);

        NdArrayRef BitRevB(const NdArrayRef &bb, size_t start, size_t end);

        NdArrayRef MultiplexB(const NdArrayRef &ss, const NdArrayRef &bb);

        NdArrayRef DivAA(const NdArrayRef &aa, const NdArrayRef &bb, size_t log2_scale);

        NdArrayRef ReciprocalP(const NdArrayRef &aa, size_t log2_scale);

        NdArrayRef ReciprocalA(const NdArrayRef &aa, size_t log2_scale, SignType sign = SignType::Unknown);

        NdArrayRef PolynomialA(const NdArrayRef &aa, const std::vector<double> &coeffs, size_t log2_scale,
                               SignType sign = SignType::Unknown);

        NdArrayRef RSqrtA(const NdArrayRef &aa, size_t log2_scale);

        NdArrayRef ArgMax(const NdArrayRef &aa);

        NdArrayRef RecoveryA(const NdArrayRef &aa);

        NdArrayRef RecoveryB(const NdArrayRef &bb);

        psi::PsiRoleType getRole() const {
            return role_;
        }

        SPUContext *getContext() const {
            return kectx_ptr_->sctx();
        }

    private:
        NdArrayRef ReciprocalA_Positive(const NdArrayRef &aa, size_t log2_scale, size_t fxp_num_iters);

        std::array<NdArrayRef, 2> MultiplexA(const NdArrayRef &mm, const std::array<NdArrayRef, 2> &aa);

        std::array<NdArrayRef, 2> MuxA(const NdArrayRef &mm, const std::array<NdArrayRef, 2> &aa,
                                       const std::array<NdArrayRef, 2> &bb);

        // separate even and odd bits. e.g.
        //   xAyBzCwD -> xyzwABCD
        NdArrayRef BitDeintlB(const NdArrayRef &aa);

        NdArrayRef BitParityB(const NdArrayRef &bb, size_t bits);


        std::unique_ptr<spu::SPUContext> sctx_ptr_;
        std::unique_ptr<spu::KernelEvalContext> kectx_ptr_;
        psi::PsiRoleType role_;
    };
}

#endif //SPU_OPERATORS_H
