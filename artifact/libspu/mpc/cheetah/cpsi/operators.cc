

#include "libspu/mpc/cheetah/cpsi/operators.h"

#include "yacl/crypto/rand/rand.h"

#include "libspu/core/config.h"
#include "libspu/core/type_util.h"
#include "libspu/mpc/cheetah/arithmetic.h"
#include "libspu/mpc/cheetah/boolean.h"
#include "libspu/mpc/cheetah/conversion.h"
#include "libspu/mpc/cheetah/protocol.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/pv2k.h"
#include "libspu/mpc/factory.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/kernel/hal/fxp_approx.h"

namespace spu::mpc::cheetah {

using namespace std;

//    SPUContext makeSPUContext(ProtocolKind prot_kind, FieldType field,
//                              const std::shared_ptr<yacl::link::Context>&
//                              lctx) {
//        RuntimeConfig config;
//        config.set_protocol(prot_kind);
//        config.set_field(field);
//        config.set_enable_action_trace(false);
//
//        populateRuntimeConfig(config);
//
//        SPUContext ctx(config, lctx);
//        mpc::Factory::RegisterProtocol(&ctx, lctx);
//
//        return ctx;
//    }

Operators::Operators(const std::shared_ptr<yacl::link::Context> &link_ctx,
                     psi::PsiRoleType role)
    : role_(role) {
  RuntimeConfig config;

  config.set_protocol(ProtocolKind::CHEETAH);
  config.set_field(FieldType::FM64);
  config.set_enable_action_trace(false);
  config.set_fxp_fraction_bits(20);
  config.set_fxp_div_goldschmidt_iters(10);
  config.set_max_concurrency(1);
  populateRuntimeConfig(config);

  sctx_ptr_ = std::make_unique<spu::SPUContext>(config, link_ctx);
  regCheetahProtocol(sctx_ptr_.get(), link_ctx);
  kectx_ptr_ = std::make_unique<spu::KernelEvalContext>(sctx_ptr_.get());
}

NdArrayRef Operators::from(const std::vector<uint64_t> &aa, size_t cols) {
  auto total_num = static_cast<int64_t>(aa.size());
  auto each_num = static_cast<int64_t>(cols);
  Shape shape = {total_num};
  if (each_num != 0) {
    shape = {total_num / each_num, each_num};
    SPU_ENFORCE(shape.numel() == total_num);
  }

  const FieldType field =
      kectx_ptr_.get()->getState<Z2kState>()->getDefaultField();
  NdArrayRef vv = ring_zeros(field, shape);
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto oup = NdArrayView<ring2k_t>(vv);
    pforeach(0, oup.numel(), [&](int64_t i) { oup[i] = aa[i]; });
  });
  return vv;
}

NdArrayRef Operators::from(const std::vector<uint8_t> &aa, size_t cols) {
  auto total_num = static_cast<int64_t>(aa.size());
  auto each_num = static_cast<int64_t>(cols);
  Shape shape = {total_num};
  if (each_num != 0) {
    shape = {total_num / each_num, each_num};
    SPU_ENFORCE(shape.numel() == total_num);
  }

  const FieldType field =
      kectx_ptr_.get()->getState<Z2kState>()->getDefaultField();
  NdArrayRef vv = ring_zeros(field, shape);
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto oup = NdArrayView<ring2k_t>(vv);
    pforeach(0, oup.numel(), [&](int64_t i) { oup[i] = aa[i]; });
  });
  return vv;
}

std::vector<uint64_t> Operators::to(const NdArrayRef &aa) {
  std::vector<uint64_t> res(aa.numel());
  const FieldType field =
      kectx_ptr_.get()->getState<Z2kState>()->getDefaultField();
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto inp = NdArrayView<ring2k_t>(aa);
    pforeach(0, inp.numel(), [&](int64_t i) { res[i] = inp[i]; });
  });

  return res;
}

NdArrayRef Operators::from(const std::vector<uint64_t> &aa, const Shape &shape) {
    const FieldType field = kectx_ptr_.get()->getState<Z2kState>()->getDefaultField();
    NdArrayRef res = ring_zeros(field, shape);

    SPU_ENFORCE(static_cast<size_t>(res.numel()) == aa.size());

    DISPATCH_ALL_FIELDS(field, "_", [&]() {
        auto oup = NdArrayView<ring2k_t>(res);
        pforeach(0, oup.numel(),[&](int64_t i) {
            oup[i] = aa[i];
        });
    });

    return res;
}

NdArrayRef Operators::from(uint64_t aa, const Shape &shape) {
    const FieldType field = kectx_ptr_.get()->getState<Z2kState>()->getDefaultField();
    NdArrayRef res = ring_zeros(field, shape);

    DISPATCH_ALL_FIELDS(field, "_", [&]() {
        auto oup = NdArrayView<ring2k_t>(res);
        pforeach(0, oup.numel(),[&](int64_t i) {
            oup[i] = aa;
        });
    });

    return res;
}

NdArrayRef Operators::encodeD(const std::vector<double> &aa, const Shape &shape,
                              size_t log2_scale) {
  const FieldType field =
      kectx_ptr_.get()->getState<Z2kState>()->getDefaultField();
  NdArrayRef res = ring_zeros(field, shape);

  double scale = std::pow(2, log2_scale);

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto oup = NdArrayView<ring2k_t>(res);
    pforeach(0, oup.numel(), [&](int64_t i) {
      SPU_ENFORCE(static_cast<int64_t>(aa.size()) == oup.numel());
      auto vv = static_cast<int64_t>(std::round(scale * aa[i]));
      oup[i] = static_cast<uint64_t>(vv);
    });
  });

  return res;
}

NdArrayRef Operators::encodeD(double aa, const Shape &shape,
                              size_t log2_scale) {
  const FieldType field =
      kectx_ptr_.get()->getState<Z2kState>()->getDefaultField();
  NdArrayRef res = ring_zeros(field, shape);

  double scale = std::pow(2, log2_scale);

  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto oup = NdArrayView<ring2k_t>(res);
    pforeach(0, oup.numel(), [&](int64_t i) {
      auto vv = static_cast<int64_t>(std::round(scale * aa));
      oup[i] = static_cast<uint64_t>(vv);
    });
  });

  return res;
}

NdArrayRef Operators::AddAA(const NdArrayRef &aa, const NdArrayRef &bb) {
  return cheetah::AddAA().proc(kectx_ptr_.get(), aa, bb);
}

NdArrayRef Operators::AddAP(const NdArrayRef &aa, const NdArrayRef &bb) {
  return cheetah::AddAP().proc(kectx_ptr_.get(), aa, bb);
}

NdArrayRef Operators::SubAA(const NdArrayRef &aa, const NdArrayRef &bb) {
  SPU_ENFORCE(aa.shape() == bb.shape());
  return ring_sub(aa, bb);
}

NdArrayRef Operators::SubPA(const NdArrayRef &aa, const NdArrayRef &bb) {
  SPU_ENFORCE(aa.shape() == bb.shape());
  auto *comm = kectx_ptr_.get()->getState<Communicator>();

  if (comm->getRank() == 0) {
    return ring_sub(aa, bb).as(bb.eltype());
  } else {
    return ring_neg(bb);
  }
}

NdArrayRef Operators::MulAA(const NdArrayRef &aa, const NdArrayRef &bb) {
  return cheetah::MulAA().proc(kectx_ptr_.get(), aa, bb);
}

NdArrayRef Operators::MulAP(const NdArrayRef &aa, const NdArrayRef &bb) {
  return cheetah::MulAP().proc(kectx_ptr_.get(), aa, bb);
}

NdArrayRef Operators::SquareA(const NdArrayRef &aa) {
  return cheetah::SquareA().proc(kectx_ptr_.get(), aa);
}

NdArrayRef Operators::TruncA(const NdArrayRef &aa, size_t shift,
                             SignType sign) {
  return cheetah::TruncA().proc(kectx_ptr_.get(), aa, shift, sign);
}

NdArrayRef Operators::NegA(const NdArrayRef &aa) {
  return cheetah::NotA().proc(kectx_ptr_.get(), aa);
}

NdArrayRef Operators::AbsA(const NdArrayRef &aa) {
  auto mm = cheetah::MsbA2B().proc(kectx_ptr_.get(), aa);
  return MuxA(mm, aa, NegA(aa));
}

NdArrayRef Operators::MuxA(const NdArrayRef &mm, const NdArrayRef &aa,
                           const NdArrayRef &bb) {
  auto cc = cheetah::MulA1B().proc(kectx_ptr_.get(), SubAA(bb, aa), mm);
  return AddAA(aa, cc);
}

NdArrayRef Operators::MultiplexA(const NdArrayRef &mm, const NdArrayRef &vv) {
  return MulA1B().proc(kectx_ptr_.get(), vv, mm);
};

NdArrayRef Operators::A2B(const NdArrayRef &aa) {
  return cheetah::A2B().proc(kectx_ptr_.get(), aa);
}

NdArrayRef Operators::B2A(const NdArrayRef &bb) {
  return cheetah::B2A().proc(kectx_ptr_.get(), bb);
}

NdArrayRef Operators::asBShares(const NdArrayRef &aa, size_t packed_bits) {
  const FieldType field =
      kectx_ptr_.get()->getState<Z2kState>()->getDefaultField();
  packed_bits = packed_bits == 0 ? SizeOf(field) * 8 : packed_bits;
  return aa.as(makeType<BShrTy>(field, packed_bits), true);
}

NdArrayRef Operators::asAShares(const NdArrayRef &aa) {
  const FieldType field =
      kectx_ptr_.get()->getState<Z2kState>()->getDefaultField();
  return aa.as(makeType<AShrTy>(field), true);
}

NdArrayRef Operators::MulAB(const NdArrayRef &aa, const NdArrayRef &bb,
                            size_t shift, bool out_of_range) {
  auto ctx = kectx_ptr_.get();
  SPU_ENFORCE_EQ(aa.shape(), bb.shape());
  const int64_t numel = aa.numel();
  NdArrayRef out(aa.eltype(), aa.shape());

  if (numel == 0) {
    return out;
  }

  const FieldType field = ctx->getState<Z2kState>()->getDefaultField();
  const auto nbits = static_cast<int64_t>(SizeOf(field) * 8);

  NdArrayRef hh(bb.eltype(), bb.shape());
  NdArrayRef ll(bb.eltype(), bb.shape());

    if (!out_of_range || nbits <= static_cast<int64_t>(shift)) {
        DISPATCH_ALL_FIELDS(field, "split_bits", [&]() {
            auto vals = NdArrayView<ring2k_t>(bb);
            auto subl = NdArrayView<ring2k_t>(ll);

            pforeach(0, subl.numel(), [&](int64_t i) {
                subl[i] = vals[i];
            });
        });

        ll = B2A(ll);
        ll = MulAA(aa, ll);
        return TruncA(ll, shift, SignType::Positive);
    }

  DISPATCH_ALL_FIELDS(field, "split_bits", [&]() {
    auto vals = NdArrayView<ring2k_t>(bb);
    auto subh = NdArrayView<ring2k_t>(hh);
    auto subl = NdArrayView<ring2k_t>(ll);

    pforeach(0, subh.numel(), [&](int64_t i) {
      subh[i] = vals[i] >> shift;
      subl[i] = vals[i] & ((1ULL << shift) - 1ULL);
    });
  });

  hh = asBShares(hh, nbits - shift);
  ll = asBShares(ll, shift);

  hh = B2A(hh);
  hh = MulAA(aa, hh);

  ll = B2A(ll);
  ll = MulAA(aa, ll);
  ll = TruncA(ll, shift, SignType::Positive);
  out = AddAA(hh, ll);
  return out;
}

NdArrayRef Operators::LShiftB(const NdArrayRef &bb, size_t shift) {
  return cheetah::LShiftB().proc(kectx_ptr_.get(), bb, shift);
}

NdArrayRef Operators::RShiftB(const NdArrayRef &bb, size_t shift) {
  return cheetah::RShiftB().proc(kectx_ptr_.get(), bb, shift);
}

NdArrayRef Operators::AndBB(const NdArrayRef &b0, const NdArrayRef &b1) {
    return cheetah::AndBB().proc(kectx_ptr_.get(), b0, b1);
}

NdArrayRef Operators::AndBP(const NdArrayRef &b0, const NdArrayRef &b1) {
    return cheetah::AndBP().proc(kectx_ptr_.get(), b0, b1);
}

NdArrayRef Operators::OrBB(const NdArrayRef &b0, const NdArrayRef &b1) {
  return cheetah::OrBB().proc(kectx_ptr_.get(), b0, b1);
}

NdArrayRef Operators::XorBB(const NdArrayRef &b0, const NdArrayRef &b1) {
  return cheetah::XorBB().proc(kectx_ptr_.get(), b0, b1);
}

NdArrayRef Operators::XorBP(const NdArrayRef &b0, const NdArrayRef &b1) {
    return cheetah::XorBP().proc(kectx_ptr_.get(), b0, b1);
}

NdArrayRef Operators::BitRevB(const NdArrayRef &bb, size_t start, size_t end) {
  return cheetah::BitrevB().proc(kectx_ptr_.get(), bb, start, end);
}

NdArrayRef Operators::MultiplexB(const NdArrayRef &ss, const NdArrayRef &bb) {
    const auto *shareType = ss.eltype().as<BShrTy>();
            SPU_ENFORCE_EQ(shareType->nbits(), 1UL);

    size_t nbits = bb.eltype().as<BShrTy>()->nbits();

    NdArrayRef select = ss;
    for (int idx = 0; idx < Log2Ceil(nbits); idx++) {
        const size_t offset = 1UL << idx;
        auto temp = LShiftB(select, offset);
        select = XorBB(select, temp);
    }

    return AndBB(select, bb);
}

NdArrayRef Operators::MSB(const NdArrayRef &aa, size_t nbits) {
  return cheetah::MsbA2B(nbits).proc(kectx_ptr_.get(), aa);
}

NdArrayRef Operators::MSBSigmoid(const NdArrayRef &aa, uint64_t low_seg,
                                 uint64_t high_seg, size_t skip_bits, size_t nbits) {
  return cheetah::MsbA2BSigmoid(nbits, low_seg, high_seg, skip_bits)
      .proc(kectx_ptr_.get(), aa);
}

//NdArrayRef Operators::DivAA(const NdArrayRef &aa, const NdArrayRef &bb,
//                            size_t log2_scale) {
//  size_t fxp_goldschmidt_iters = 30;
//
//  auto is_aa_neg = MSB(aa);
//  auto is_bb_neg = MSB(bb);
//
//  auto aa_abs = MuxA(is_aa_neg, aa, NegA(aa));
//  auto bb_abs = MuxA(is_bb_neg, bb, NegA(bb));
//
//  auto bb_msb = HighestOneBit(bb_abs);
//
//  auto factor_bb = BitRevB(bb_msb, 0, 2 * log2_scale);
//  auto cc = MulAB(bb_abs, factor_bb, log2_scale);
//
//  auto k2_9142 = encodeD(2.9142F, cc.shape(), log2_scale);
//  auto rr = SubPA(k2_9142, AddAA(cc, cc));
//
//  auto ee = MulAA(rr, cc);
//  ee = TruncA(ee, log2_scale, SignType::Positive);
//  auto k1 = encodeD(1.0F, ee.shape(), log2_scale);
//  ee = SubPA(k1, ee);
//
//  for (size_t itr = 0; itr < fxp_goldschmidt_iters; itr++) {
//    rr = MulAA(rr, AddAP(ee, k1));
//    rr = TruncA(rr, log2_scale, SignType::Positive);
//
//    if (itr + 1 < fxp_goldschmidt_iters) {
//      ee = SquareA(ee);
//      ee = TruncA(ee, log2_scale, SignType::Positive);
//    }
//  }
//
//  rr = MulAA(aa_abs, rr);
//  rr = TruncA(rr, log2_scale, SignType::Positive);
//  rr = MulAB(rr, factor_bb, log2_scale);
//
//  return MuxA(XorBB(is_aa_neg, is_bb_neg), rr, NegA(rr));
//}

NdArrayRef Operators::DivAA(const NdArrayRef &aa, const NdArrayRef &bb, size_t log2_scale) {
    auto rr = ReciprocalA(bb, log2_scale);
    rr = MulAA(aa, rr);
    return TruncA(rr, log2_scale, SignType::Unknown);
}

NdArrayRef Operators::ReciprocalP(const NdArrayRef &aa, size_t log2_scale) {
  const auto field = kectx_ptr_.get()->getState<Z2kState>()->getDefaultField();
  NdArrayRef pp(aa.eltype(), aa.shape());

  double scale = std::pow(2, log2_scale);
  DISPATCH_ALL_FIELDS(field, "_", [&]() {
    auto inp = NdArrayView<ring2k_t>(aa);
    auto oup = NdArrayView<ring2k_t>(pp);
    pforeach(0, inp.numel(), [&](int64_t i) {
      double ww = static_cast<int64_t>(inp[i]) / scale;
      auto vv = static_cast<int64_t>(std::round(scale / ww));
      oup[i] = static_cast<uint64_t>(vv);
    });
  });

  return pp;
}

NdArrayRef Operators::ReciprocalA(const NdArrayRef &bb, size_t log2_scale,
                                  SignType sign) {
    size_t fxp_goldschmidt_iters = sctx_ptr_->config().fxp_div_goldschmidt_iters();

    auto HighestOneBitForUnknownSign = [&](const NdArrayRef &xx) -> std::array<NdArrayRef, 2> {
        const size_t offset = 10;
        const size_t bit_width = SizeOf(kectx_ptr_.get()->getState<Z2kState>()->getDefaultField()) * 8;

        auto x0 = MulAP(xx, from(1ULL << offset, xx.shape()));
        x0 = A2B(x0);

        auto b0 = RShiftB(x0, bit_width - 1);

        NdArrayRef p0 = b0;
        for (int idx = 0; idx < absl::bit_width(bit_width) - 1; idx++) {
            auto pp = LShiftB(p0, 1UL << idx);
            p0 = XorBB(p0, pp);
        }

        x0 = XorBB(x0, p0);
        x0 = RShiftB(x0, offset);
        for (int idx = 0; idx < absl::bit_width(bit_width) - 1; idx++) {
            auto x1 = RShiftB(x0, 1UL << idx);
            x0 = OrBB(x0, x1);
        }

        auto x1 = RShiftB(x0, 1);
        return {asBShares(b0, 1), XorBB(x0, x1)};
    };

    NdArrayRef rr;
    switch (sign) {
        case SignType::Positive:
            rr = ReciprocalA_Positive(bb, log2_scale, fxp_goldschmidt_iters);
            break;
        case SignType::Negative:
            rr = ReciprocalA_Positive(NegA(bb), log2_scale, fxp_goldschmidt_iters);
            rr = NegA(rr);
            break;
        default:
            auto [neg_bb, bb_msb] = HighestOneBitForUnknownSign(bb);
            auto bb_abs = MuxA(neg_bb, bb, NegA(bb));

            auto factor_bb = BitRevB(bb_msb, 0, 2 * log2_scale);
            auto cc = MulAB(bb_abs, factor_bb, log2_scale);

            auto k2_9142 = encodeD(2.9142F, cc.shape(), log2_scale);
            rr = SubPA(k2_9142, AddAA(cc, cc));

            auto ee = MulAA(rr, cc);
            ee = TruncA(ee, log2_scale, SignType::Positive);
            auto k1 = encodeD(1.0F, ee.shape(), log2_scale);
            ee = SubPA(k1, ee);

            for (size_t itr = 0; itr < fxp_goldschmidt_iters; itr++) {
                rr = MulAA(rr, AddAP(ee, k1));
                rr = TruncA(rr, log2_scale, SignType::Positive);

                if (itr + 1 < fxp_goldschmidt_iters) {
                    ee = SquareA(ee);
                    ee = TruncA(ee, log2_scale, SignType::Positive);
                }
            }

            rr = MulAB(rr, factor_bb, log2_scale);
            rr = MuxA(neg_bb, rr, NegA(rr));
    }
    return rr;
}

//    NdArrayRef Operators::ArgMax(const NdArrayRef &aa) {
//        const auto field =
//        kectx_ptr_.get()->getState<Z2kState>()->getDefaultField(); int64_t
//        size = aa.numel();
//
//        NdArrayRef bb(aa.eltype(), {size});
//        bb.copy_slice(aa, {0}, {0}, size);
//
//        NdArrayRef idx = ring_zeros(field, {size});
//        if (kectx_ptr_.get()->getState<Communicator>()->getRank() == 0) {
//            DISPATCH_ALL_FIELDS(field, "_", [&]() {
//                auto xinp = NdArrayView<ring2k_t>(idx);
//                pforeach(0, xinp.numel(), [&](int64_t i) {
//                    xinp[i] = i;
//                });
//            });
//        }
//        idx = asAShares(idx);
//
//        while (size > 1) {
//            size = (size + 1) / 2;
//            NdArrayRef hi(bb.eltype(), {size});
//            hi.copy_slice(bb, {0}, {0}, hi.numel());
//            NdArrayRef lo = hi.clone();
//            lo.copy_slice(bb, {hi.numel()}, {0}, bb.numel() - hi.numel());
//
//            NdArrayRef hi_idx(idx.eltype(), {size});
//            hi_idx.copy_slice(idx, {0}, {0}, hi_idx.numel());
//
//            NdArrayRef lo_idx = hi_idx.clone();
//            lo_idx.copy_slice(idx, {hi_idx.numel()}, {0}, idx.numel() -
//            hi_idx.numel());
//
//            auto is_negative = MSB(SubAA(hi, lo));
//            bb  = MuxA(is_negative, hi, lo);
//            idx = MuxA(is_negative, hi_idx, lo_idx);
//        }
//
//        return idx;
//    }

std::array<NdArrayRef, 2> Operators::MultiplexA(
    const NdArrayRef &select, const std::array<NdArrayRef, 2> &msgs) {
  SPU_ENFORCE_EQ(msgs[0].shape(), select.shape());
  SPU_ENFORCE_EQ(msgs[1].shape(), select.shape());
  const auto *shareType = select.eltype().as<BShrTy>();
  SPU_ENFORCE_EQ(shareType->nbits(), 1UL);

  const auto field = msgs[0].eltype().as<Ring2k>()->field();
  const int64_t size = msgs[0].numel();

  SPU_ENFORCE_EQ(field, FieldType::FM64);

  auto *comm = kectx_ptr_->getState<Communicator>();
  auto *ot_state = kectx_ptr_->getState<cheetah::CheetahOTState>();
  ot_state->LazyInit(comm, 0);

  auto ferret_ot = ot_state->get(0);
  auto sender = ferret_ot->GetSenderCOT();
  auto receiver = ferret_ot->GetReceiverCOT();

  const auto mask64 = static_cast<uint64_t>(-1);

  // Compute (b0 ^ b1) ? (x0 + x1) : 0
  std::vector<uint128_t> rand0(size);
  std::vector<uint128_t> rand1(size);
  std::vector<uint128_t> recv(size);
  std::vector<uint128_t> corr(size);
  std::vector<uint8_t> sel(size);

  auto res0 = ring_zeros(field, msgs[0].shape());
  auto res1 = ring_zeros(field, msgs[1].shape());

  DISPATCH_ALL_FIELDS(field, "Multiplexer", [&]() {
    NdArrayView<const ring2k_t> _msg0(msgs[0]);
    NdArrayView<const ring2k_t> _msg1(msgs[1]);
    NdArrayView<const ring2k_t> _sel(select);

    NdArrayView<ring2k_t> _res0(res0);
    NdArrayView<ring2k_t> _res1(res1);

    pforeach(0, size,
             [&](int64_t i) { sel[i] = static_cast<uint8_t>(_sel[i] & 1); });

    if (role_ == psi::PsiRoleType::Sender) {
      sender->SendRMCC(absl::MakeSpan(rand0), absl::MakeSpan(rand1));
      sender->Flush();

      receiver->RecvRMCC(absl::MakeSpan(sel), absl::MakeSpan(recv));
    } else {
      receiver->RecvRMCC(absl::MakeSpan(sel), absl::MakeSpan(recv));

      sender->SendRMCC(absl::MakeSpan(rand0), absl::MakeSpan(rand1));
      sender->Flush();
    }

    pforeach(0, size, [&](int64_t i) {
      uint64_t high =
          (_msg0[i] * (1 - 2 * sel[i])) - (rand0[i] >> 64U) + (rand1[i] >> 64U);
      uint64_t low = (_msg1[i] * (1 - 2 * sel[i])) - (rand0[i] & mask64) +
                     (rand1[i] & mask64);
      corr[i] = (static_cast<uint128_t>(high) << 64U) | low;
    });

    auto link_ctx = kectx_ptr_.get()->lctx();

    if (role_ == psi::PsiRoleType::Sender) {
      yacl::Buffer buf(corr.data(), corr.size() * sizeof(uint128_t));
      link_ctx->SendAsyncThrottled(
          link_ctx->NextRank(), buf,
          fmt::format("send correlated msg of size {}", buf.size()));

      buf = link_ctx->Recv(link_ctx->NextRank(),
                           fmt::format("recv correlated msg"));
      SPU_ENFORCE(buf.size() == static_cast<int64_t>(size * sizeof(uint128_t)));

      std::memcpy(reinterpret_cast<uint8_t *>(rand1.data()),
                  reinterpret_cast<uint8_t *>(buf.data()), buf.size());
    } else {
      yacl::Buffer buf = link_ctx->Recv(link_ctx->NextRank(),
                                        fmt::format("recv correlated msg"));
      SPU_ENFORCE(buf.size() == static_cast<int64_t>(size * sizeof(uint128_t)));

      std::memcpy(reinterpret_cast<uint8_t *>(rand1.data()),
                  reinterpret_cast<uint8_t *>(buf.data()), buf.size());

      std::memcpy(reinterpret_cast<uint8_t *>(buf.data()),
                  reinterpret_cast<uint8_t *>(corr.data()), buf.size());
      link_ctx->SendAsyncThrottled(
          link_ctx->NextRank(), buf,
          fmt::format("send correlated msg of size {}", buf.size()));
    }

    pforeach(0, size, [&](int64_t i) {
      uint64_t high0 = (_msg0[i] * sel[i]) + (rand0[i] >> 64U);
      uint64_t low0 = (_msg1[i] * sel[i]) + (rand0[i] & mask64);

      uint64_t high1 = (sel[i] * (rand1[i] >> 64U)) - (recv[i] >> 64U);
      uint64_t low1 = (sel[i] * (rand1[i] & mask64)) - (recv[i] & mask64);

      _res0[i] = high0 + high1;
      _res1[i] = low0 + low1;
    });
  });

  res0 = asAShares(res0);
  res1 = asAShares(res1);

  return {res0, res1};
}

std::array<NdArrayRef, 2> Operators::MuxA(const NdArrayRef &mm,
                                          const std::array<NdArrayRef, 2> &aa,
                                          const std::array<NdArrayRef, 2> &bb) {
  auto in0 = SubAA(bb[0], aa[0]);
  auto in1 = SubAA(bb[1], aa[1]);

  auto cc = MultiplexA(mm, {in0, in1});

  cc[0] = AddAA(aa[0], cc[0]);
  cc[1] = AddAA(aa[1], cc[1]);

  return {cc[0], cc[1]};
}

NdArrayRef Operators::ArgMax(const NdArrayRef &aa) {
  const auto field = kectx_ptr_.get()->getState<Z2kState>()->getDefaultField();
  Shape shape = aa.shape();

  NdArrayRef bb(aa.eltype(), shape);
  bb.copy_slice(aa, {0, 0}, {0, 0}, aa.numel());

  NdArrayRef idx = ring_zeros(field, shape);
  if (kectx_ptr_.get()->getState<Communicator>()->getRank() == 0) {
    DISPATCH_ALL_FIELDS(field, "_", [&]() {
      auto xinp = NdArrayView<ring2k_t>(idx);
      pforeach(0, idx.shape()[0], [&](int64_t i) {
        pforeach(0, idx.shape()[1],
                 [&](int64_t j) { xinp[i * idx.shape()[1] + j] = j; });
      });
    });
  }
  idx = asAShares(idx);

  int64_t cols = bb.shape()[1];
  while (cols > 1) {
    cols = (cols + 1) / 2;
    NdArrayRef hi(bb.eltype(), {bb.shape()[0], cols});
    NdArrayRef lo(bb.eltype(), {bb.shape()[0], cols});

    DISPATCH_ALL_FIELDS(field, "slice_share", [&]() {
      auto xhi = NdArrayView<ring2k_t>(hi);
      auto xlo = NdArrayView<ring2k_t>(lo);
      auto xvv = NdArrayView<ring2k_t>(bb);

      pforeach(0, bb.shape()[0], [&](int64_t i) {
        pforeach(0, cols, [&](int64_t j) {
          xhi[i * cols + j] = xvv[i * bb.shape()[1] + j];
        });

        pforeach(0, bb.shape()[1] - cols, [&](int64_t j) {
          xlo[i * cols + j] = xvv[i * bb.shape()[1] + j + cols];
        });

        if (bb.shape()[1] & 0x1) {
          xlo[i * cols + cols - 1] = xhi[i * cols + cols - 1];
        }
      });
    });

    NdArrayRef hi_idx(idx.eltype(), {idx.shape()[0], cols});
    NdArrayRef lo_idx(idx.eltype(), {idx.shape()[0], cols});

    DISPATCH_ALL_FIELDS(field, "slice_index", [&]() {
      auto xhi = NdArrayView<ring2k_t>(hi_idx);
      auto xlo = NdArrayView<ring2k_t>(lo_idx);
      auto xvv = NdArrayView<ring2k_t>(idx);

      pforeach(0, idx.shape()[0], [&](int64_t i) {
        pforeach(0, cols, [&](int64_t j) {
          xhi[i * cols + j] = xvv[i * idx.shape()[1] + j];
        });

        pforeach(0, idx.shape()[1] - cols, [&](int64_t j) {
          xlo[i * cols + j] = xvv[i * idx.shape()[1] + j + cols];
        });

        if (idx.shape()[1] & 0x1) {
          xlo[i * cols + cols - 1] = xhi[i * cols + cols - 1];
        }
      });
    });

    auto is_negative = MSB(SubAA(hi, lo));
    auto ou = MuxA(is_negative, {hi, hi_idx}, {lo, lo_idx});
    bb = ou[0];
    idx = ou[1];
  }

  return idx;
}

NdArrayRef Operators::HighestOneBit(const NdArrayRef &b_abs) {
  const size_t bit_width =
      SizeOf(kectx_ptr_.get()->getState<Z2kState>()->getDefaultField()) * 8;

  auto b0 = A2B(b_abs);

  for (int idx = 0; idx < absl::bit_width(bit_width) - 1; idx++) {
    const size_t offset = 1UL << idx;
    auto b1 = RShiftB(b0, offset);
    b0 = OrBB(b0, b1);
  }

  auto b1 = RShiftB(b0, 1);
  return XorBB(b0, b1);
}

NdArrayRef Operators::ReciprocalA_Positive(const NdArrayRef &b_abs,
                                           size_t log2_scale,
                                           size_t fxp_goldschmidt_iters) {
  //        const size_t bit_width =
  //        SizeOf(kectx_ptr_.get()->getState<Z2kState>()->getDefaultField()) *
  //        8;
  //
  //        auto b0 = A2B(b_abs);
  //
  //        for (int idx = 0; idx < absl::bit_width(bit_width) - 1; idx++) {
  //            const size_t offset = 1UL << idx;
  //            auto b1 = RShiftB(b0, offset);
  //            b0 = OrBB(b0, b1);
  //        }
  //
  //        auto b1 = RShiftB(b0, 1);
  //        auto b_msb = XorBB(b0, b1);

  auto b_msb = HighestOneBit(b_abs);

  auto factor = BitRevB(b_msb, 0, 2 * log2_scale);
  //        factor = B2A(factor);

  auto c = MulAB(b_abs, factor, log2_scale);
  //        auto c = MulAA(b_abs, factor);
  //        c = TruncA(c, log2_scale, SignType::Positive);

  auto k2_9142 = encodeD(2.9142F, c.shape(), log2_scale);
  auto r = SubPA(k2_9142, AddAA(c, c));

  //        r = MulAA(r, factor);
  //        r = TruncA(r, log2_scale, SignType::Positive);

  auto e = MulAA(r, c);
  e = TruncA(e, log2_scale, SignType::Positive);
  auto k1 = encodeD(1.0F, e.shape(), log2_scale);
  e = SubPA(k1, e);

  for (size_t itr = 0; itr < fxp_goldschmidt_iters; itr++) {
    r = MulAA(r, AddAP(e, k1));
    r = TruncA(r, log2_scale, SignType::Positive);

    if (itr + 1 < fxp_goldschmidt_iters) {
      e = SquareA(e);
      e = TruncA(e, log2_scale, SignType::Positive);
    }
  }

  return MulAB(r, factor, log2_scale);
}

NdArrayRef Operators::PolynomialA(const NdArrayRef &aa, const std::vector<double> &coeffs,
                                  size_t log2_scale, SignType sign) {
    NdArrayRef vv = encodeD(coeffs[0] / 2, aa.shape(), 2*log2_scale);
    NdArrayRef x_pow;
    vv = asAShares(vv);

    for (size_t i = 1; i < coeffs.size(); i++) {
        if (i == 1) {
            x_pow = asAShares(aa);
        } else {
            x_pow = MulAA(x_pow, aa);

            if ((i & 1) == 0U) {
                x_pow = TruncA(x_pow, log2_scale, SignType::Positive);
            } else {
                x_pow = TruncA(x_pow, log2_scale, sign);
            }
        }
        // x_pow = MulAA(x_pow, aa);

        NdArrayRef cc = encodeD(coeffs[i], aa.shape(), log2_scale);
        cc = asAShares(cc);
        cc = MulAP(x_pow, cc);
        vv = AddAA(vv, cc);
    }

    return TruncA(vv, log2_scale, sign);
}

NdArrayRef Operators::RSqrtA(const NdArrayRef &aa, size_t log2_scale) {
//    auto value_aa = spu::Value(aa, DT_F64);
//    auto value_res = kernel::hal::f_rsqrt(sctx_ptr_.get(), value_aa);
//    return value_res.data();

    NdArrayRef zz = HighestOneBit(aa);

    // approx rsqrt for [0.25, 0.5]
    NdArrayRef zz_rev = BitRevB(zz, 0, 2 * log2_scale);
    zz_rev = asBShares(zz_rev, 2 * log2_scale);
    NdArrayRef uu = MulAB(aa, zz_rev, log2_scale);

    std::vector<double> coeffs = {
        2.9390912778777811, -5.526333964374774, 6.908849226372312, -4.5026771451995931, 1.1811597257071498,
    };
    NdArrayRef rr = PolynomialA(uu, coeffs, log2_scale, SignType::Positive);

    // rsqrt for scale=2^{z}
    const auto field = aa.eltype().as<Ring2k>()->field();
    const size_t k = SizeOf(field) * 8;

    NdArrayRef zz_sep = BitDeintlB(zz);

    NdArrayRef lo_mask = asBShares(from((1ULL << (k / 2)) - 1, aa.shape()));
    NdArrayRef zz_even = AndBP(zz_sep, lo_mask);
    NdArrayRef zz_odd = AndBP(RShiftB(zz_sep, k / 2), lo_mask);

    NdArrayRef sa = XorBB(zz_odd, zz_even);
    // sb ^= zz[2*i]
    NdArrayRef sb = BitParityB(zz_even, k / 2);
    sb = sb.as(makeType<BShrTy>(field, 1), true);

    NdArrayRef sa_rev = BitRevB(sa, 0, log2_scale + 1);
    sa_rev = asBShares(sa_rev, log2_scale + 1);

    NdArrayRef c0;
    NdArrayRef c1;
    if ((log2_scale & 0x1) != 0U) {
        uint64_t v0 = static_cast<int64_t>(std::round(std::pow(2, log2_scale / 2) / std::sqrt(2)));
        uint64_t v1 = static_cast<int64_t>(std::pow(2, log2_scale / 2));
        c0 = from(v0, aa.shape());
        c1 = from(v1, aa.shape());
    } else {
        uint64_t v0 = static_cast<int64_t>(std::pow(2, log2_scale / 2 - 1));
        uint64_t v1 = static_cast<int64_t>(std::round(std::pow(2, log2_scale / 2) / std::sqrt(2)));
        c0 = from(v0, aa.shape());
        c1 = from(v1, aa.shape());
    }

    NdArrayRef ss = cheetah::MulA1B().proc(kectx_ptr_.get(), rr, sb);
    ss = AddAA(MulAP(rr, c0), MulAP(ss, SubAA(c1, c0)));
    return MulAB(ss, sa_rev, log2_scale);
}

NdArrayRef Operators::BitDeintlB(const NdArrayRef &bb) {
    NdArrayRef out = bb;

    const auto field = bb.eltype().as<Ring2k>()->field();
    const size_t k = SizeOf(field) * 8;

    for (int64_t idx = 0; idx + 1 < Log2Ceil(k); idx++) {
        NdArrayRef keep = from(detail::kBitIntlKeepMasks[idx], bb.shape());
        NdArrayRef move = from(detail::kBitIntlSwapMasks[idx], bb.shape());

        keep = asBShares(keep);
        move = asBShares(move);

        int64_t shift = 1 << idx;
        // out = (out & keep) ^ ((out >> shift) & move) ^ ((out & move) << shift);
        NdArrayRef t0 = AndBP(out, keep);
        NdArrayRef t1 = AndBP(RShiftB(out, shift), move);
        NdArrayRef t2 = LShiftB(AndBP(out, move), shift);

        out = XorBB(XorBB(t0, t1), t2);
    }
    return out;
}

NdArrayRef Operators::BitParityB(const NdArrayRef &bb, size_t bits) {
    SPU_ENFORCE(absl::has_single_bit(bits), "currently only support power of 2");

    NdArrayRef ret = bb;
    while (bits > 1) {
        ret = XorBB(ret, RShiftB(ret, bits >> 1));
        bits >>= 1;
    }

    ret = AndBP(ret, asBShares(from(1ULL, bb.shape()), bits));
    return ret;
}

NdArrayRef Operators::RecoveryA(const NdArrayRef &aa) {
  return A2P().proc(kectx_ptr_.get(), aa);
}

NdArrayRef Operators::RecoveryB(const NdArrayRef &bb) {
  return B2P().proc(kectx_ptr_.get(), bb);
}

}  // namespace spu::mpc::cheetah
