// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/cheetah/nonlinear/compare_sigmoid_prot.h"
#include <cstddef>

#include "yacl/crypto/tools/prg.h"
#include "yacl/link/link.h"

#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/ot/ot_util.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

CompareSigmoidProtocol::CompareSigmoidProtocol(std::shared_ptr<BasicOTProtocols> base,
                                 size_t compare_radix)
    : compare_radix_(compare_radix), basic_ot_prot_(base) {
  SPU_ENFORCE(base != nullptr);
  SPU_ENFORCE(compare_radix_ >= 1 && compare_radix_ <= 8);
  is_sender_ = base->Rank() == 0;

}

CompareSigmoidProtocol::~CompareSigmoidProtocol() { basic_ot_prot_->Flush(); }

void CompareSigmoidProtocol::SetLeafOTMsg(absl::Span<uint8_t> ot_messages, uint8_t digit, uint8_t rnd_cmp_bit,  uint8_t rnd_eq_bit, bool gt) {
  size_t N = ot_messages.size();
  SPU_ENFORCE(digit <= N, fmt::format("N={} got digit={}", N, digit));
  for (size_t i = 0; i < N; i++) {
    if (gt) {
      ot_messages[i] = rnd_cmp_bit ^ static_cast<uint8_t>(digit > i);
    } else {
      ot_messages[i] = rnd_cmp_bit ^ static_cast<uint8_t>(digit < i);
    }

    // compact two bits into one OT message
    ot_messages[i] |= ((rnd_eq_bit ^ static_cast<uint8_t>(digit == i)) << 1);
  }
}

// The Mill protocol from "CrypTFlow2: Practical 2-Party Secure Inference"
// Algorithm 1. REF: https://arxiv.org/pdf/2010.06457.pdf
std::tuple<NdArrayRef, NdArrayRef> CompareSigmoidProtocol::DoCompute(const NdArrayRef& inp_a,const NdArrayRef& inp_b, bool greater_than, int64_t bitwidth, size_t skip_bits) {
    SPU_ENFORCE(inp_a.shape().size() == 1, "need 1D array");
    SPU_ENFORCE(inp_b.shape().size() == 1, "need 1D array");
    auto field = inp_a.eltype().as<Ring2k>()->field();
    int64_t num_digits = CeilDiv(bitwidth, (int64_t)compare_radix_);
    size_t radix = static_cast<size_t>(1) << compare_radix_;  // one-of-N OT
    int64_t num_cmp = inp_a.numel();
    // init to all zero
    std::vector<uint8_t> digits_a(num_cmp * num_digits, 0);
    std::vector<uint8_t> digits_b(num_cmp * num_digits, 0);

    auto skip_digits = static_cast<int64_t>(skip_bits/compare_radix_);

    // Step 1 break into digits \in [0, radix)
    DISPATCH_ALL_FIELDS(field, "break_digits", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const auto mask_radix = makeBitsMask<u2k>(compare_radix_);
    NdArrayView<u2k> xinp_a(inp_a);
    NdArrayView<u2k> xinp_b(inp_b);

    for (int64_t i = 0; i < num_cmp; ++i) {
        for (int64_t j = 0; j < num_digits; ++j) {
            uint32_t shft = j * compare_radix_;
            digits_a[i * num_digits + j] = (xinp_a[i] >> shft) & mask_radix;
            digits_b[i * num_digits + j] = (xinp_b[i] >> shft) & mask_radix;
        }
    }
    });

    std::vector<uint8_t> leaf_cmp_a(num_cmp * num_digits, 0);
    std::vector<uint8_t> leaf_eq_a(num_cmp * num_digits, 0);
    std::vector<uint8_t> leaf_cmp_b_rand(num_cmp * (num_digits-skip_digits), 0);
    std::vector<uint8_t> leaf_eq_b_rand(num_cmp * (num_digits-skip_digits), 0);
    std::vector<uint8_t> leaf_cmp_b(num_cmp * num_digits, 0);
    std::vector<uint8_t> leaf_eq_b(num_cmp * num_digits, 0);

    // Step 2 sample random bits
    yacl::crypto::Prg<uint8_t> prg;
    prg.Fill(absl::MakeSpan(leaf_cmp_a));
    prg.Fill(absl::MakeSpan(leaf_eq_a));
    prg.Fill(absl::MakeSpan(leaf_cmp_b_rand));
    prg.Fill(absl::MakeSpan(leaf_eq_b_rand));

    // convert u8 random to boolean random
    std::transform(leaf_cmp_a.begin(), leaf_cmp_a.end(), leaf_cmp_a.data(),
                    [](uint8_t v) { return v & 1; });
    std::transform(leaf_eq_a.begin(), leaf_eq_a.end(), leaf_eq_a.data(),
                    [](uint8_t v) { return v & 1; });
    for (int64_t i = 0; i < num_cmp; i++) {
      for(int64_t j = 0; j < skip_digits; j++) {
        leaf_cmp_b[i * num_digits + j] = leaf_cmp_a[i * num_digits + j];
        leaf_eq_b[i * num_digits + j] = leaf_eq_a[i * num_digits + j];
      }
      for(int64_t j = skip_digits; j < num_digits; j++) {
        leaf_cmp_b[i * num_digits + j] = leaf_cmp_b_rand[i * (num_digits-skip_digits) + j - skip_digits] & 1;
        leaf_eq_b[i * num_digits + j] = leaf_eq_b_rand[i * (num_digits-skip_digits) + j - skip_digits] & 1;
      }
    }

    // Step 6-7 set the OT messages with two packed bits (one for compare, one
    // for equal)
    std::vector<uint8_t> leaf_ot_msg_a(radix * num_cmp * num_digits, 0);
    std::vector<absl::Span<uint8_t> > each_leaf_ot_msg_a(num_cmp * num_digits);
    for (size_t i = 0; i < each_leaf_ot_msg_a.size(); ++i) {
        each_leaf_ot_msg_a[i] =
            absl::Span<uint8_t>{leaf_ot_msg_a.data() + i * radix, radix};
    }

    std::vector<uint8_t> leaf_ot_msg_b(radix * num_cmp * (num_digits - skip_digits), 0);
    std::vector<absl::Span<uint8_t> > each_leaf_ot_msg_b(num_cmp * (num_digits - skip_digits));
    for (size_t i = 0; i < each_leaf_ot_msg_b.size(); ++i) {
        each_leaf_ot_msg_b[i] =
            absl::Span<uint8_t>{leaf_ot_msg_b.data() + i * radix, radix};
    }

    for (int64_t i = 0; i < num_cmp; ++i) {
        auto* this_ot_msg_a = each_leaf_ot_msg_a.data() + i * num_digits;
        auto* this_ot_msg_b = each_leaf_ot_msg_b.data() + i * (num_digits - skip_digits);
        auto* this_digit_a = digits_a.data() + i * num_digits;
        auto* this_digit_b = digits_b.data() + i * num_digits;
        auto* this_leaf_cmp_a = leaf_cmp_a.data() + i * num_digits;
        auto* this_leaf_eq_a = leaf_eq_a.data() + i * num_digits;
        auto* this_leaf_cmp_b = leaf_cmp_b.data() + i * num_digits;
        auto* this_leaf_eq_b = leaf_eq_b.data() + i * num_digits;

        for (int64_t j = 0; j < num_digits; ++j) {
            uint8_t rnd_cmp_a = this_leaf_cmp_a[j] & 1;
            uint8_t rnd_eq_a = this_leaf_eq_a[j] & 1;
            SetLeafOTMsg(this_ot_msg_a[j], this_digit_a[j], rnd_cmp_a, rnd_eq_a, greater_than);
        }

        for (int64_t j = skip_digits; j < num_digits; ++j) {
            uint8_t rnd_cmp_b = this_leaf_cmp_b[j] & 1;
            uint8_t rnd_eq_b = this_leaf_eq_b[j] & 1;
            SetLeafOTMsg(this_ot_msg_b[j-skip_digits], this_digit_b[j], rnd_cmp_b, rnd_eq_b, greater_than);
        }
    }

    // Step 9: sender of n*M instances of 1-of-N OT
    basic_ot_prot_->GetSenderCOT()->SendCMCC(absl::MakeSpan(leaf_ot_msg_a), radix,
                                                /*bitwidth*/ 2);
    basic_ot_prot_->GetSenderCOT()->Flush();
    basic_ot_prot_->GetSenderCOT()->SendCMCC(absl::MakeSpan(leaf_ot_msg_b), radix,
                                                /*bitwidth*/ 2);
    basic_ot_prot_->GetSenderCOT()->Flush();

    auto boolean_t = makeType<BShrTy>(field, 1);
    NdArrayRef prev_cmp_a = 
        ring_zeros(field, {static_cast<int64_t>(num_digits * num_cmp)})
            .as(boolean_t);
    NdArrayRef prev_eq_a =
        ring_zeros(field, {static_cast<int64_t>(num_digits * num_cmp)})
            .as(boolean_t);
    NdArrayRef prev_cmp_b =
        ring_zeros(field, {static_cast<int64_t>(num_digits * num_cmp)})
            .as(boolean_t);
    NdArrayRef prev_eq_b =
        ring_zeros(field, {static_cast<int64_t>(num_digits * num_cmp)})
            .as(boolean_t);

    DISPATCH_ALL_FIELDS(field, "copy_leaf", [&]() {
      NdArrayView<ring2k_t> xprev_cmp_a(prev_cmp_a);
      NdArrayView<ring2k_t> xprev_eq_a(prev_eq_a);
      NdArrayView<ring2k_t> xprev_cmp_b(prev_cmp_b);
      NdArrayView<ring2k_t> xprev_eq_b(prev_eq_b);
      pforeach(0, prev_cmp_a.numel(), [&](int64_t i) {
          xprev_cmp_a[i] = leaf_cmp_a[i];
          xprev_eq_a[i] = leaf_eq_a[i];
          xprev_cmp_b[i] = leaf_cmp_b[i];
          xprev_eq_b[i] = leaf_eq_b[i];
      });
    });

    auto [gt_a, gt_b] = TraversalAND(prev_cmp_a, prev_eq_a, prev_cmp_b, prev_eq_b, num_cmp, num_digits, skip_digits);
    // auto gt_b = TraversalAND(num_cmp, num_digits).as(boolean_t);
    return {gt_a.as(boolean_t), gt_b.as(boolean_t)};
}


// The Mill protocol from "CrypTFlow2: Practical 2-Party Secure Inference"
// Algorithm 1. REF: https://arxiv.org/pdf/2010.06457.pdf
std::tuple<NdArrayRef, NdArrayRef> CompareSigmoidProtocol::DoCompute(const NdArrayRef& inp, bool greater_than, int64_t bitwidth, size_t skip_bits) {
    SPU_ENFORCE(inp.shape().size() == 1, "need 1D array");
    auto field = inp.eltype().as<Ring2k>()->field();
    int64_t num_digits = CeilDiv(bitwidth, (int64_t)compare_radix_);
    size_t radix = static_cast<size_t>(1) << compare_radix_;  // one-of-N OT
    int64_t num_cmp = inp.numel();
    auto skip_digits = static_cast<int64_t>(skip_bits/compare_radix_);
    // init to all zero
    std::vector<uint8_t> digits_a(num_cmp * num_digits, 0);
    std::vector<uint8_t> digits_b(num_cmp * (num_digits-skip_digits), 0);


    // Step 1 break into digits \in [0, radix)
    DISPATCH_ALL_FIELDS(field, "break_digits", [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const auto mask_radix = makeBitsMask<u2k>(compare_radix_);
    NdArrayView<u2k> xinp(inp);

    for (int64_t i = 0; i < num_cmp; ++i) {
        for (int64_t j = 0; j < num_digits; ++j) {
          uint32_t shft = j * compare_radix_;
          digits_a[i * num_digits + j] = (xinp[i] >> shft) & mask_radix;
        }
        for (int64_t j = 0; j < num_digits-skip_digits; ++j) {
          uint32_t shft = (j+skip_digits) * compare_radix_;
          digits_b[i * (num_digits-skip_digits) + j] = (xinp[i] >> shft) & mask_radix;
        }
    }
    });
    std::vector<uint8_t> leaf_res_a(num_cmp * num_digits, 0);
    std::vector<uint8_t> leaf_res_b(num_cmp * (num_digits-skip_digits), 0);
    std::vector<uint8_t> leaf_cmp_a(num_cmp * num_digits, 0);
    std::vector<uint8_t> leaf_eq_a(num_cmp * num_digits, 0);
    std::vector<uint8_t> leaf_cmp_b(num_cmp * num_digits, 0);
    std::vector<uint8_t> leaf_eq_b(num_cmp * num_digits, 0);
  
    // Step 10: receiver of 1-of-N OT
    basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(digits_a), radix,
                                                absl::MakeSpan(leaf_res_a), 2);
    basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(digits_b), radix,
                                                absl::MakeSpan(leaf_res_b), 2);
    // extract equality bits from packed messages
    for (int64_t i = 0; i < num_cmp; ++i) {
        auto* this_leaf_res_a = leaf_res_a.data() + i * num_digits;
        auto* this_leaf_res_b = leaf_res_b.data() + i * (num_digits - skip_digits);
        auto* this_leaf_cmp_a = leaf_cmp_a.data() + i * num_digits;
        auto* this_leaf_eq_a = leaf_eq_a.data() + i * num_digits;
        auto* this_leaf_cmp_b = leaf_cmp_b.data() + i * num_digits;
        auto* this_leaf_eq_b = leaf_eq_b.data() + i * num_digits;
        for (int64_t j = 0; j < num_digits; ++j) {
            this_leaf_cmp_a[j] = this_leaf_res_a[j] & 1;
            this_leaf_eq_a[j] = (this_leaf_res_a[j] >> 1) & 1;
            if (j < skip_digits) {
              this_leaf_cmp_b[j] = this_leaf_cmp_a[j];
              this_leaf_eq_b[j] = this_leaf_eq_a[j];
            } else {
              this_leaf_cmp_b[j] = this_leaf_res_b[j-skip_digits] & 1;
              this_leaf_eq_b[j] = (this_leaf_res_b[j-skip_digits] >> 1) & 1;
            }
        }
    }

    auto boolean_t = makeType<BShrTy>(field, 1);
    NdArrayRef prev_cmp_a =
        ring_zeros(field, {static_cast<int64_t>(num_digits * num_cmp)})
            .as(boolean_t);
    NdArrayRef prev_eq_a =
        ring_zeros(field, {static_cast<int64_t>(num_digits * num_cmp)})
            .as(boolean_t);
    NdArrayRef prev_cmp_b =
        ring_zeros(field, {static_cast<int64_t>(num_digits * num_cmp)})
            .as(boolean_t);
    NdArrayRef prev_eq_b =
        ring_zeros(field, {static_cast<int64_t>(num_digits * num_cmp)})
            .as(boolean_t);

    DISPATCH_ALL_FIELDS(field, "copy_leaf", [&]() {
        NdArrayView<ring2k_t> xprev_cmp_a(prev_cmp_a);
        NdArrayView<ring2k_t> xprev_eq_a(prev_eq_a);
        NdArrayView<ring2k_t> xprev_cmp_b(prev_cmp_b);
        NdArrayView<ring2k_t> xprev_eq_b(prev_eq_b);
        pforeach(0, prev_cmp_a.numel(), [&](int64_t i) {
            xprev_cmp_a[i] = leaf_cmp_a[i];
            xprev_eq_a[i] = leaf_eq_a[i];
            xprev_cmp_b[i] = leaf_cmp_b[i];
            xprev_eq_b[i] = leaf_eq_b[i];
        });
    });

    auto [gt_a, gt_b] = TraversalAND(prev_cmp_a, prev_eq_a, prev_cmp_b, prev_eq_b, num_cmp, num_digits, skip_digits);
    return {gt_a.as(boolean_t), gt_b.as(boolean_t)};
}

std::tuple<NdArrayRef,NdArrayRef> CompareSigmoidProtocol::TraversalAND(
                                          NdArrayRef cmp_a, NdArrayRef eq_a, 
                                          NdArrayRef cmp_b, NdArrayRef eq_b,
                                          size_t num_input, size_t num_digits, int64_t skip_digits) {
  // Tree-based traversal ANDs
  // lt0[0], lt0[1], ..., lt0[M],
  // lt1[0], lt1[1], ..., lt1[M],
  // ...
  // ltN[0], ltN[1], ..., ltN[M],
  //
  // View two slices as Nx(M/2) matrix
  // Each row contains M/2 digits.
  // Slice0 contains the even digits
  // slice0: lt0[0], lt0[2], ..., lt0[2*j]
  //         lt1[0], lt1[2], ..., lt1[2*j]
  //         ....
  //         ltn[0], ltn[2], ..., ltn[2*j]
  //
  // Slice1 contains the odd digits
  // slice1: lt0[1], lt0[3], ..., lt0[2*j+1]
  //         lt1[1], lt0[3], ..., lt0[2*j+1]
  //         ....
  //         ltn[1], ltn[3], ..., ltn[2*j+1]
  SPU_ENFORCE(cmp_a.shape().size() == 1, "need 1D Array");
  SPU_ENFORCE(cmp_b.shape().size() == 1, "need 1D Array");
  SPU_ENFORCE_EQ(cmp_a.shape(), eq_a.shape());
  SPU_ENFORCE_EQ(cmp_b.shape(), eq_b.shape());
  SPU_ENFORCE_EQ(cmp_a.shape(), cmp_b.shape());
  SPU_ENFORCE_EQ(num_input * num_digits, (size_t)cmp_a.numel());
  for (size_t i = 1; i <= num_digits; i += 1) {
    size_t current_num_digits = num_digits / (1 << (i - 1));
    auto skip_current_num_digits = static_cast<int64_t>(skip_digits / (1 << i));
    if (current_num_digits == 1) {
      break;
    }
    // eq[i-1, j] <- eq[i, 2*j] * eq[i, 2*j+1]
    // cmp[i-1, j] <- cmp[i,2*j] * eq[i,2*j+1] ^ cmp[i,2*j+1]
    int64_t n = current_num_digits * num_input;

    auto lhs_eq_a = eq_a.slice({0}, {n}, {2});
    auto rhs_eq_a = eq_a.slice({1}, {n}, {2});
    auto lhs_cmp_a = cmp_a.slice({0}, {n}, {2});
    auto rhs_cmp_a = cmp_a.slice({1}, {n}, {2});

    auto lhs_eq_b = eq_b.slice({0}, {n}, {2});
    auto rhs_eq_b = eq_b.slice({1}, {n}, {2});
    auto lhs_cmp_b = cmp_b.slice({0}, {n}, {2});
    auto rhs_cmp_b = cmp_b.slice({1}, {n}, {2});

    if (current_num_digits == 2) {
      cmp_a = basic_ot_prot_->BitwiseAnd(lhs_cmp_a, rhs_eq_a);
      ring_xor_(cmp_a, rhs_cmp_a);
      cmp_b = basic_ot_prot_->BitwiseAnd(lhs_cmp_b, rhs_eq_b);
      ring_xor_(cmp_b, rhs_cmp_b);
      // We skip the ANDs for eq on the last loop
      continue;
    }

    // We skip the AND on the 0-th digit which is unnecessary for the next loop.
    int64_t nrow = num_input;
    int64_t ncol = current_num_digits / 2;
    auto col_b_begin = skip_current_num_digits;
    if (col_b_begin == 0) {
      col_b_begin = 1;
    }
    SPU_ENFORCE_EQ(lhs_eq_a.numel(), nrow * ncol);

    Shape subshape_a = {nrow * (ncol - 1)};
    NdArrayRef _lhs_eq_a(lhs_eq_a.eltype(), subshape_a);
    NdArrayRef _rhs_eq_a(rhs_eq_a.eltype(), subshape_a);
    NdArrayRef _lhs_cmp_a(lhs_cmp_a.eltype(), subshape_a);

    NdArrayRef _lhs_cmp_a_col0(lhs_cmp_a.eltype(), {static_cast<int64_t>(nrow)});
    NdArrayRef _rhs_eq_a_col0(rhs_eq_a.eltype(), _lhs_cmp_a_col0.shape());

    Shape subshape_b = {nrow * (ncol - col_b_begin)};
    NdArrayRef _lhs_eq_b(lhs_eq_b.eltype(), subshape_b);
    NdArrayRef _rhs_eq_b(rhs_eq_b.eltype(), subshape_b);
    NdArrayRef _lhs_cmp_b(lhs_cmp_b.eltype(), subshape_b);

    NdArrayRef _lhs_cmp_b_col0(lhs_cmp_b.eltype(), {static_cast<int64_t>(nrow)});
    NdArrayRef _rhs_eq_b_col0(rhs_eq_b.eltype(), _lhs_cmp_b_col0.shape());

    // Skip the 0-th column and take the remains columns
    // TODO(lwj): Can we have a better way to avoid such copying?
    for (int64_t r = 0; r < nrow; ++r) {
      std::memcpy(&_rhs_eq_a_col0.at(r), &rhs_eq_a.at(r * ncol), rhs_eq_a.elsize());
      std::memcpy(&_lhs_cmp_a_col0.at(r), &lhs_cmp_a.at(r * ncol),
                  lhs_cmp_a.elsize());
      if (skip_current_num_digits == 0) {
        std::memcpy(&_rhs_eq_b_col0.at(r), &rhs_eq_b.at(r * ncol), rhs_eq_b.elsize());
        std::memcpy(&_lhs_cmp_b_col0.at(r), &lhs_cmp_b.at(r * ncol),
                    lhs_cmp_b.elsize());
      }

      for (int64_t c = 1; c < ncol; ++c) {
        std::memcpy(&_lhs_cmp_a.at(r * (ncol - 1) + c - 1),
                    &lhs_cmp_a.at(r * ncol + c), lhs_eq_a.elsize());
        std::memcpy(&_lhs_eq_a.at(r * (ncol - 1) + c - 1),
                    &lhs_eq_a.at(r * ncol + c), lhs_eq_a.elsize());
        std::memcpy(&_rhs_eq_a.at(r * (ncol - 1) + c - 1),
                    &rhs_eq_a.at(r * ncol + c), rhs_eq_a.elsize());
      }
      for (int64_t c = col_b_begin; c < ncol; ++c) {
        std::memcpy(&_lhs_cmp_b.at(r * (ncol - col_b_begin) + c - col_b_begin),
                    &lhs_cmp_b.at(r * ncol + c), lhs_cmp_b.elsize());
        std::memcpy(&_lhs_eq_b.at(r * (ncol - col_b_begin) + c - col_b_begin),
                    &lhs_eq_b.at(r * ncol + c), lhs_eq_b.elsize());
        std::memcpy(&_rhs_eq_b.at(r * (ncol - col_b_begin) + c - col_b_begin),
                    &rhs_eq_b.at(r * ncol + c), rhs_eq_b.elsize());
      }
    }

    // Normal AND on 0-th column
    auto _next_cmp_a_col0 =
        basic_ot_prot_->BitwiseAnd(_rhs_eq_a_col0, _lhs_cmp_a_col0);
    NdArrayRef _next_cmp_b_col0(_next_cmp_a_col0.eltype(), _lhs_cmp_b_col0.shape());
    if (skip_current_num_digits == 0) {
      _next_cmp_b_col0 = basic_ot_prot_->BitwiseAnd(_rhs_eq_b_col0, _lhs_cmp_b_col0);
    }

    // Correlated AND on the remain columns
    auto [_next_cmp_a, _next_eq_a] =
        basic_ot_prot_->CorrelatedBitwiseAnd(_rhs_eq_a, _lhs_cmp_a, _lhs_eq_a);
    auto [_next_cmp_b, _next_eq_b] =
        basic_ot_prot_->CorrelatedBitwiseAnd(_rhs_eq_b, _lhs_cmp_b, _lhs_eq_b);

    // Concat two ANDs
    eq_a = NdArrayRef(eq_a.eltype(), {_next_cmp_a_col0.numel() + _next_cmp_a.numel()});
    cmp_a = NdArrayRef(cmp_a.eltype(), eq_a.shape());
    eq_b = NdArrayRef(eq_b.eltype(), {_next_cmp_a_col0.numel() + _next_cmp_a.numel()});
    cmp_b = NdArrayRef(cmp_b.eltype(), eq_b.shape());
    
    for (int64_t r = 0; r < nrow; ++r) {
      std::memcpy(&cmp_a.at(r * ncol), &_next_cmp_a_col0.at(r), cmp_a.elsize());
      if (skip_current_num_digits == 0) {
        std::memcpy(&cmp_b.at(r * ncol), &_next_cmp_b_col0.at(r), cmp_b.elsize());
      }

      for (int64_t c = 1; c < ncol; ++c) {
        std::memcpy(&cmp_a.at(r * ncol + c),
                    &_next_cmp_a.at(r * (ncol - 1) + c - 1), _next_cmp_a.elsize());
        std::memcpy(&eq_a.at(r * ncol + c), &_next_eq_a.at(r * (ncol - 1) + c - 1),
                    _next_eq_a.elsize());
      }
      if (skip_current_num_digits > 0) {
        std::memcpy(&cmp_b.at(r * ncol),
                      &cmp_a.at(r * ncol), skip_current_num_digits*cmp_a.elsize());
        std::memcpy(&eq_b.at(r * ncol),
                      &eq_a.at(r * ncol), skip_current_num_digits*eq_a.elsize());
      }

      for (int64_t c = col_b_begin; c < ncol; ++c) {
        std::memcpy(&cmp_b.at(r * ncol + c),
                    &_next_cmp_b.at(r * (ncol - col_b_begin) + c - col_b_begin), 
                    _next_cmp_b.elsize());
        std::memcpy(&eq_b.at(r * ncol + c), 
                  &_next_eq_b.at(r * (ncol - col_b_begin) + c - col_b_begin),
                    _next_eq_b.elsize());
      }
    }

    ring_xor_(cmp_a, rhs_cmp_a);
    ring_xor_(cmp_b, rhs_cmp_b);
  }
  return {cmp_a, cmp_b};
}

// sender
std::tuple<NdArrayRef, NdArrayRef> CompareSigmoidProtocol::Compute(const NdArrayRef& inp_a, const NdArrayRef& inp_b, bool greater_than, int64_t bitwidth, size_t skip_bits) {
  int64_t bw = SizeOf(inp_a.eltype().as<Ring2k>()->field()) * 8;
  SPU_ENFORCE(bitwidth >= 0 && bitwidth <= bw, "bit_width={} out of bound",
              bitwidth);
  if (bitwidth == 0) {
    bitwidth = bw;
  }
  // NOTE(lwj): reshape might need copy
  auto flatten_a = inp_a.reshape({inp_a.numel()});
  auto flatten_b = inp_b.reshape({inp_b.numel()});
  auto [res_a, res_b] = DoCompute(flatten_a, flatten_b, greater_than, bitwidth, skip_bits);
  return {res_a.reshape(inp_a.shape()), res_b.reshape(inp_b.shape())};
}

// receiver
std::tuple<NdArrayRef, NdArrayRef> CompareSigmoidProtocol::Compute(const NdArrayRef& inp, bool greater_than,
                                    int64_t bitwidth, size_t skip_bits) {
  int64_t bw = SizeOf(inp.eltype().as<Ring2k>()->field()) * 8;
  SPU_ENFORCE(bitwidth >= 0 && bitwidth <= bw, "bit_width={} out of bound",
              bitwidth);
  if (bitwidth == 0) {
    bitwidth = bw;
  }
  // NOTE(lwj): reshape might need copy
  auto flatten = inp.reshape({inp.numel()});
  auto [res_a, res_b] = DoCompute(flatten, greater_than, bitwidth, skip_bits);
  return {res_a.reshape(inp.shape()), res_b.reshape(inp.shape())};
}

}  // namespace spu::mpc::cheetah
