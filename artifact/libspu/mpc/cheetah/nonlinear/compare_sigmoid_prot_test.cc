// Copyright 2022 Ant Group Co., Ltd.
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

#include <random>

#include "gtest/gtest.h"

#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class CompareSigmoidProtTest
    : public ::testing::TestWithParam<std::tuple<FieldType, bool, size_t>> {
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, CompareSigmoidProtTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64),
                     testing::Values(true, false),
                     testing::Values(4UL)),  // divide k
    [](const testing::TestParamInfo<CompareSigmoidProtTest::ParamType> &p) {
      return fmt::format("{}Radix{}Greater{}", std::get<0>(p.param),
                         (int)std::get<2>(p.param), (int)std::get<1>(p.param));
    });

TEST_P(CompareSigmoidProtTest, Basic) {
  size_t kWorldSize = 2;
  Shape shape = {100};
  FieldType field = std::get<0>(GetParam());
  size_t radix = std::get<2>(GetParam());
  bool greater_than = std::get<1>(GetParam());

  NdArrayRef inp[3];
  inp[0] = ring_rand(field, shape);
  inp[1] = ring_rand(field, shape);
  inp[2] = ring_rand(field, shape);

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto xinp = NdArrayView<ring2k_t>(inp[0]);
    xinp[0] = 1;
    xinp[1] = 10;
    xinp[2] = 100;

    xinp = NdArrayView<ring2k_t>(inp[1]);
    xinp[0] = 1;
    xinp[1] = 9;
    xinp[2] = 1000;
  });

  NdArrayRef cmp1_oup[2];
  NdArrayRef cmp2_oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    int rank = ctx->Rank();
    auto base = std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::EMP_Ferret);
    CompareSigmoidProtocol comp_prot(base, radix);
    if (rank == 0) {
      std::tie(cmp1_oup[rank], cmp2_oup[rank]) = comp_prot.Compute(inp[0], inp[1], greater_than);
    } else {
      std::tie(cmp1_oup[rank], cmp2_oup[rank]) = comp_prot.Compute(inp[2], greater_than);
    }
  });

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto xout1_0 = NdArrayView<ring2k_t>(cmp1_oup[0]);
    auto xout1_1 = NdArrayView<ring2k_t>(cmp1_oup[1]);
    auto xout2_0 = NdArrayView<ring2k_t>(cmp2_oup[0]);
    auto xout2_1 = NdArrayView<ring2k_t>(cmp2_oup[1]);
    auto xinp0 = NdArrayView<ring2k_t>(inp[0]);
    auto xinp1 = NdArrayView<ring2k_t>(inp[1]);
    auto xinp2 = NdArrayView<ring2k_t>(inp[2]);

    for (int64_t i = 0; i < shape.numel(); ++i) {
      bool expected_1 = greater_than ? xinp0[i] > xinp2[i] : xinp0[i] < xinp2[i];
      bool got_cmp_1 = xout1_0[i] ^ xout1_1[i];
      bool expected_2 = greater_than ? xinp1[i] > xinp2[i] : xinp1[i] < xinp2[i];
      bool got_cmp_2 = xout2_0[i] ^ xout2_1[i];
      ASSERT_EQ(expected_1, got_cmp_1);
      ASSERT_EQ(expected_2, got_cmp_2);
    }
  });
}

}  // namespace spu::mpc::cheetah
