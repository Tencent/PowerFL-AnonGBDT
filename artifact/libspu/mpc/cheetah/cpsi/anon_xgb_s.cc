
#include "yacl/crypto/rand/rand.h"

#include "libspu/mpc/cheetah/cpsi/anon_xgb.h"
#include "libspu/mpc/cheetah/state.h"

namespace spu::mpc::cheetah {
// single cpsi
AnonXGB::AnonXGB(std::shared_ptr<yacl::link::Context> link_ctx,
                 psi::PsiRoleType role, size_t log2_scale, int single)
    : link_ctx_(link_ctx), role_(role), log2_scale_(log2_scale) {
  psi::PsiRoleType duel_role = role_ == (psi::PsiRoleType::Sender)
    ? psi::PsiRoleType::Receiver : psi::PsiRoleType::Sender;

  // init base ot
  auto conn = std::make_shared<Communicator>(link_ctx_);
  base_ot_ =
      std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::YACL_Ferret);

  // Sender executes simple hash, Receiver executes cuckoo hash
  vole_cpsi_ptr0_ = std::make_shared<VoleCPsi>(link_ctx_, base_ot_, duel_role);

  histogram_ptr0_ = std::make_shared<HistoGram>(link_ctx_, role_);
  operators_ptr_ = std::make_shared<Operators>(link_ctx_, role_);
}

void AnonXGB::initializeSingle(const std::vector<std::string> &items,
                               const std::vector<double> &labels, size_t cols) {
  auto bgn_bytes =
      link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;

  histogram_ptr0_->initialize(true);

  self_rows_ = items.size();
  self_cols_ = cols;
  peer_rows_ = ExchangeSize(self_rows_);
  peer_cols_ = ExchangeSize(self_cols_);

  printf("self_cols = %zu, peer_cols = %zu\n", self_cols_, peer_cols_);

  auto ids = absl::MakeSpan(items);
  mask0_ = vole_cpsi_ptr0_->RunCPsi(ids);

  if (role_ == psi::PsiRoleType::Sender) {
    SPU_ENFORCE(items.size() == labels.size());

    std::vector<uint64_t> encoded_labels = encode(labels, log2_scale_);
    label_ashr0_ = vole_cpsi_ptr0_->adjustOrder(encoded_labels);
    label_ashr0_ = vole_cpsi_ptr0_->multiplexer(label_ashr0_, mask0_);

    std::vector<uint64_t> encode_grad(items.size(), encode(0.5, log2_scale_));
    for (size_t i = 0; i < encode_grad.size(); i++) {
      encode_grad[i] = encode_grad[i] - encoded_labels[i];
    }
    std::vector<uint64_t> encode_hess(items.size(), encode(0.25, log2_scale_));

    GradHess encode_grad_hess = GradHess(encode_grad, encode_hess);
    grad_hess_ashr0_ = self_share(encode_grad_hess, 0);
  } else {
    label_ashr0_.assign(vole_cpsi_ptr0_->getNumBin(), 0ULL);
    label_ashr0_ = vole_cpsi_ptr0_->multiplexer(label_ashr0_, mask0_);

    GradHess encode_grad_hess;
    grad_hess_ashr0_ = self_share(encode_grad_hess, 0);
  }

  tree_eval_ashr0_.assign(vole_cpsi_ptr0_->getNumBin(), 0ULL);

  auto end_bytes =
      link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;
  std::cout << "initialize takes " << (end_bytes - bgn_bytes) << " comm bytes"
            << std::endl;

  //            grad_hess_sum_test(grad_hess_ashr0_, grad_hess_ashr1_,
  //            "initialize");
}

void AnonXGB::compute_histogramSingle(
    std::vector<GradHess> &grad_hess_ashr_vec,
    const std::vector<std::vector<uint8_t>> &split_bools) {
  std::vector<std::vector<uint64_t>> grad_ashr_vec;
  std::vector<std::vector<uint64_t>> hess_ashr_vec;

  for (const auto &grad_hess_ashr : grad_hess_ashr_vec) {
    grad_ashr_vec.push_back(grad_hess_ashr.getGrad());
    hess_ashr_vec.push_back(grad_hess_ashr.getHess());
  }

  double scale = std::pow(2, log2_scale_);

  if (role_ == psi::PsiRoleType::Sender) {
    self_split_bools_ = adjust_split_bools(split_bools, vole_cpsi_ptr0_);

    yacl::Buffer buf = link_ctx_->Recv(
        link_ctx_->NextRank(), fmt::format("recv encrypted grad0 and hess0"));

    scale = histogram_ptr0_->HomCompHistoGram(grad_ashr_vec, hess_ashr_vec, buf,
                                              scale, self_split_bools_);

    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf,
        fmt::format("send summed grad0 and hess0 of size {} bytes",
                    buf.size()));
  } else {
    yacl::Buffer buf =
        histogram_ptr0_->CKKSEncrypt(grad_ashr_vec, hess_ashr_vec, scale);

    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf,
        fmt::format("send encrypted grad0 and hess0 of size {} bytes",
                    buf.size()));

    buf = link_ctx_->Recv(link_ctx_->NextRank(),
                          fmt::format("recv summed grad0 and hess0"));

    scale = histogram_ptr0_->PostProcess(grad_ashr_vec, hess_ashr_vec, buf,
                                         peer_cols_);
  }

  for (size_t i = 0; i < grad_hess_ashr_vec.size(); i++) {
    grad_hess_ashr_vec[i] = GradHess(grad_ashr_vec[i], hess_ashr_vec[i]);
    adjust_scale(grad_hess_ashr_vec[i], scale);
  }
}

void AnonXGB::compute_histogramSingle(
    std::vector<GradHess> &hist_gh_ashr_vec0,
    std::vector<GradHess> &hist_gh_ashr_vec1,
    const std::vector<std::vector<uint8_t>> &split_bools) {
  compute_histogramSingle(hist_gh_ashr_vec0, split_bools);

  size_t bit_width = sizeof(uint64_t) * 8;

  if (split_bool_shrs_.empty()) {
    if (role_ == psi::PsiRoleType::Receiver) {
      size_t len = (self_cols_ + bit_width - 1) / bit_width;

      for (size_t i = 0; i < len; i++) {
        std::vector<uint64_t> bools(split_bools.size(), 0ULL);
        for (size_t k = 0; k < bools.size(); k++) {
          size_t bgn = std::min(i * bit_width, self_cols_);
          size_t end = std::min(bgn + bit_width, self_cols_);
          for (size_t j = bgn; j < end; j++) {
            bools[k] |= static_cast<uint64_t>(split_bools[k][j] & 0x1)
                        << (j - bgn);
          }
        }

        bools = vole_cpsi_ptr0_->OpprfShareSender(bools, true);

        split_bool_shrs_.resize(bools.size());
        for (auto &rows : split_bool_shrs_) {
          rows.resize(self_cols_);
        }

        for (size_t k = 0; k < bools.size(); k++) {
          size_t bgn = std::min(i * bit_width, self_cols_);
          size_t end = std::min(bgn + bit_width, self_cols_);
          for (size_t j = bgn; j < end; j++) {
            split_bool_shrs_[k][j] = (bools[k] >> (j - bgn)) & 0x1;
          }
        }
      }
    } else {
      size_t len = (peer_cols_ + bit_width - 1) / bit_width;

      for (size_t i = 0; i < len; i++) {
        auto bools = vole_cpsi_ptr0_->OpprfShareReceiver(true);

        split_bool_shrs_.resize(bools.size());
        for (auto &rows : split_bool_shrs_) {
          rows.resize(peer_cols_);
        }

        for (size_t k = 0; k < bools.size(); k++) {
          size_t bgn = std::min(i * bit_width, peer_cols_);
          size_t end = std::min(bgn + bit_width, peer_cols_);
          for (size_t j = bgn; j < end; j++) {
            split_bool_shrs_[k][j] = (bools[k] >> (j - bgn)) & 0x1;
          }
        }
      }
    }
  }

  size_t rows = split_bool_shrs_.size();
  size_t cols = split_bool_shrs_[0].size();

  std::vector<std::vector<uint8_t>> mask_vec;
  for (size_t j = 0; j < cols; j++) {
    std::vector<uint8_t> mask(rows);
    for (size_t i = 0; i < rows; i++) {
      mask[i] = split_bool_shrs_[i][j];
    }
    mask_vec.push_back(mask);
  }

  for (auto &hist_gh_ashr : hist_gh_ashr_vec1) {
    std::vector<GradHess> grad_hess_ashr_vec(cols, hist_gh_ashr);
    multiplex(grad_hess_ashr_vec, mask_vec);

    for (auto &gh_ashr : grad_hess_ashr_vec) {
      gh_ashr = gh_ashr.sum();
    }

    hist_gh_ashr = GradHess::concat(grad_hess_ashr_vec);
  }
}

std::vector<uint8_t> AnonXGB::synchron_col(size_t col_index) {
  size_t rows = split_bool_shrs_.size();
  size_t cols = split_bool_shrs_[0].size();
  size_t log2_cols = std::ceil(std::log2(cols));

  auto *context = operators_ptr_->getContext();
  auto *comm = context->getState<Communicator>();
  auto *ot_state = context->getState<cheetah::CheetahOTState>();
  ot_state->LazyInit(comm, base_ot_, 0);

  auto sender = ot_state->get(0)->GetSenderCOT();
  auto receiver = ot_state->get(0)->GetReceiverCOT();

  auto prng_gen = [&](const std::vector<uint64_t> &values, size_t index,
                      size_t len) {
    SPU_ENFORCE(values.size() == log2_cols);

    std::vector<uint8_t> hash_input(sizeof(uint64_t) * log2_cols);
    std::memcpy(hash_input.data(),
                reinterpret_cast<const uint8_t *>(values.data()),
                sizeof(uint64_t) * log2_cols);
    uint128_t prng_seed = yacl::crypto::Blake3_128(hash_input);

    std::vector<uint8_t> rr(len, 0);

    constexpr yacl::crypto::SymmetricCrypto::CryptoType kCryptoType =
        yacl::crypto::SymmetricCrypto::CryptoType::AES128_CTR;
    yacl::crypto::FillPRand(kCryptoType, prng_seed, 0, index,
                            absl::MakeSpan(rr));

    return rr;
  };

  std::vector<uint8_t> output;

  yacl::Buffer buffer(cols * rows);

  if (role_ == psi::PsiRoleType::Receiver) {
    std::vector<uint8_t> mask(log2_cols);
    for (size_t i = 0; i < log2_cols; i++) {
      mask[i] = (col_index >> i) & 0x1;
    }

    std::vector<uint64_t> recv(log2_cols);
    receiver->RecvRMCC(absl::MakeSpan(mask), absl::MakeSpan(recv));

    auto rands = prng_gen(recv, col_index, rows);
    output.resize(rows);

    buffer = link_ctx_->Recv(link_ctx_->NextRank(),
                             fmt::format("recv opprf-share-table"));
    SPU_ENFORCE(buffer.size() == static_cast<int64_t>(cols * rows));

    std::memcpy(output.data(),
                reinterpret_cast<uint8_t *>(buffer.data()) + col_index * rows,
                rows);

    for (size_t i = 0; i < rows; i++) {
      output[i] = split_bool_shrs_[i][col_index] ^ output[i] ^ rands[i];
    }

    std::transform(output.begin(), output.end(), output.begin(),
                   [&](uint8_t v) { return v & 0x1; });

    for (size_t i = 0; i < rows; i++) {
      SPU_ENFORCE(output[i] <= 1, "{} is not in {0, 1}", output[i]);
    }
  } else {
    output = yacl::crypto::SecureRandBytes(rows);
    std::transform(output.begin(), output.end(), output.begin(),
                   [&](uint8_t v) { return v & 0x1; });

    std::vector<uint64_t> sent0(log2_cols);
    std::vector<uint64_t> sent1(log2_cols);

    sender->SendRMCC(absl::MakeSpan(sent0), absl::MakeSpan(sent1));
    sender->Flush();

    for (size_t j = 0; j < cols; j++) {
      std::vector<uint64_t> values;
      for (size_t k = 0; k < log2_cols; k++) {
        if (((j >> k) & 0x1) != 0U) {
          values.push_back(sent1[k]);
        } else {
          values.push_back(sent0[k]);
        }
      }

      auto rands = prng_gen(values, j, rows);
      for (size_t i = 0; i < rows; i++) {
        rands[i] ^= split_bool_shrs_[i][j] ^ output[i];
      }

      std::memcpy(reinterpret_cast<uint8_t *>(buffer.data()) + j * rows,
                  rands.data(), rows);
    }

    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buffer,
        fmt::format("AnonXGBSingle:totalSize={}", buffer.size()));
  }

  return output;
}

void AnonXGB::share_bool_splitsSingle(
    std::vector<std::vector<uint8_t>> &bool_vec,
    const std::vector<std::vector<uint8_t>> &split_bools,
    const std::vector<size_t> &index_arr) {
  std::vector<uint64_t> bool0;
  std::vector<uint64_t> bool1;

  std::vector<size_t> idx_pos_arr0;
  std::vector<size_t> idx_pos_arr1;

  if (role_ == psi::PsiRoleType::Sender) {
    bool0.resize(self_rows_, 0ULL);

    for (size_t l = 0; l < index_arr.size(); l++) {
      size_t index = index_arr[l];
      if (index != static_cast<uint64_t>(-1)) {
        SPU_ENFORCE(index < self_cols_);
        SPU_ENFORCE(l < 64, "only support 64-parallel opprf-share");

        for (size_t i = 0; i < bool0.size(); i++) {
          bool0[i] ^= static_cast<uint64_t>(split_bools[i][index]) << l;
        }

        idx_pos_arr0.push_back(l);
      } else {
        idx_pos_arr1.push_back(l);
      }
    }
  } else {
    //            bool1.resize(self_rows_, 0ULL);

    for (size_t l = 0; l < index_arr.size(); l++) {
      size_t index = index_arr[l];
      if (index != static_cast<uint64_t>(-1)) {
        SPU_ENFORCE(index < self_cols_);
        SPU_ENFORCE(l < 64, "only support 64-parallel opprf-share");

        //                    for (size_t i = 0; i < bool1.size(); i++) {
        //                        bool1[i] ^=
        //                        static_cast<uint64_t>(split_bools[i][index])
        //                        << l;
        //                    }

        idx_pos_arr1.push_back(l);
      } else {
        idx_pos_arr0.push_back(l);
      }
    }
  }

  bool_vec.resize(index_arr.size());

  std::vector<uint64_t> selfs;
  //        std::vector<uint64_t> peers;

  if (!idx_pos_arr0.empty()) {
    //                printf("idx0_size: %zu\n", idx_pos_arr0.size());
    if (role_ == psi::PsiRoleType::Sender) {
      selfs = vole_cpsi_ptr0_->adjustOrder(bool0);
      selfs = vole_cpsi_ptr0_->bmultiplexer(selfs, mask0_);
    } else {
      selfs.assign(vole_cpsi_ptr0_->getNumBin(), 0ULL);
      selfs = vole_cpsi_ptr0_->bmultiplexer(selfs, mask0_);
    }
  }

  for (const auto &pos : idx_pos_arr0) {
    bool_vec[pos].resize(selfs.size());
    std::transform(selfs.begin(), selfs.end(), bool_vec[pos].begin(),
                   [&](uint64_t vv) { return (vv >> pos) & 0x1; });
  }

  //        if (!idx_pos_arr1.empty()) {
  //            if (role_ == psi::PsiRoleType::Receiver) {
  //                peers = vole_cpsi_ptr0_->OpprfShareSender(bool1, true);
  //            } else {
  //                peers = vole_cpsi_ptr0_->OpprfShareReceiver(true);
  //            }
  //        }

  //        for (const auto &pos : idx_pos_arr1) {
  //            bool_vec[pos].resize(peers.size());
  //            std::transform(peers.begin(), peers.end(),
  //            bool_vec[pos].begin(), [&](uint64_t vv) {
  //                return (vv >> pos) & 0x1;
  //            });
  //        }

  // printf("idx_pos_arr1_size = %zu \n", idx_pos_arr1.size());

  for (const auto &pos : idx_pos_arr1) {
    bool_vec[pos] = synchron_col(index_arr[pos]);
    //            bool_vec[pos] =
    //            yacl::crypto::SecureRandBytes(split_bool_shrs_.size());
    //            //synchron_col(index_arr[pos]);
    //
    //            std::transform(bool_vec[pos].begin(), bool_vec[pos].end(),
    //            bool_vec[pos].begin(), [&](uint8_t v) {
    //                return v & 0x1;
    //            });
  }
}

void AnonXGB::update_tree_evalSingle(
    const std::vector<std::vector<uint8_t>> &leaf_mask_bshr_vec,
    const std::vector<uint64_t> &leaf_wt_ashr) {
  auto tree_eval_ashr = multiplex_single(leaf_wt_ashr, leaf_mask_bshr_vec);

  SPU_ENFORCE(tree_eval_ashr.size() == tree_eval_ashr0_.size(), "{} != {}",
              tree_eval_ashr.size(), tree_eval_ashr0_.size());
  for (size_t i = 0; i < tree_eval_ashr.size(); i++) {
    tree_eval_ashr0_[i] += tree_eval_ashr[i];
  }
}

void AnonXGB::update_grad_hess_ashrSingle() {
  printf("tree_eval_ashr0_size = %zu \n", tree_eval_ashr0_.size());
  auto pp_ashr0 = sigmoid(tree_eval_ashr0_, 1);
  auto pp_square_ashr0 = square(pp_ashr0);
  truncate(pp_square_ashr0, log2_scale_, SignType::Positive);

  auto gg_ashr0 = sub(pp_ashr0, label_ashr0_);
  auto hh_ashr0 = sub(pp_ashr0, pp_square_ashr0);
  grad_hess_ashr0_ = GradHess(gg_ashr0, hh_ashr0);
}

std::pair<std::vector<std::vector<size_t>>, std::vector<uint64_t>> AnonXGB::gen_treeSingle(
    const std::vector<std::vector<uint8_t>> &split_bools, double lambda,
    int tree_level, bool is_last_tree) {
  auto time_start = std::chrono::high_resolution_clock::now();

  auto bgn_bytes =
      link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;

  size_t begin = 0;
  size_t end = 1;

  std::vector<GradHess> gh_ashr_vec;
  gh_ashr_vec.push_back(grad_hess_ashr0_);

  std::vector<std::vector<uint8_t>> mask_bshr_vec;
  mask_bshr_vec.push_back(mask0_);

  std::vector<GradHess> hist_ashr_svec;
  std::vector<GradHess> hist_ashr_rvec;

  std::vector<std::vector<size_t>> opt_cols_vec;

  size_t leaf_nums = 1ULL << tree_level;
  while ((end - begin) < leaf_nums) {
    std::vector<GradHess> hist_ashr_schild;
    std::vector<GradHess> hist_ashr_rchild;

    for (size_t i = begin; i < end; i += 2) {
      hist_ashr_schild.push_back(gh_ashr_vec[i]);
      hist_ashr_rchild.push_back(gh_ashr_vec[i]);
    }

    compute_histogramSingle(hist_ashr_schild, hist_ashr_rchild, split_bools);
    SPU_ENFORCE(hist_ashr_svec.size() == begin &&
                hist_ashr_rvec.size() == begin);

    if (begin == 0) {
      hist_ashr_svec.push_back(hist_ashr_schild[0]);
      hist_ashr_rvec.push_back(hist_ashr_rchild[0]);
    } else {
      for (size_t i = begin, j = 0; i < end; i += 2, j++) {
        hist_ashr_svec.push_back(hist_ashr_schild[j]);
        hist_ashr_svec.push_back(hist_ashr_svec[i / 2] - hist_ashr_schild[j]);

        hist_ashr_rvec.push_back(hist_ashr_rchild[j]);
        hist_ashr_rvec.push_back(hist_ashr_rvec[i / 2] - hist_ashr_rchild[j]);
      }
    }

    std::vector<GradHess> gh_ashr;
    std::vector<GradHess> sum_gh_ashr;
    for (size_t i = begin; i < end; i++) {
      gh_ashr.push_back(gh_ashr_vec[i]);
      sum_gh_ashr.push_back(gh_ashr_vec[i].sum());
    }

    std::vector<GradHess> level_hist_ashr0(hist_ashr_svec.data() + begin,
                                           hist_ashr_svec.data() + end);
    std::vector<GradHess> level_hist_ashr1(hist_ashr_rvec.data() + begin,
                                           hist_ashr_rvec.data() + end);
    std::vector<size_t> opt_cols = find_optimal_cols(
        lambda, level_hist_ashr0, level_hist_ashr1, sum_gh_ashr, sum_gh_ashr);
    opt_cols_vec.push_back(opt_cols);

    std::vector<std::vector<uint8_t>> level_mask_bshr_vec;
    share_bool_splitsSingle(level_mask_bshr_vec, split_bools, opt_cols);

    for (size_t i = begin, j = 0; i < end; i++, j++) {
      mask_bshr_vec.push_back(level_mask_bshr_vec[j]);
      mask_bshr_vec.push_back(
          bits_xor(mask_bshr_vec[i], level_mask_bshr_vec[j]));
    }

    multiplex(gh_ashr, level_mask_bshr_vec);

    for (size_t i = begin, j = 0; i < end; i++, j++) {
      gh_ashr_vec.push_back(gh_ashr[j]);
      gh_ashr_vec.push_back(gh_ashr_vec[i] - gh_ashr[j]);
    }

    begin = end;
    end = gh_ashr_vec.size();
  }

  std::vector<uint64_t> leaf_grad_ashr;
  std::vector<uint64_t> leaf_hess_ashr;
  for (size_t i = begin; i < end; i++) {
    GradHess gh_sum = gh_ashr_vec[i].sum();
    leaf_grad_ashr.push_back(gh_sum.getGrad()[0]);
    leaf_hess_ashr.push_back(gh_sum.getHess()[0]);
  }

  negate(leaf_grad_ashr);
  add_plain(leaf_hess_ashr, lambda, std::pow(2.0, log2_scale_));
  auto leaf_wt_ashr = divide(leaf_grad_ashr, leaf_hess_ashr, SignType::Unknown);

  auto leaf_wt = recoveryA(leaf_wt_ashr);
  if (role_ == psi::PsiRoleType::Sender) {
    printf("comp leaf weight: ");
    for (size_t i = 0; i < leaf_wt.size(); i++) {
      printf("%f ", decode(leaf_wt[i], log2_scale_));
    }
    printf("\n");
  }

  // update tree_eval and grad-hess
  if (!is_last_tree) {
    std::vector<std::vector<uint8_t>> leaf_mask_bshr_vec(
        mask_bshr_vec.data() + begin, mask_bshr_vec.data() + end);
    update_tree_evalSingle(leaf_mask_bshr_vec, leaf_wt_ashr);
    update_grad_hess_ashrSingle();
  }

  leaf_wt_ashr_vec_.push_back(leaf_wt_ashr);

  auto end_bytes =
      link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;

  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
      time_end - time_start);
  std::cout << "gen_tree of depth " << tree_level << " takes "
            << time_diff.count() << " us" << std::endl;
  std::cout << "gen_tree of depth " << tree_level << " takes "
            << (end_bytes - bgn_bytes) << " comm bytes" << std::endl;

  return std::make_pair(opt_cols_vec, leaf_wt_ashr);
}
}  // namespace spu::mpc::cheetah
