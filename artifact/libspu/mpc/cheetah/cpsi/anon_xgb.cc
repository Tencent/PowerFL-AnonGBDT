#include "libspu/mpc/cheetah/cpsi/anon_xgb.h"

#include <omp.h>
#include <cstddef>

#include "yacl/crypto/rand/rand.h"

#include "libspu/core/prelude.h"
#include "libspu/mpc/cheetah/arith/common.h"
#include "libspu/mpc/cheetah/state.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

GradHess::GradHess(uint64_t gg, uint64_t hh) {
  grad_.resize(1);
  hess_.resize(1);
  grad_[0] = gg;
  hess_[0] = hh;
}

GradHess::GradHess(const std::vector<uint64_t> &ggs,
                   const std::vector<uint64_t> &hhs) {
  SPU_ENFORCE(ggs.size() == hhs.size());
  grad_.resize(ggs.size());
  hess_.resize(hhs.size());
  std::copy_n(ggs.data(), ggs.size(), grad_.data());
  std::copy_n(hhs.data(), hhs.size(), hess_.data());
}

GradHess::GradHess(const GradHess &other) {
  grad_.resize(other.grad_.size());
  hess_.resize(other.hess_.size());
  std::copy_n(other.grad_.data(), grad_.size(), grad_.data());
  std::copy_n(other.hess_.data(), hess_.size(), hess_.data());
}

GradHess GradHess::extend(size_t len) {
  size_t size = this->size();

  std::vector<uint64_t> grad(len, 0);
  std::vector<uint64_t> hess(len, 0);

  for (size_t i = 0; i < len; i += size) {
    uint64_t n = std::min(size, len - i);
    std::copy_n(this->grad_.data(), n, grad.data() + i);
    std::copy_n(this->hess_.data(), n, hess.data() + i);
  }

  return GradHess(grad, hess);
}

const GradHess &GradHess::operator=(const GradHess &other) {
  if (this != &other) {
    this->grad_.resize(other.size());
    this->hess_.resize(other.size());
    for (size_t i = 0; i < this->size(); i++) {
      this->grad_[i] = other.grad_[i];
      this->hess_[i] = other.hess_[i];
    }
  }

  return *this;
}

const GradHess &GradHess::operator+(const GradHess &other) {
  SPU_ENFORCE(this->size() == other.size());

  for (size_t i = 0; i < this->size(); i++) {
    this->grad_[i] += other.grad_[i];
    this->hess_[i] += other.hess_[i];
  }

  return *this;
}

const GradHess &GradHess::operator-(const GradHess &other) {
  SPU_ENFORCE(this->size() == other.size());

  for (size_t i = 0; i < this->size(); i++) {
    this->grad_[i] -= other.grad_[i];
    this->hess_[i] -= other.hess_[i];
  }

  return *this;
}

GradHess GradHess::concat(const GradHess &aa, const GradHess &bb) {
  size_t size1 = aa.size();
  size_t size2 = bb.size();

  std::vector<uint64_t> grad(size1 + size2);
  std::vector<uint64_t> hess(size1 + size2);

  std::copy_n(aa.getGrad().data(), size1, grad.data());
  std::copy_n(bb.getGrad().data(), size2, grad.data() + size1);

  std::copy_n(aa.getHess().data(), size1, hess.data());
  std::copy_n(bb.getHess().data(), size2, hess.data() + size1);

  return GradHess(grad, hess);
}

GradHess GradHess::concat(const std::vector<GradHess> &arr) {
  size_t size = 0;
  for (const auto &vv : arr) {
    size += vv.size();
  }

  std::vector<uint64_t> grad(size);
  std::vector<uint64_t> hess(size);

  size_t offset = 0;
  for (const auto &vv : arr) {
    std::copy_n(vv.getGrad().data(), vv.size(), grad.data() + offset);
    std::copy_n(vv.getHess().data(), vv.size(), hess.data() + offset);
    offset += vv.size();
  }

  return GradHess(grad, hess);
}

AnonXGB::AnonXGB(std::shared_ptr<yacl::link::Context> link_ctx,
                 psi::PsiRoleType role, size_t log2_scale, 
                 bool is_inference, bool enable_sigmoid)
    : link_ctx_(std::move(link_ctx)), role_(role), log2_scale_(log2_scale), 
    is_inference_(is_inference), enable_sigmoid_(enable_sigmoid) {
  psi::PsiRoleType duel_role = role_ == (psi::PsiRoleType::Sender)
  ? psi::PsiRoleType::Receiver : psi::PsiRoleType::Sender;

  // init base ot
  auto conn = std::make_shared<Communicator>(link_ctx_);
  base_ot_ =
      std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::YACL_Ferret);

  // Sender executes simple hash, Receiver executes cuckoo hash
  vole_cpsi_ptr0_ = std::make_shared<VoleCPsi>(link_ctx_, base_ot_, duel_role);

  operators_ptr_ = std::make_shared<Operators>(link_ctx_, role_);

  
  if (!is_inference) {
    vole_cpsi_ptr1_ = std::make_shared<VoleCPsi>(link_ctx_, base_ot_, role_);

    histogram_ptr0_ = std::make_shared<HistoGram>(link_ctx_, role_);
    histogram_ptr1_ = std::make_shared<HistoGram>(link_ctx_, duel_role);
  }

  seal::EncryptionParameters parms(seal::scheme_type::bfv);
  parms.set_poly_modulus_degree(bfv_coeff_count_);
  parms.set_coeff_modulus(
      seal::CoeffModulus::Create(bfv_coeff_count_, {60, 49}));
  parms.set_plain_modulus(bfv_plain_mod_);
  parms.set_use_special_prime(false);

  bfv_context_ = std::make_shared<seal::SEALContext>(parms);
  bfv_encoder_ = std::make_shared<seal::BatchEncoder>(*bfv_context_);

  seal::KeyGenerator keygen(*bfv_context_);
  secret_key_ = keygen.secret_key();

  bfv_encryptor_ =
      std::make_shared<seal::Encryptor>(*bfv_context_, secret_key_);
  bfv_decryptor_ =
      std::make_shared<seal::Decryptor>(*bfv_context_, secret_key_);
  bfv_evaluator_ = std::make_shared<seal::Evaluator>(*bfv_context_);
}

GradHess AnonXGB::multiplex(const GradHess &grad_hess_ashr,
                            const std::vector<uint8_t> &masks, int type) {
  SPU_ENFORCE(grad_hess_ashr.size() == masks.size());

  auto grad_ashr = grad_hess_ashr.getGrad();
  auto hess_ashr = grad_hess_ashr.getHess();

  multiplex(grad_ashr, hess_ashr, masks, type);

  return GradHess(grad_ashr, hess_ashr);
}

GradHess AnonXGB::self_share(const GradHess &encoded_grad_hess, int pos) {
  auto vole_cpsi_ptr = pos == 0 ? vole_cpsi_ptr0_ : vole_cpsi_ptr1_;
  auto mask = pos == 0 ? mask0_ : mask1_;

  std::vector<uint64_t> grad = encoded_grad_hess.getGrad();
  std::vector<uint64_t> hess = encoded_grad_hess.getHess();

  if (vole_cpsi_ptr->getRole() == psi::PsiRoleType::Receiver) {
    grad = vole_cpsi_ptr->adjustOrder(grad);
    hess = vole_cpsi_ptr->adjustOrder(hess);
  } else {
    grad.assign(vole_cpsi_ptr->getNumBin(), 0);
    hess.assign(vole_cpsi_ptr->getNumBin(), 0);
  }

  multiplex(grad, hess, mask);

  return GradHess(grad, hess);
}

GradHess AnonXGB::opprf_share(const GradHess &encoded_grad_hess, int pos) {
  auto vole_cpsi_ptr = pos == 0 ? vole_cpsi_ptr0_ : vole_cpsi_ptr1_;

  std::vector<uint64_t> grad = encoded_grad_hess.getGrad();
  std::vector<uint64_t> hess = encoded_grad_hess.getHess();

  if (vole_cpsi_ptr->getRole() == psi::PsiRoleType::Sender) {
    grad = vole_cpsi_ptr->OpprfShareSender(grad, false);
    hess = vole_cpsi_ptr->OpprfShareSender(hess, false);
  } else {
    grad = vole_cpsi_ptr->OpprfShareReceiver(false);
    hess = vole_cpsi_ptr->OpprfShareReceiver(false);
  }

  return GradHess(grad, hess);
}

void AnonXGB::init_histogram() {
  histogram_ptr0_->initialize(true);
  histogram_ptr1_->initialize(true);
}

void AnonXGB::init_histogram(std::shared_ptr<HistoGram> &histogram_ptr0,
                             std::shared_ptr<HistoGram> &histogram_ptr1) {
  histogram_ptr0_->set_ckks_params(histogram_ptr0);
  histogram_ptr1_->set_ckks_params(histogram_ptr1);
}

void AnonXGB::initialize(const std::vector<std::string> &items,
                         const std::vector<double> &labels, size_t cols,
                         bool init_ckks) {
  auto bgn_bytes =
      link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;

  if (init_ckks) {
    init_histogram();
  }

  self_rows_ = items.size();
  self_cols_ = cols;
  peer_rows_ = ExchangeSize(self_rows_);
  peer_cols_ = ExchangeSize(self_cols_);

  // printf("self_cols = %zu, peer_cols = %zu\n", self_cols_, peer_cols_);

  // printf("initialize begin");
  // link_ctx_->PrintStats();

  auto time_start = std::chrono::high_resolution_clock::now();
  auto ids = absl::MakeSpan(items);
  mask0_ = vole_cpsi_ptr0_->RunCPsi(ids);
  mask1_ = vole_cpsi_ptr1_->RunCPsi(ids);
  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
      time_end - time_start);
  std::cout << "two cpsi running of size " << items.size() << " takes "
            << time_diff.count() << "us" << std::endl;

  if (role_ == psi::PsiRoleType::Sender) {
    printf("after two cpsi running: ");
    link_ctx_->PrintStats();
  }

  //        auto mask0 = recoveryB<uint8_t>(mask0_);
  //        auto mask1 = recoveryB<uint8_t>(mask1_);
  //        uint64_t sum0 = std::accumulate(mask0.begin(), mask0.end(), 0ULL);
  //        uint64_t sum1 = std::accumulate(mask1.begin(), mask1.end(), 0ULL);
  //        SPU_ENFORCE(sum0 == sum1);

  if (role_ == psi::PsiRoleType::Sender) {
    SPU_ENFORCE(items.size() == labels.size());

    std::vector<uint64_t> encoded_labels = encode(labels, log2_scale_);
    label_ashr0_ = vole_cpsi_ptr0_->adjustOrder(encoded_labels);
    label_ashr0_ = vole_cpsi_ptr0_->multiplexer(label_ashr0_, mask0_, -1);
    label_ashr1_ = vole_cpsi_ptr1_->OpprfShareSender(encoded_labels, false);

    std::vector<uint64_t> encode_grad(items.size(), encode(0.5, log2_scale_) + 1);
    for (size_t i = 0; i < encode_grad.size(); i++) {
      encode_grad[i] = encode_grad[i] - encoded_labels[i];
    }
    std::vector<uint64_t> encode_hess(items.size(), encode(0.25, log2_scale_));

    GradHess encode_grad_hess = GradHess(encode_grad, encode_hess);

    grad_hess_ashr0_ = self_share(encode_grad_hess, 0);
    grad_hess_ashr1_ = opprf_share(encode_grad_hess, 1);
  } else {
    label_ashr0_.assign(vole_cpsi_ptr0_->getNumBin(), 0ULL);
    label_ashr0_ = vole_cpsi_ptr0_->multiplexer(label_ashr0_, mask0_, -1);
    label_ashr1_ = vole_cpsi_ptr1_->OpprfShareReceiver(false);

    GradHess encode_grad_hess;

    grad_hess_ashr0_ = self_share(encode_grad_hess, 0);
    grad_hess_ashr1_ = opprf_share(encode_grad_hess, 1);
  }

  tree_eval_ashr0_.assign(vole_cpsi_ptr0_->getNumBin(), 0ULL);
  tree_eval_ashr1_.assign(vole_cpsi_ptr1_->getNumBin(), 0ULL);

  // printf("initialize end: ");
  // link_ctx_->PrintStats();
  //            grad_hess_sum_test(grad_hess_ashr0_, grad_hess_ashr1_,
  //            "initialize");

  auto end_bytes =
      link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;
  std::cout << "initialize takes " << (end_bytes - bgn_bytes) << " comm bytes"
            << std::endl;
}

void AnonXGB::initialize_inference(const std::vector<std::string> &items,size_t cols) {
  SPU_ENFORCE(this->is_inference_==true, "only support inference");
  auto bgn_bytes =
      link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;

  self_rows_ = items.size();
  self_cols_ = cols;
  peer_rows_ = ExchangeSize(self_rows_);
  peer_cols_ = ExchangeSize(self_cols_);

  printf("self_cols = %zu, peer_cols = %zu\n", self_cols_, peer_cols_);

  // printf("initialize begin");
  // link_ctx_->PrintStats();

  auto time_start = std::chrono::high_resolution_clock::now();
  auto ids = absl::MakeSpan(items);
  mask0_ = vole_cpsi_ptr0_->RunCPsi(ids);
  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
      time_end - time_start);
  std::cout << "cpsi running of size " << items.size() << " takes "
            << time_diff.count() << "us" << std::endl;

  if (role_ == psi::PsiRoleType::Sender) {
    printf("after cpsi running: ");
    link_ctx_->PrintStats();
  }
  // printf("initialize end: ");
  // link_ctx_->PrintStats();
  //            grad_hess_sum_test(grad_hess_ashr0_, grad_hess_ashr1_,
  //            "initialize");

  auto end_bytes =
      link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;
  std::cout << "initialize takes " << (end_bytes - bgn_bytes) << " comm bytes"
            << std::endl;
}

void AnonXGB::grad_hess_sum_test(GradHess &grad_hess_ashr0,
                                 GradHess &grad_hess_ashr1,
                                 const std::string &str) {
  GradHess sum_grad_hess_ashr0 = grad_hess_ashr0.sum();
  GradHess sum_grad_hess_ashr1 = grad_hess_ashr1.sum();

  std::vector<uint64_t> sum_grad_hess0 = {sum_grad_hess_ashr0.getGrad()[0],
                                          sum_grad_hess_ashr0.getHess()[0]};

  std::vector<uint64_t> sum_grad_hess1 = {sum_grad_hess_ashr1.getGrad()[0],
                                          sum_grad_hess_ashr1.getHess()[0]};

  sum_grad_hess0 = recoveryA(sum_grad_hess0);
  sum_grad_hess1 = recoveryA(sum_grad_hess1);

  double sum_grad0 = decode(sum_grad_hess0[0], log2_scale_);
  double sum_hess0 = decode(sum_grad_hess0[1], log2_scale_);

  double sum_grad1 = decode(sum_grad_hess1[0], log2_scale_);
  double sum_hess1 = decode(sum_grad_hess1[1], log2_scale_);

  if (role_ == psi::PsiRoleType::Sender) {
    printf("%s, send sum grad: %f %f sum hess: %f %f\n", str.c_str(), sum_grad0,
           sum_grad1, sum_hess0, sum_hess1);
  } else {
    printf("%s, recv sum grad: %f %f sum hess: %f %f\n", str.c_str(), sum_grad0,
           sum_grad1, sum_hess0, sum_hess1);
  }
}

void AnonXGB::adjust_scale(GradHess &grad_hess_ashr, double new_scale) {
  double scale = std::pow(2, log2_scale_);

  auto grad = grad_hess_ashr.getGrad();
  auto hess = grad_hess_ashr.getHess();

  mul_plain(grad, scale / new_scale, log2_scale_, SignType::Unknown);
  mul_plain(hess, scale / new_scale, log2_scale_, SignType::Positive);

  grad_hess_ashr = GradHess(grad, hess);
}

std::vector<std::vector<uint8_t>> AnonXGB::adjust_split_bools(
    const std::vector<std::vector<uint8_t>> &split_bools,
    std::shared_ptr<VoleCPsi> vole_cpsi_ptr) {
  size_t cols = split_bools[0].size();

  std::vector<std::vector<uint8_t>> bin_values(vole_cpsi_ptr->getNumBin());
  if (vole_cpsi_ptr->getRole() == psi::PsiRoleType::Receiver) {
    for (size_t idx = 0; idx < bin_values.size(); idx++) {
      auto item_idx = vole_cpsi_ptr->getItemIndex(idx);
      SPU_ENFORCE(item_idx.size() <= 1, "only support cuckoo hash bin");
      if (!item_idx.empty()) {
        bin_values[idx] = split_bools[item_idx[0]];
      }
    }
  }

  for (auto &values : bin_values) {
    if (values.empty()) {
      values.assign(cols, 0);
    }
  }

  return bin_values;
}

double AnonXGB::compute_histogram(
    GradHess &grad_hess_ashr,
    const std::vector<std::vector<uint8_t>> &split_bools, int pos) {
  auto grad_ashr = grad_hess_ashr.getGrad();
  auto hess_ashr = grad_hess_ashr.getHess();

  double scale = std::pow(2, log2_scale_);
  auto histogram_ptr = pos == 0 ? histogram_ptr0_ : histogram_ptr1_;
  auto vole_cpsi_ptr = pos == 0 ? vole_cpsi_ptr0_ : vole_cpsi_ptr1_;

  if (histogram_ptr->getRole() == psi::PsiRoleType::Sender) {
    if (self_split_bools_.empty()) {
      self_split_bools_ = adjust_split_bools(split_bools, vole_cpsi_ptr);
    } else {
      SPU_ENFORCE(self_split_bools_.size() == vole_cpsi_ptr->getNumBin(),
                  "stored split bools size {} != {}", self_split_bools_.size(),
                  vole_cpsi_ptr->getNumBin());
    }

    //                printf("send%d rows = %zu, cols = %zu\n", pos,
    //                self_split_bools_.size(), self_split_bools_[0].size());
    scale = histogram_ptr->ComputeHistoGram(grad_ashr, hess_ashr, scale,
                                            self_split_bools_);
  } else {
    std::vector<std::vector<uint8_t>> zeros(vole_cpsi_ptr->getNumBin());
    for (auto &zero : zeros) {
      zero.resize(peer_cols_);
    }

    //                printf("recv%d rows = %zu, cols = %zu\n", pos,
    //                zeros.size(), zeros[0].size());
    scale = histogram_ptr->ComputeHistoGram(grad_ashr, hess_ashr, scale, zeros);
  }

  grad_hess_ashr = GradHess(grad_ashr, hess_ashr);
  return scale;
}

double AnonXGB::compute_histogram(
    std::vector<GradHess> &grad_hess_ashr_vec,
    const std::vector<std::vector<uint8_t>> &split_bools, int pos) {
  std::vector<std::vector<uint64_t>> grad_ashr_vec;
  std::vector<std::vector<uint64_t>> hess_ashr_vec;

  for (const auto &grad_hess_ashr : grad_hess_ashr_vec) {
    grad_ashr_vec.push_back(grad_hess_ashr.getGrad());
    hess_ashr_vec.push_back(grad_hess_ashr.getHess());
  }

  double scale = std::pow(2, log2_scale_);
  auto histogram_ptr = pos == 0 ? histogram_ptr0_ : histogram_ptr1_;
  auto vole_cpsi_ptr = pos == 0 ? vole_cpsi_ptr0_ : vole_cpsi_ptr1_;

  if (histogram_ptr->getRole() == psi::PsiRoleType::Sender) {
    self_split_bools_ = adjust_split_bools(split_bools, vole_cpsi_ptr);

    //            printf("send%d rows = %zu, cols = %zu\n", pos,
    //            self_split_bools_.size(), self_split_bools_[0].size());
    scale = histogram_ptr->ComputeHistoGram(grad_ashr_vec, hess_ashr_vec, scale,
                                            self_split_bools_);
  } else {
    std::vector<std::vector<uint8_t>> zeros(vole_cpsi_ptr->getNumBin());
    for (auto &zero : zeros) {
      zero.resize(peer_cols_);
    }

    //            printf("recv%d rows = %zu, cols = %zu\n", pos, zeros.size(),
    //            zeros[0].size());
    scale = histogram_ptr->ComputeHistoGram(grad_ashr_vec, hess_ashr_vec, scale,
                                            zeros);
  }

  for (size_t i = 0; i < grad_hess_ashr_vec.size(); i++) {
    grad_hess_ashr_vec[i] = GradHess(grad_ashr_vec[i], hess_ashr_vec[i]);
  }
  return scale;
}

size_t AnonXGB::split_node(const std::vector<std::vector<uint8_t>> &split_bools,
                           double lambda, const GradHess &grad_hess_ashr0,
                           const GradHess &grad_hess_ashr1) {
  double scale0 = 0.0;
  double scale1 = 0.0;

  GradHess lhist_gh_ashr0 = grad_hess_ashr0.clone();
  GradHess lhist_gh_ashr1 = grad_hess_ashr1.clone();

  scale0 = compute_histogram(lhist_gh_ashr0, split_bools, 0);
  scale1 = compute_histogram(lhist_gh_ashr1, split_bools, 1);

  adjust_scale(lhist_gh_ashr0, scale0);
  adjust_scale(lhist_gh_ashr1, scale1);

  //            grad_hess_sum_test(lhist_gh_ashr0, lhist_gh_ashr1, "split
  //            node");

  GradHess sum_gh_ashr0 = grad_hess_ashr0.sum();
  GradHess sum_gh_ashr1 = grad_hess_ashr1.sum();

  GradHess rhist_gh_ashr0 =
      sum_gh_ashr0.extend(lhist_gh_ashr0.size()) - lhist_gh_ashr0;
  GradHess rhist_gh_ashr1 =
      sum_gh_ashr1.extend(lhist_gh_ashr1.size()) - lhist_gh_ashr1;

  GradHess lhist_gh_ashr = GradHess::concat(lhist_gh_ashr0, lhist_gh_ashr1);
  GradHess rhist_gh_ashr = GradHess::concat(rhist_gh_ashr0, rhist_gh_ashr1);

  std::vector<uint64_t> gl_ashr = lhist_gh_ashr.getGrad();
  std::vector<uint64_t> hl_ashr = lhist_gh_ashr.getHess();
  std::vector<uint64_t> gr_ashr = rhist_gh_ashr.getGrad();
  std::vector<uint64_t> hr_ashr = rhist_gh_ashr.getHess();

  add_plain(hl_ashr, lambda, std::pow(2.0, log2_scale_));
  add_plain(hr_ashr, lambda, std::pow(2.0, log2_scale_));

  gl_ashr = square(gl_ashr);
  gr_ashr = square(gr_ashr);
  truncate(gl_ashr, log2_scale_, SignType::Positive);
  truncate(gr_ashr, log2_scale_, SignType::Positive);

  gl_ashr = mul(gl_ashr, hr_ashr);
  gr_ashr = mul(gr_ashr, hl_ashr);

  auto aa_ashr = add(gl_ashr, gr_ashr);
  auto bb_ashr = mul(hl_ashr, hr_ashr);

  truncate(aa_ashr, log2_scale_, SignType::Positive);
  truncate(bb_ashr, log2_scale_, SignType::Positive);

  auto rr_ashr = divide(aa_ashr, bb_ashr, SignType::Positive);

  auto values = recoveryA(rr_ashr);
  for (size_t i = 0; i < values.size(); i++) {
    printf("%f ", decode(values[i], log2_scale_));
  }
  printf("\n");

  auto idx_ashr = arg_max(rr_ashr);
  SPU_ENFORCE(idx_ashr.size() == 1);

  // sub plain
  if (operators_ptr_->getRole() == psi::PsiRoleType::Sender) {
    std::transform(idx_ashr.begin(), idx_ashr.end(), idx_ashr.begin(),
                   [&](uint64_t vv) { return vv - peer_cols_; });
  }

  auto is_left = msb(idx_ashr);
  is_left = this->recoveryB<uint8_t>(is_left);

  // add plain
  if (operators_ptr_->getRole() == psi::PsiRoleType::Sender) {
    std::transform(idx_ashr.begin(), idx_ashr.end(), idx_ashr.begin(),
                   [&](uint64_t vv) { return vv + peer_cols_; });
  }

  if (is_left[0] != 0U) {
    if (role_ == psi::PsiRoleType::Sender) {
      size_t peer_idx_ashr = psi::utils::DeserializeSize(link_ctx_->Recv(
          link_ctx_->NextRank(), fmt::format("AnonXGB:ArgMaxShare")));
      return peer_idx_ashr + idx_ashr[0];
    } else {
      link_ctx_->SendAsyncThrottled(
          link_ctx_->NextRank(), psi::utils::SerializeSize(idx_ashr[0]),
          fmt::format("AnonXGB:ArgMaxShare={}", idx_ashr[0]));
      return static_cast<uint64_t>(-1);
    }
  } else {
    if (role_ == psi::PsiRoleType::Receiver) {
      size_t peer_idx_ashr = psi::utils::DeserializeSize(link_ctx_->Recv(
          link_ctx_->NextRank(), fmt::format("AnonXGB:ArgMaxShare")));
      return peer_idx_ashr + idx_ashr[0] - peer_cols_;
    } else {
      link_ctx_->SendAsyncThrottled(
          link_ctx_->NextRank(), psi::utils::SerializeSize(idx_ashr[0]),
          fmt::format("AnonXGB:ArgMaxShare={}", idx_ashr[0]));
      return static_cast<uint64_t>(-1);
    }
  }
}

void AnonXGB::compute_histogram(
    std::vector<GradHess> &hist_gh_ashr_vec0,
    std::vector<GradHess> &hist_gh_ashr_vec1,
    const std::vector<std::vector<uint8_t>> &split_bools) {
  auto time_start = std::chrono::high_resolution_clock::now();

  SPU_ENFORCE(hist_gh_ashr_vec0.size() == hist_gh_ashr_vec1.size());

  //        double scale0 = compute_histogram(hist_gh_ashr_vec0, split_bools,
  //        0); double scale1 = compute_histogram(hist_gh_ashr_vec1,
  //        split_bools, 1);

  std::vector<std::vector<uint64_t>> grad_ashr_vec0;
  std::vector<std::vector<uint64_t>> hess_ashr_vec0;

  for (const auto &grad_hess_ashr0 : hist_gh_ashr_vec0) {
    grad_ashr_vec0.push_back(grad_hess_ashr0.getGrad());
    hess_ashr_vec0.push_back(grad_hess_ashr0.getHess());
  }

  std::vector<std::vector<uint64_t>> grad_ashr_vec1;
  std::vector<std::vector<uint64_t>> hess_ashr_vec1;

  for (const auto &grad_hess_ashr1 : hist_gh_ashr_vec1) {
    grad_ashr_vec1.push_back(grad_hess_ashr1.getGrad());
    hess_ashr_vec1.push_back(grad_hess_ashr1.getHess());
  }

  double scale = std::pow(2, log2_scale_);
  double scale0 = 0.0;
  double scale1 = 0.0;

  if (role_ == psi::PsiRoleType::Sender) {
    yacl::Buffer buf1 =
        histogram_ptr1_->CKKSEncrypt(grad_ashr_vec1, hess_ashr_vec1, scale);

    yacl::Buffer buf0 = link_ctx_->Recv(
        link_ctx_->NextRank(), fmt::format("recv encrypted grad0 and hess0"));

    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf1,
        fmt::format("send encrypted grad1 and hess1 of size {} bytes",
                    buf1.size()));

    self_split_bools_ = adjust_split_bools(split_bools, vole_cpsi_ptr0_);
    scale0 = histogram_ptr0_->HomCompHistoGram(grad_ashr_vec0, hess_ashr_vec0,
                                               buf0, scale, self_split_bools_);

    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf0,
        fmt::format("send summed grad0 and hess0 of size {} bytes",
                    buf0.size()));

    buf1 = link_ctx_->Recv(link_ctx_->NextRank(),
                           fmt::format("recv summed grad1 and hess1"));

    scale1 = histogram_ptr1_->PostProcess(grad_ashr_vec1, hess_ashr_vec1, buf1,
                                          peer_cols_);

  } else {
    yacl::Buffer buf0 =
        histogram_ptr0_->CKKSEncrypt(grad_ashr_vec0, hess_ashr_vec0, scale);

    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf0,
        fmt::format("send encrypted grad0 and hess0 of size {} bytes",
                    buf0.size()));

    yacl::Buffer buf1 = link_ctx_->Recv(
        link_ctx_->NextRank(), fmt::format("recv encrypted grad1 and hess1"));

    self_split_bools_ = adjust_split_bools(split_bools, vole_cpsi_ptr1_);
    scale1 = histogram_ptr1_->HomCompHistoGram(grad_ashr_vec1, hess_ashr_vec1,
                                               buf1, scale, self_split_bools_);

    buf0 = link_ctx_->Recv(link_ctx_->NextRank(),
                           fmt::format("recv summed grad0 and hess0"));

    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf1,
        fmt::format("send summed grad1 and hess1 of size {} bytes",
                    buf1.size()));

    scale0 = histogram_ptr0_->PostProcess(grad_ashr_vec0, hess_ashr_vec0, buf0,
                                          peer_cols_);
  }

  for (size_t i = 0; i < hist_gh_ashr_vec0.size(); i++) {
    hist_gh_ashr_vec0[i] = GradHess(grad_ashr_vec0[i], hess_ashr_vec0[i]);
  }

  for (size_t i = 0; i < hist_gh_ashr_vec1.size(); i++) {
    hist_gh_ashr_vec1[i] = GradHess(grad_ashr_vec1[i], hess_ashr_vec1[i]);
  }

  for (size_t i = 0; i < hist_gh_ashr_vec0.size(); i++) {
    adjust_scale(hist_gh_ashr_vec0[i], scale0);
    adjust_scale(hist_gh_ashr_vec1[i], scale1);

    //                grad_hess_sum_test(hist_gh_ashr_vec0[i],
    //                hist_gh_ashr_vec1[i], "compute histogram");
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
      time_end - time_start);

  // if (role_ == psi::PsiRoleType::Sender) {
  //   std::cout << "after compute_histogram: takes " << time_diff.count()
  //             << " us ";
  //   link_ctx_->PrintStats();
  // }
}

std::vector<size_t> AnonXGB::find_optimal_cols(
    double lambda, const std::vector<GradHess> &lhist_ashr_vec0,
    const std::vector<GradHess> &lhist_ashr_vec1,
    const std::vector<GradHess> &sum_gh_ashr_vec0,
    const std::vector<GradHess> &sum_gh_ashr_vec1) {
  auto time_start = std::chrono::high_resolution_clock::now();

  SPU_ENFORCE(lhist_ashr_vec0.size() == lhist_ashr_vec1.size());
  SPU_ENFORCE(sum_gh_ashr_vec0.size() == sum_gh_ashr_vec1.size());
  SPU_ENFORCE(lhist_ashr_vec0.size() == sum_gh_ashr_vec1.size());

  std::vector<uint64_t> gl_ashr_vec;
  std::vector<uint64_t> hl_ashr_vec;
  std::vector<uint64_t> gr_ashr_vec;
  std::vector<uint64_t> hr_ashr_vec;

  for (size_t i = 0; i < lhist_ashr_vec0.size(); i++) {
    GradHess sum_gh_ashr0 = sum_gh_ashr_vec0[i];
    GradHess sum_gh_ashr1 = sum_gh_ashr_vec1[i];

    GradHess rhist_gh_ashr0 =
        sum_gh_ashr0.extend(lhist_ashr_vec0[i].size()) - lhist_ashr_vec0[i];
    GradHess rhist_gh_ashr1 =
        sum_gh_ashr1.extend(lhist_ashr_vec1[i].size()) - lhist_ashr_vec1[i];

    GradHess lhist_gh_ashr =
        GradHess::concat(lhist_ashr_vec0[i], lhist_ashr_vec1[i]);
    GradHess rhist_gh_ashr = GradHess::concat(rhist_gh_ashr0, rhist_gh_ashr1);

    auto gl_ashr = lhist_gh_ashr.getGrad();
    auto hl_ashr = lhist_gh_ashr.getHess();
    auto gr_ashr = rhist_gh_ashr.getGrad();
    auto hr_ashr = rhist_gh_ashr.getHess();

    add_plain(hl_ashr, lambda, std::pow(2.0, log2_scale_));
    add_plain(hr_ashr, lambda, std::pow(2.0, log2_scale_));

    gl_ashr_vec.insert(gl_ashr_vec.end(), gl_ashr.begin(), gl_ashr.end());
    hl_ashr_vec.insert(hl_ashr_vec.end(), hl_ashr.begin(), hl_ashr.end());
    gr_ashr_vec.insert(gr_ashr_vec.end(), gr_ashr.begin(), gr_ashr.end());
    hr_ashr_vec.insert(hr_ashr_vec.end(), hr_ashr.begin(), hr_ashr.end());
  }

  auto aa_ashr = square_and_divide(gl_ashr_vec, hl_ashr_vec);
  auto bb_ashr = square_and_divide(gr_ashr_vec, hr_ashr_vec);
  auto rr_ashr = add(aa_ashr, bb_ashr);

  auto idx_ashr_vec = arg_max(rr_ashr, self_cols_ + peer_cols_);
  auto is_left_vec = is_left(idx_ashr_vec);

  std::vector<uint64_t> values(is_left_vec.size());
  std::transform(
      values.begin(), values.end(), values.begin(), [&](uint64_t vv) {
        return (role_ == psi::PsiRoleType::Sender) ? static_cast<uint64_t>(-1)
                                                   : 0ULL;
      });

  auto idx0_ashr = mux(is_left_vec, values, idx_ashr_vec);
  auto idx1_ashr = mux(is_left_vec, idx_ashr_vec, values);

  values.resize(idx0_ashr.size() + idx1_ashr.size());
  std::copy_n(idx0_ashr.data(), idx0_ashr.size(), values.data());
  std::copy_n(idx1_ashr.data(), idx1_ashr.size(),
              values.data() + idx0_ashr.size());

  values = recoveryA(values);

  std::vector<size_t> res(idx0_ashr.size());
  for (size_t i = 0; i < res.size(); i++) {
    size_t j = i + res.size();
    bool bi = values[i] == static_cast<uint64_t>(-1);
    bool bj = values[j] == static_cast<uint64_t>(-1);

    SPU_ENFORCE(bi || (values[i] < (self_cols_ + peer_cols_)), "value = {}",
                values[i]);
    SPU_ENFORCE(bj || (values[j] < (self_cols_ + peer_cols_)), "value = {}",
                values[j]);
    SPU_ENFORCE(((!bi) && bj) || (bi && (!bj)));

    if (role_ == psi::PsiRoleType::Sender) {
      res[i] = values[i];
    } else {
      res[i] = bj ? values[j] : values[j] - peer_cols_;
    }
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
      time_end - time_start);

  if (role_ == psi::PsiRoleType::Sender) {
    std::cout << "after find_optimal_cols: takes " << time_diff.count()
              << " us ";
    link_ctx_->PrintStats();
  }

  return res;
}

void AnonXGB::init_bool_vectors(
        std::vector<uint64_t>& bool0,
        std::vector<uint64_t>& bool1,
        std::vector<size_t>& idx_pos_arr0,
        std::vector<size_t>& idx_pos_arr1,
        const std::vector<std::vector<uint8_t>>& split_bools,
        const std::vector<size_t>& index_arr) {
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
        bool1.resize(self_rows_, 0ULL);

        for (size_t l = 0; l < index_arr.size(); l++) {
            size_t index = index_arr[l];
            if (index != static_cast<uint64_t>(-1)) {
                SPU_ENFORCE(index < self_cols_);
                SPU_ENFORCE(l < 64, "only support 64-parallel opprf-share");

                for (size_t i = 0; i < bool1.size(); i++) {
                    bool1[i] ^= static_cast<uint64_t>(split_bools[i][index]) << l;
                }

                idx_pos_arr1.push_back(l);
            } else {
                idx_pos_arr0.push_back(l);
            }
        }
    }
}

void AnonXGB::process_both_non_empty(
        std::vector<std::vector<uint8_t>>& bool0_vec,
        std::vector<std::vector<uint8_t>>& bool1_vec,
        std::vector<uint64_t>& bool0,
        std::vector<uint64_t>& bool1,
        const std::vector<size_t>& idx_pos_arr0,
        const std::vector<size_t>& idx_pos_arr1) {
    std::vector<uint64_t> selfs0;
    std::vector<uint64_t> peers0;
    std::vector<uint64_t> selfs1;
    std::vector<uint64_t> peers1;

    if (role_ == psi::PsiRoleType::Sender) {
        selfs0 = vole_cpsi_ptr0_->adjustOrder(bool0);
        selfs0 = vole_cpsi_ptr0_->bmultiplexer(selfs0, mask0_, -1);
        selfs1.assign(vole_cpsi_ptr1_->getNumBin(), 0ULL);
        selfs1 = vole_cpsi_ptr1_->bmultiplexer(selfs1, mask1_, -1);

        vole_cpsi_ptr1_->OpprfSharePrevProc();
        vole_cpsi_ptr0_->OpprfSharePrevProc();

        yacl::Buffer gc_buf0 = vole_cpsi_ptr1_->OpprfShareSendProc(bool0, true);

        link_ctx_->SendAsyncThrottled(
                link_ctx_->NextRank(), gc_buf0,
                fmt::format("send opprf-share-table0 of size {}", gc_buf0.size()));

        yacl::Buffer gc_buf1 = link_ctx_->Recv(
                link_ctx_->NextRank(), fmt::format("recv opprf-share-table1"));

        peers1 = vole_cpsi_ptr0_->OpprfShareRecvProc(gc_buf1);

        peers0 = vole_cpsi_ptr1_->OpprfSharePostProc(bool0, true);
        peers1 = vole_cpsi_ptr0_->OpprfSharePostProc(peers1, true);

    } else {
        selfs0.assign(vole_cpsi_ptr0_->getNumBin(), 0ULL);
        selfs0 = vole_cpsi_ptr0_->bmultiplexer(selfs0, mask0_, -1);
        selfs1 = vole_cpsi_ptr1_->adjustOrder(bool1);
        selfs1 = vole_cpsi_ptr1_->bmultiplexer(selfs1, mask1_, -1);

        vole_cpsi_ptr1_->OpprfSharePrevProc();
        vole_cpsi_ptr0_->OpprfSharePrevProc();

        yacl::Buffer gc_buf1 = vole_cpsi_ptr0_->OpprfShareSendProc(bool1, true);

        yacl::Buffer gc_buf0 = link_ctx_->Recv(
                link_ctx_->NextRank(), fmt::format("recv opprf-share-table0"));

        link_ctx_->SendAsyncThrottled(
                link_ctx_->NextRank(), gc_buf1,
                fmt::format("send opprf-share-table1 of size {}", gc_buf1.size()));

        peers0 = vole_cpsi_ptr1_->OpprfShareRecvProc(gc_buf0);

        peers0 = vole_cpsi_ptr1_->OpprfSharePostProc(peers0, true);
        peers1 = vole_cpsi_ptr0_->OpprfSharePostProc(bool1, true);
    }

    for (const auto& pos : idx_pos_arr0) {
        bool0_vec[pos].resize(selfs0.size());
        std::transform(selfs0.begin(), selfs0.end(), bool0_vec[pos].begin(),
                       [&](uint64_t vv) { return (vv >> pos) & 0x1; });

        bool1_vec[pos].resize(peers0.size());
        std::transform(peers0.begin(), peers0.end(), bool1_vec[pos].begin(),
                       [&](uint64_t vv) { return (vv >> pos) & 0x1; });
    }

    for (const auto& pos : idx_pos_arr1) {
        bool0_vec[pos].resize(peers1.size());
        std::transform(peers1.begin(), peers1.end(), bool0_vec[pos].begin(),
                       [&](uint64_t vv) { return (vv >> pos) & 0x1; });

        bool1_vec[pos].resize(selfs1.size());
        std::transform(selfs1.begin(), selfs1.end(), bool1_vec[pos].begin(),
                       [&](uint64_t vv) { return (vv >> pos) & 0x1; });
    }
}

void AnonXGB::process_single_non_empty(
        std::vector<std::vector<uint8_t>>& bool0_vec,
        std::vector<std::vector<uint8_t>>& bool1_vec,
        const std::vector<uint64_t>& bool0,
        const std::vector<uint64_t>& bool1,
        const std::vector<size_t>& idx_pos_arr0,
        const std::vector<size_t>& idx_pos_arr1) {
    std::vector<uint64_t> selfs;
    std::vector<uint64_t> peers;

    if (!idx_pos_arr0.empty()) {
        if (role_ == psi::PsiRoleType::Sender) {
            selfs = vole_cpsi_ptr0_->adjustOrder(bool0);
            selfs = vole_cpsi_ptr0_->bmultiplexer(selfs, mask0_, -1);
            peers = vole_cpsi_ptr1_->OpprfShareSender(bool0, true);
        } else {
            selfs.assign(vole_cpsi_ptr0_->getNumBin(), 0ULL);
            selfs = vole_cpsi_ptr0_->bmultiplexer(selfs, mask0_, -1);
            peers = vole_cpsi_ptr1_->OpprfShareReceiver(true);
        }
    }

    for (const auto& pos : idx_pos_arr0) {
        bool0_vec[pos].resize(selfs.size());
        std::transform(selfs.begin(), selfs.end(), bool0_vec[pos].begin(),
                       [&](uint64_t vv) { return (vv >> pos) & 0x1; });

        bool1_vec[pos].resize(peers.size());
        std::transform(peers.begin(), peers.end(), bool1_vec[pos].begin(),
                       [&](uint64_t vv) { return (vv >> pos) & 0x1; });
    }

    if (!idx_pos_arr1.empty()) {
        if (role_ == psi::PsiRoleType::Receiver) {
            selfs = vole_cpsi_ptr1_->adjustOrder(bool1);
            selfs = vole_cpsi_ptr1_->bmultiplexer(selfs, mask1_, -1);
            peers = vole_cpsi_ptr0_->OpprfShareSender(bool1, true);
        } else {
            selfs.assign(vole_cpsi_ptr1_->getNumBin(), 0ULL);
            selfs = vole_cpsi_ptr1_->bmultiplexer(selfs, mask1_, -1);
            peers = vole_cpsi_ptr0_->OpprfShareReceiver(true);
        }
    }

    for (const auto& pos : idx_pos_arr1) {
        bool0_vec[pos].resize(peers.size());
        std::transform(peers.begin(), peers.end(), bool0_vec[pos].begin(),
                       [&](uint64_t vv) { return (vv >> pos) & 0x1; });

        bool1_vec[pos].resize(selfs.size());
        std::transform(selfs.begin(), selfs.end(), bool1_vec[pos].begin(),
                       [&](uint64_t vv) { return (vv >> pos) & 0x1; });
    }
}

void AnonXGB::share_bool_splits(
        std::vector<std::vector<uint8_t>>& bool0_vec,
        std::vector<std::vector<uint8_t>>& bool1_vec,
        const std::vector<std::vector<uint8_t>>& split_bools,
        const std::vector<size_t>& index_arr) {
    auto time_start = std::chrono::high_resolution_clock::now();

    std::vector<uint64_t> bool0;
    std::vector<uint64_t> bool1;
    std::vector<size_t> idx_pos_arr0;
    std::vector<size_t> idx_pos_arr1;

    init_bool_vectors(bool0, bool1, idx_pos_arr0, idx_pos_arr1, split_bools, index_arr);

    bool0_vec.resize(index_arr.size());
    bool1_vec.resize(index_arr.size());

    std::atomic<size_t> comm_bytes =
            link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;

    if (!idx_pos_arr0.empty() && !idx_pos_arr1.empty()) {
        process_both_non_empty(bool0_vec, bool1_vec, bool0, bool1, idx_pos_arr0, idx_pos_arr1);
    } else {
        process_single_non_empty(bool0_vec, bool1_vec, bool0, bool1, idx_pos_arr0, idx_pos_arr1);
    }

    auto time_end = std::chrono::high_resolution_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
            time_end - time_start);

    if (role_ == psi::PsiRoleType::Sender) {
        comm_bytes = (link_ctx_->GetStats()->sent_bytes +
                      link_ctx_->GetStats()->recv_bytes) - comm_bytes;
        std::cout << "after share_bool_splits: take " << time_diff.count()
                  << " us, comm_bytes = " << comm_bytes << std::endl;
    }
}

void AnonXGB::prepare_bool_vectors(
        std::vector<uint64_t>& bool0, std::vector<uint64_t>& bool1,
        std::vector<size_t>& idx_pos_arr0, std::vector<size_t>& idx_pos_arr1,
        const std::vector<std::vector<uint8_t>>& split_bools,
        const std::vector<size_t>& index_arr) {

    if (role_ == psi::PsiRoleType::Sender) {
        bool0.resize(self_rows_, 0ULL);
        for (size_t l = 0; l < index_arr.size(); l++) {
            size_t index = index_arr[l];
            if (index != static_cast<uint64_t>(-1)) {
                SPU_ENFORCE(index < self_cols_);
                SPU_ENFORCE(l < 64, "only support 64-parallel opprf-share");
                SPU_ENFORCE(split_bools.size() == self_rows_);

                for (size_t i = 0; i < bool0.size(); i++) {
                    bool0[i] ^= static_cast<uint64_t>(split_bools[i][index]) << l;
                }

                idx_pos_arr0.push_back(l);
            } else {
                idx_pos_arr1.push_back(l);
            }
        }
    } else {
        bool1.resize(self_rows_, 0ULL);
        for (size_t l = 0; l < index_arr.size(); l++) {
            size_t index = index_arr[l];
            if (index != static_cast<uint64_t>(-1)) {
                SPU_ENFORCE(index < self_cols_);
                SPU_ENFORCE(l < 64, "only support 64-parallel opprf-share");
                SPU_ENFORCE(split_bools.size() == self_rows_);

                for (size_t i = 0; i < bool1.size(); i++)
                    bool1[i] ^= static_cast<uint64_t>(split_bools[i][index]) << l;

                idx_pos_arr1.push_back(l);
            } else {
                idx_pos_arr0.push_back(l);
            }
        }
    }
}

std::array<std::vector<std::vector<uint8_t>>, 2>
AnonXGB::split_bool_vec_from_bools(const std::vector<uint64_t>& selfs,
                                   const std::vector<uint64_t>& peers,
                                   const std::vector<size_t>& idx_pos_arr) {
    std::vector<std::vector<uint8_t>> self_bools(idx_pos_arr.size());
    std::vector<std::vector<uint8_t>> peer_bools(idx_pos_arr.size());

    for (size_t i = 0; i < idx_pos_arr.size(); i++) {
        self_bools[i].resize(selfs.size());
        peer_bools[i].resize(peers.size());

        size_t pos = idx_pos_arr[i];
        std::transform(selfs.begin(), selfs.end(), self_bools[i].begin(),
                       [&](uint64_t vv) { return (vv >> pos) & 0x1; });
        std::transform(peers.begin(), peers.end(), peer_bools[i].begin(),
                       [&](uint64_t vv) { return (vv >> pos) & 0x1; });
    }

    return {self_bools, peer_bools};
}

void AnonXGB::process_both_non_empty(
        std::vector<uint64_t>& bool0, std::vector<uint64_t>& bool1,
        std::vector<size_t>& idx_pos_arr0, std::vector<size_t>& idx_pos_arr1,
        std::vector<GradHess>& gh_ashr0, std::vector<GradHess>& gh_ashr1,
        size_t& comm_bytes) {

    std::vector<uint64_t> selfs0, peers0;
    std::vector<uint64_t> selfs1, peers1;

    if (role_ == psi::PsiRoleType::Sender) {
        selfs0 = vole_cpsi_ptr0_->adjustOrder(bool0);
        selfs1.assign(vole_cpsi_ptr1_->getNumBin(), 0ULL);

        vole_cpsi_ptr1_->OpprfSharePrevProc();
        vole_cpsi_ptr0_->OpprfSharePrevProc();

        yacl::Buffer gc_buf0 = vole_cpsi_ptr1_->OpprfShareSendProc(bool0, true);
        link_ctx_->SendAsyncThrottled(
                link_ctx_->NextRank(), gc_buf0,
                fmt::format("send opprf-share-table0 of size {}", gc_buf0.size()));

        yacl::Buffer gc_buf1 = link_ctx_->Recv(
                link_ctx_->NextRank(), fmt::format("recv opprf-share-table1"));
        peers1 = vole_cpsi_ptr0_->OpprfShareRecvProc(gc_buf1);

        peers0 = vole_cpsi_ptr1_->OpprfSharePostProc(bool0, true);
        peers1 = vole_cpsi_ptr0_->OpprfSharePostProc(peers1, true);
    } else {
        selfs0.assign(vole_cpsi_ptr0_->getNumBin(), 0ULL);
        selfs1 = vole_cpsi_ptr1_->adjustOrder(bool1);

        vole_cpsi_ptr1_->OpprfSharePrevProc();
        vole_cpsi_ptr0_->OpprfSharePrevProc();

        yacl::Buffer gc_buf1 = vole_cpsi_ptr0_->OpprfShareSendProc(bool1, true);
        yacl::Buffer gc_buf0 = link_ctx_->Recv(
                link_ctx_->NextRank(), fmt::format("recv opprf-share-table0"));

        link_ctx_->SendAsyncThrottled(
                link_ctx_->NextRank(), gc_buf1,
                fmt::format("send opprf-share-table1 of size {}", gc_buf1.size()));

        peers0 = vole_cpsi_ptr1_->OpprfShareRecvProc(gc_buf0);
        peers0 = vole_cpsi_ptr1_->OpprfSharePostProc(peers0, true);
        peers1 = vole_cpsi_ptr0_->OpprfSharePostProc(bool1, true);
    }

    sync_indicator_comm_bytes += link_ctx_->GetStats()->sent_bytes +
                                 link_ctx_->GetStats()->recv_bytes - comm_bytes;
    comm_bytes = link_ctx_->GetStats()->sent_bytes +
                 link_ctx_->GetStats()->recv_bytes;

    // Prepare grad/hess vectors for multiplexing
    std::vector<GradHess> self_gh_ashr0, peer_gh_ashr0;
    std::vector<GradHess> self_gh_ashr1, peer_gh_ashr1;

    for (const auto& pos : idx_pos_arr0) {
        self_gh_ashr0.push_back(gh_ashr0[pos]);
        peer_gh_ashr0.push_back(gh_ashr1[pos]);
    }

    for (const auto& pos : idx_pos_arr1) {
        peer_gh_ashr1.push_back(gh_ashr0[pos]);
        self_gh_ashr1.push_back(gh_ashr1[pos]);
    }

    // Split bool vectors and multiplex
    auto [self_bools0, peer_bools0] =
            split_bool_vec_from_bools(selfs0, peers0, idx_pos_arr0);
    auto [self_bools1, peer_bools1] =
            split_bool_vec_from_bools(selfs1, peers1, idx_pos_arr1);

    multiplex(self_gh_ashr0, self_bools0, 1);
    multiplex(peer_gh_ashr0, peer_bools0);
    multiplex(self_gh_ashr1, self_bools1, -1);
    multiplex(peer_gh_ashr1, peer_bools1);

    // Update results
    for (size_t i = 0; i < idx_pos_arr0.size(); i++) {
        size_t pos = idx_pos_arr0[i];
        gh_ashr0[pos] = self_gh_ashr0[i];
        gh_ashr1[pos] = peer_gh_ashr0[i];
    }

    for (size_t i = 0; i < idx_pos_arr1.size(); i++) {
        size_t pos = idx_pos_arr1[i];
        gh_ashr0[pos] = peer_gh_ashr1[i];
        gh_ashr1[pos] = self_gh_ashr1[i];
    }

    update_indicator_comm_bytes +=
            link_ctx_->GetStats()->sent_bytes +
            link_ctx_->GetStats()->recv_bytes - comm_bytes;
}

void AnonXGB::process_single_array(
        bool is_arr0,
        std::vector<uint64_t>& bools,
        const std::vector<size_t>& idx_pos_arr,
        std::vector<GradHess>& gh_ashr0,
        std::vector<GradHess>& gh_ashr1,
        size_t& comm_bytes) {

    std::vector<uint64_t> selfs, peers;

    if (is_arr0) {
        if (role_ == psi::PsiRoleType::Sender) {
            selfs = vole_cpsi_ptr0_->adjustOrder(bools);
            peers = vole_cpsi_ptr1_->OpprfShareSender(bools, true);
        } else {
            selfs.assign(vole_cpsi_ptr0_->getNumBin(), 0ULL);
            peers = vole_cpsi_ptr1_->OpprfShareReceiver(true);
        }
    } else {
        if (role_ == psi::PsiRoleType::Receiver) {
            selfs = vole_cpsi_ptr1_->adjustOrder(bools);
            peers = vole_cpsi_ptr0_->OpprfShareSender(bools, true);
        } else {
            selfs.assign(vole_cpsi_ptr1_->getNumBin(), 0ULL);
            peers = vole_cpsi_ptr0_->OpprfShareReceiver(true);
        }
    }

    sync_indicator_comm_bytes +=
            link_ctx_->GetStats()->sent_bytes +
            link_ctx_->GetStats()->recv_bytes - comm_bytes;
    comm_bytes = link_ctx_->GetStats()->sent_bytes +
                 link_ctx_->GetStats()->recv_bytes;

    std::vector<GradHess> self_gh_ashr, peer_gh_ashr;

    if (is_arr0) {
        for (const auto& pos : idx_pos_arr) {
            self_gh_ashr.push_back(gh_ashr0[pos]);
            peer_gh_ashr.push_back(gh_ashr1[pos]);
        }
    } else {
        for (const auto& pos : idx_pos_arr) {
            peer_gh_ashr.push_back(gh_ashr0[pos]);
            self_gh_ashr.push_back(gh_ashr1[pos]);
        }
    }

    auto [self_bools, peer_bools] =
            split_bool_vec_from_bools(selfs, peers, idx_pos_arr);

    if (is_arr0) {
        multiplex(self_gh_ashr, self_bools, 1);
    } else {
        multiplex(self_gh_ashr, self_bools, -1);
    }
    multiplex(peer_gh_ashr, peer_bools);

    for (size_t i = 0; i < idx_pos_arr.size(); i++) {
        size_t pos = idx_pos_arr[i];
        if (is_arr0) {
            gh_ashr0[pos] = self_gh_ashr[i];
            gh_ashr1[pos] = peer_gh_ashr[i];
        } else {
            gh_ashr0[pos] = peer_gh_ashr[i];
            gh_ashr1[pos] = self_gh_ashr[i];
        }
    }

    update_indicator_comm_bytes +=
            link_ctx_->GetStats()->sent_bytes +
            link_ctx_->GetStats()->recv_bytes - comm_bytes;
}

void AnonXGB::share_bool_splits_and_multiplex(
        std::vector<GradHess>& gh_ashr0, std::vector<GradHess>& gh_ashr1,
        const std::vector<std::vector<uint8_t>>& split_bools,
        const std::vector<size_t>& index_arr) {

    std::vector<uint64_t> bool0, bool1;
    std::vector<size_t> idx_pos_arr0, idx_pos_arr1;

    prepare_bool_vectors(bool0, bool1, idx_pos_arr0, idx_pos_arr1,
                         split_bools, index_arr);

    size_t comm_bytes = link_ctx_->GetStats()->sent_bytes +
                        link_ctx_->GetStats()->recv_bytes;

    if (!idx_pos_arr0.empty() && !idx_pos_arr1.empty()) {
        process_both_non_empty(bool0, bool1, idx_pos_arr0, idx_pos_arr1,
                               gh_ashr0, gh_ashr1, comm_bytes);
    } else {
        if (!idx_pos_arr0.empty()) {
            process_single_array(true, bool0, idx_pos_arr0,
                                 gh_ashr0, gh_ashr1, comm_bytes);
        }
        if (!idx_pos_arr1.empty()) {
            process_single_array(false, bool1, idx_pos_arr1,
                                 gh_ashr0, gh_ashr1, comm_bytes);
        }
    }
}

void AnonXGB::multiplex(std::vector<GradHess> &grad_hess_ashr_vec,
                        const std::vector<std::vector<uint8_t>> &mask_vec,
                        int type) {
  SPU_ENFORCE(grad_hess_ashr_vec.size() == mask_vec.size(), "{} != {}",
              grad_hess_ashr_vec.size(), mask_vec.size());

  size_t size = 0;
  for (size_t i = 0; i < mask_vec.size(); i++) {
    SPU_ENFORCE(grad_hess_ashr_vec[i].size() == mask_vec[i].size());
    size += mask_vec[i].size();
  }

  std::vector<uint8_t> mask(size);
  size_t offset = 0;
  for (const auto &vv : mask_vec) {
    std::copy_n(vv.data(), vv.size(), mask.data() + offset);
    offset += vv.size();
  }

  GradHess grad_hess_ashr = GradHess::concat(grad_hess_ashr_vec);
  grad_hess_ashr = multiplex(grad_hess_ashr, mask, type);

  offset = 0;
  for (size_t i = 0; i < grad_hess_ashr_vec.size(); i++) {
    grad_hess_ashr_vec[i] =
      grad_hess_ashr.slice(offset, offset + grad_hess_ashr_vec[i].size());
    offset += grad_hess_ashr_vec[i].size();
  }
}

std::vector<uint64_t> AnonXGB::multiplex_single(
    const std::vector<uint64_t> &shares,
    const std::vector<std::vector<uint8_t>> &mask_vec) {
  SPU_ENFORCE(shares.size() == mask_vec.size(), "{} != {}", shares.size(),
              mask_vec.size());

  size_t total_size = std::accumulate(
    mask_vec.begin(), mask_vec.end(), 0, [&](size_t sum, const vector<uint8_t> &sub_vec) {
      return sum + sub_vec.size();
  });

  std::vector<uint64_t> values(total_size);
  std::vector<uint8_t> mask(total_size);

  size_t size = 0;
  for (size_t i = 0; i < mask_vec.size(); i++) {
    std::fill_n(values.data() + size, mask_vec[i].size(), shares[i]);
    std::copy_n(mask_vec[i].data(), mask_vec[i].size(), mask.data() + size);

    size += mask_vec[i].size();
  }

  values = multiplex(mask, values);

  size = mask_vec[0].size();
  for (size_t i = 1; i < mask_vec.size(); i++) {
    SPU_ENFORCE(size == mask_vec[i].size());
    for (size_t j = 0; j < size; j++) {
      values[j] += values[j + i * size];
    }
  }

  values.resize(size);
  return values;
}

void AnonXGB::update_tree_eval(
    const std::vector<std::vector<uint8_t>> &leaf_mask_bshr_vec0,
    const std::vector<std::vector<uint8_t>> &leaf_mask_bshr_vec1,
    const std::vector<uint64_t> &leaf_wt_ashr) {
  //        auto tree_eval_ashr0 = multiplex_single(leaf_wt_ashr,
  //        leaf_mask_bshr_vec0); auto tree_eval_ashr1 =
  //        multiplex_single(leaf_wt_ashr, leaf_mask_bshr_vec1);

  auto concat = [&](const std::vector<uint8_t> &aa,
                    const std::vector<uint8_t> &bb) {
    std::vector<uint8_t> res(aa.begin(), aa.end());
    res.insert(res.end(), bb.begin(), bb.end());
    return res;
  };

  SPU_ENFORCE(leaf_wt_ashr.size() == leaf_mask_bshr_vec0.size() &&
                  leaf_wt_ashr.size() == leaf_mask_bshr_vec1.size(),
              "{}, {} and {} should be equal", leaf_wt_ashr.size(),
              leaf_mask_bshr_vec0.size(), leaf_mask_bshr_vec1.size());


  std::vector<std::vector<uint8_t>> leaf_mask_bshr(leaf_wt_ashr.size());
  for (size_t j = 0; j < leaf_mask_bshr.size(); j++) {
    leaf_mask_bshr[j] = concat(leaf_mask_bshr_vec0[j], leaf_mask_bshr_vec1[j]);
  }

  auto tree_eval_ashr = multiplex_single(leaf_wt_ashr, leaf_mask_bshr);

  SPU_ENFORCE(tree_eval_ashr0_.size() + tree_eval_ashr1_.size() ==
                  tree_eval_ashr.size(),
              "{} + {} != {}", tree_eval_ashr0_.size(), tree_eval_ashr1_.size(),
              tree_eval_ashr.size());
  for (size_t i = 0; i < tree_eval_ashr0_.size(); i++) {
    tree_eval_ashr0_[i] += tree_eval_ashr[i];
  }

  for (size_t i = 0; i < tree_eval_ashr1_.size(); i++) {
    tree_eval_ashr1_[i] += tree_eval_ashr[i + tree_eval_ashr0_.size()];
  }
}

std::vector<uint8_t> AnonXGB::msb(const std::vector<uint64_t> &shares, size_t nbits) {
  NdArrayRef aa = operators_ptr_->from(shares);
  aa = operators_ptr_->asAShares(aa);
  aa = operators_ptr_->MSB(aa, nbits);

  std::vector<uint64_t> tmp = operators_ptr_->to(aa);
  std::vector<uint8_t> res(tmp.size());
  std::transform(tmp.begin(), tmp.end(), res.begin(),
                 [&](uint64_t vv) { return vv & 0x1; });

  return res;
}

std::vector<uint8_t> AnonXGB::msbsigmoid(const std::vector<uint64_t> &shares,
                                         uint64_t low_seg, uint64_t high_seg,
                                         size_t skip_bits, size_t nbits) {
  NdArrayRef aa = operators_ptr_->from(shares);
  aa = operators_ptr_->asAShares(aa);
  aa = operators_ptr_->MSBSigmoid(aa, low_seg, high_seg, skip_bits, nbits);

  std::vector<uint64_t> tmp = operators_ptr_->to(aa);
  std::vector<uint8_t> res(tmp.size());
  std::transform(tmp.begin(), tmp.end(), res.begin(),
                 [&](uint64_t vv) { return vv & 0x1; });

  return res;
}

void AnonXGB::multiplex(std::vector<uint64_t> &gshares,
                        std::vector<uint64_t> &hshares,
                        const std::vector<uint8_t> &masks, int type) {
    SPU_ENFORCE(
            gshares.size() == hshares.size() && gshares.size() == masks.size(),
            "input length should be equal");

    for (const auto mask : masks) {
        SPU_ENFORCE(mask == 1 || mask == 0, "invalid select bits");
    }

    if (type < 0) {
        multiplex_negative(gshares, hshares, masks);
    } else if (type > 0) {
        multiplex_positive(gshares, hshares, masks);
    } else {
        multiplex_zero(gshares, hshares, masks);
    }
}

void AnonXGB::multiplex_negative(std::vector<uint64_t> &gshares,
                                 std::vector<uint64_t> &hshares,
                                 const std::vector<uint8_t> &masks) {
    auto *context = operators_ptr_->getContext();
    auto *comm = context->getState<Communicator>();
    auto *ot_state = context->getState<cheetah::CheetahOTState>();
    ot_state->LazyInit(comm, base_ot_, 0);

    auto sender = ot_state->get(0)->GetSenderCOT();
    auto receiver = ot_state->get(0)->GetReceiverCOT();

    size_t size = gshares.size();
    const auto mask64 = static_cast<uint64_t>(-1);

    std::vector<uint8_t> sel(size);
    std::transform(masks.begin(), masks.end(), sel.begin(),
                   [&](uint8_t vv) { return vv & 0x1; });

    std::vector<uint128_t> rand0(size);
    std::vector<uint128_t> rand1(size);
    std::vector<uint128_t> recv(size);
    std::vector<uint128_t> corr(size);

    if (role_ == psi::PsiRoleType::Sender) {
        for (const auto vv : sel) {
            SPU_ENFORCE(vv == 0, "input mask should be zeros");
        }

        sender->SendRMCC(absl::MakeSpan(rand0), absl::MakeSpan(rand1));
        sender->Flush();

        for (size_t i = 0; i < size; i++) {
            uint64_t high = (gshares[i] * (1 - 2 * sel[i])) - (rand0[i] >> 64U) +
                            (rand1[i] >> 64U);
            uint64_t low = (hshares[i] * (1 - 2 * sel[i])) - (rand0[i] & mask64) +
                           (rand1[i] & mask64);
            corr[i] = (static_cast<uint128_t>(high) << 64U) | low;
        }

        yacl::Buffer buf(corr.data(), corr.size() * sizeof(uint128_t));
        link_ctx_->SendAsyncThrottled(
                link_ctx_->NextRank(), buf,
                fmt::format("send correlated msg of size {}", buf.size()));

        for (size_t i = 0; i < size; i++) {
            gshares[i] = (rand0[i] >> 64U);
            hshares[i] = (rand0[i] & mask64);
        }
    } else {
        receiver->RecvRMCC(absl::MakeSpan(sel), absl::MakeSpan(recv));

        yacl::Buffer buf = link_ctx_->Recv(link_ctx_->NextRank(),
                                           fmt::format("recv correlated msg"));
        SPU_ENFORCE(buf.size() == static_cast<int64_t>(size * sizeof(uint128_t)));

        std::memcpy(reinterpret_cast<uint8_t *>(rand1.data()),
                    reinterpret_cast<uint8_t *>(buf.data()), buf.size());

        for (size_t i = 0; i < size; i++) {
            gshares[i] =
                    (sel[i] * (gshares[i] + (rand1[i] >> 64U))) - (recv[i] >> 64U);
            hshares[i] =
                    (sel[i] * (hshares[i] + (rand1[i] & mask64))) - (recv[i] & mask64);
        }
    }
}

void AnonXGB::multiplex_positive(std::vector<uint64_t> &gshares,
                                 std::vector<uint64_t> &hshares,
                                 const std::vector<uint8_t> &masks) {
    auto *context = operators_ptr_->getContext();
    auto *comm = context->getState<Communicator>();
    auto *ot_state = context->getState<cheetah::CheetahOTState>();
    ot_state->LazyInit(comm, base_ot_, 0);

    auto sender = ot_state->get(0)->GetSenderCOT();
    auto receiver = ot_state->get(0)->GetReceiverCOT();

    size_t size = gshares.size();
    const auto mask64 = static_cast<uint64_t>(-1);

    std::vector<uint8_t> sel(size);
    std::transform(masks.begin(), masks.end(), sel.begin(),
                   [&](uint8_t vv) { return vv & 0x1; });

    std::vector<uint128_t> rand0(size);
    std::vector<uint128_t> rand1(size);
    std::vector<uint128_t> recv(size);
    std::vector<uint128_t> corr(size);

    if (role_ == psi::PsiRoleType::Sender) {
        receiver->RecvRMCC(absl::MakeSpan(sel), absl::MakeSpan(recv));

        yacl::Buffer buf = link_ctx_->Recv(link_ctx_->NextRank(),
                                           fmt::format("recv correlated msg"));
        SPU_ENFORCE(buf.size() == static_cast<int64_t>(size * sizeof(uint128_t)));

        std::memcpy(reinterpret_cast<uint8_t *>(rand1.data()),
                    reinterpret_cast<uint8_t *>(buf.data()), buf.size());

        for (size_t i = 0; i < size; i++) {
            gshares[i] =
                    (sel[i] * (gshares[i] + (rand1[i] >> 64U))) - (recv[i] >> 64U);
            hshares[i] =
                    (sel[i] * (hshares[i] + (rand1[i] & mask64))) - (recv[i] & mask64);
        }
    } else {
        for (const auto vv : sel) {
            SPU_ENFORCE(vv == 0, "input mask should be zeros");
        }

        sender->SendRMCC(absl::MakeSpan(rand0), absl::MakeSpan(rand1));
        sender->Flush();

        for (size_t i = 0; i < size; i++) {
            uint64_t high = (gshares[i] * (1 - 2 * sel[i])) - (rand0[i] >> 64U) +
                            (rand1[i] >> 64U);
            uint64_t low = (hshares[i] * (1 - 2 * sel[i])) - (rand0[i] & mask64) +
                           (rand1[i] & mask64);
            corr[i] = (static_cast<uint128_t>(high) << 64U) | low;
        }

        yacl::Buffer buf(corr.data(), corr.size() * sizeof(uint128_t));
        link_ctx_->SendAsyncThrottled(
                link_ctx_->NextRank(), buf,
                fmt::format("send correlated msg of size {}", buf.size()));

        for (size_t i = 0; i < size; i++) {
            gshares[i] = (rand0[i] >> 64U);
            hshares[i] = (rand0[i] & mask64);
        }
    }
}

void AnonXGB::multiplex_zero(std::vector<uint64_t> &gshares,
                             std::vector<uint64_t> &hshares,
                             const std::vector<uint8_t> &masks) {
    auto *context = operators_ptr_->getContext();
    auto *comm = context->getState<Communicator>();
    auto *ot_state = context->getState<cheetah::CheetahOTState>();
    ot_state->LazyInit(comm, base_ot_, 0);

    auto sender = ot_state->get(0)->GetSenderCOT();
    auto receiver = ot_state->get(0)->GetReceiverCOT();

    size_t size = gshares.size();
    const auto mask64 = static_cast<uint64_t>(-1);

    std::vector<uint8_t> sel(size);
    std::transform(masks.begin(), masks.end(), sel.begin(),
                   [&](uint8_t vv) { return vv & 0x1; });

    std::vector<uint128_t> rand0(size);
    std::vector<uint128_t> rand1(size);
    std::vector<uint128_t> recv(size);
    std::vector<uint128_t> corr(size);

    if (role_ == psi::PsiRoleType::Sender) {
        sender->SendRMCC(absl::MakeSpan(rand0), absl::MakeSpan(rand1));
        sender->Flush();

        receiver->RecvRMCC(absl::MakeSpan(sel), absl::MakeSpan(recv));
    } else {
        receiver->RecvRMCC(absl::MakeSpan(sel), absl::MakeSpan(recv));

        sender->SendRMCC(absl::MakeSpan(rand0), absl::MakeSpan(rand1));
        sender->Flush();
    }

    for (size_t i = 0; i < size; i++) {
        uint64_t high = (gshares[i] * (1 - 2 * sel[i])) - (rand0[i] >> 64U) +
                        (rand1[i] >> 64U);
        uint64_t low = (hshares[i] * (1 - 2 * sel[i])) - (rand0[i] & mask64) +
                       (rand1[i] & mask64);
        corr[i] = (static_cast<uint128_t>(high) << 64U) | low;
    }

    if (role_ == psi::PsiRoleType::Sender) {
        yacl::Buffer buf(corr.data(), corr.size() * sizeof(uint128_t));
        link_ctx_->SendAsyncThrottled(
                link_ctx_->NextRank(), buf,
                fmt::format("send correlated msg of size {}", buf.size()));

        buf = link_ctx_->Recv(link_ctx_->NextRank(),
                              fmt::format("recv correlated msg"));
        SPU_ENFORCE(buf.size() == static_cast<int64_t>(size * sizeof(uint128_t)));

        std::memcpy(reinterpret_cast<uint8_t *>(rand1.data()),
                    reinterpret_cast<uint8_t *>(buf.data()), buf.size());
    } else {
        yacl::Buffer buf = link_ctx_->Recv(link_ctx_->NextRank(),
                                           fmt::format("recv correlated msg"));
        SPU_ENFORCE(buf.size() == static_cast<int64_t>(size * sizeof(uint128_t)));

        std::memcpy(reinterpret_cast<uint8_t *>(rand1.data()),
                    reinterpret_cast<uint8_t *>(buf.data()), buf.size());

        std::memcpy(reinterpret_cast<uint8_t *>(buf.data()),
                    reinterpret_cast<uint8_t *>(corr.data()), buf.size());
        link_ctx_->SendAsyncThrottled(
                link_ctx_->NextRank(), buf,
                fmt::format("send correlated msg of size {}", buf.size()));
    }

    for (size_t i = 0; i < size; i++) {
        uint64_t high0 = (gshares[i] * sel[i]) + (rand0[i] >> 64U);
        uint64_t low0 = (hshares[i] * sel[i]) + (rand0[i] & mask64);

        uint64_t high1 = (sel[i] * (rand1[i] >> 64U)) - (recv[i] >> 64U);
        uint64_t low1 = (sel[i] * (rand1[i] & mask64)) - (recv[i] & mask64);

        gshares[i] = high0 + high1;
        hshares[i] = low0 + low1;
    }
}

std::vector<uint64_t> AnonXGB::multiplex(const std::vector<uint8_t> &masks,
                                         const std::vector<uint64_t> &shares0) {
  std::vector<uint64_t> res(masks.size());
  std::transform(masks.begin(), masks.end(), res.begin(),
                 [&](uint8_t vv) { return vv & 0x1; });

  NdArrayRef bb = operators_ptr_->from(res);
  NdArrayRef aa = operators_ptr_->from(shares0);

  bb = operators_ptr_->asBShares(bb, 1);
  aa = operators_ptr_->asAShares(aa);

  aa = operators_ptr_->MultiplexA(bb, aa);

  return operators_ptr_->to(aa);
}

std::vector<uint64_t> AnonXGB::mux(const std::vector<uint8_t> &masks,
                                   const std::vector<uint64_t> &shares0,
                                   const std::vector<uint64_t> &shares1) {
  std::vector<uint64_t> res(masks.size());
  std::transform(masks.begin(), masks.end(), res.begin(),
                 [&](uint8_t vv) { return vv & 0x1; });

  NdArrayRef bb = operators_ptr_->from(res);
  NdArrayRef a0 = operators_ptr_->from(shares0);
  NdArrayRef a1 = operators_ptr_->from(shares1);

  bb = operators_ptr_->asBShares(bb, 1);
  a0 = operators_ptr_->asAShares(a0);
  a1 = operators_ptr_->asAShares(a1);

  a0 = operators_ptr_->MuxA(bb, a0, a1);
  res = operators_ptr_->to(a0);

  return res;
}

std::vector<uint8_t> AnonXGB::bits_xor(const std::vector<uint8_t> &shares0,
                                       const std::vector<uint8_t> &shares1) {
  SPU_ENFORCE(shares0.size() == shares1.size());
  std::vector<uint8_t> res(shares0.size());
  for (size_t i = 0; i < res.size(); i++) {
    res[i] = shares0[i] ^ shares1[i];
  }
  return res;
}

std::vector<uint64_t> AnonXGB::add(const std::vector<uint64_t> &shares0,
                                   const std::vector<uint64_t> &shares1) {
  SPU_ENFORCE(shares0.size() == shares1.size());
  std::vector<uint64_t> res(shares0.size());
  for (size_t i = 0; i < shares0.size(); i++) {
    res[i] = shares0[i] + shares1[i];
  }
  return res;
}

std::vector<uint64_t> AnonXGB::sub(const std::vector<uint64_t> &shares0,
                                   const std::vector<uint64_t> &shares1) {
  SPU_ENFORCE(shares0.size() == shares1.size());
  std::vector<uint64_t> res(shares0.size());
  for (size_t i = 0; i < shares0.size(); i++) {
    res[i] = shares0[i] - shares1[i];
  }
  return res;
}

void AnonXGB::add_plain(std::vector<uint64_t> &shares,
                        const std::vector<double> &plains, size_t log2_scale) {
  SPU_ENFORCE(shares.size() == plains.size());

  if (operators_ptr_->getRole() == psi::PsiRoleType::Sender) {
    for (size_t i = 0; i < shares.size(); i++) {
      shares[i] = shares[i] + encode(plains[i], log2_scale);
    }
  }
}

void AnonXGB::add_plain(std::vector<uint64_t> &shares, double plain,
                        double scale) {
  if (operators_ptr_->getRole() == psi::PsiRoleType::Sender) {
    auto vv = static_cast<int64_t>(std::round(plain * scale));
    uint64_t value = vv;
    for (auto &share : shares) {
      share = share + value;
    }
  }
}

void AnonXGB::mul_plain(std::vector<uint64_t> &shares, double factor,
                        size_t log2_scale, SignType sign) {
  auto vv = static_cast<int64_t>(std::round(factor * std::pow(2, log2_scale)));
  uint64_t value = vv;
  for (auto &share : shares) {
    share = share * value;
  }
  truncate(shares, log2_scale, sign);
}

void AnonXGB::negate(std::vector<uint64_t> &shares) {
  std::transform(shares.begin(), shares.end(), shares.begin(),
                 [&](uint64_t vv) { return -vv; });
}

std::vector<uint64_t> AnonXGB::mul(const std::vector<uint64_t> &shares0,
                                   const std::vector<uint64_t> &shares1) {
  NdArrayRef aa = operators_ptr_->from(shares0);
  NdArrayRef bb = operators_ptr_->from(shares1);
  aa = operators_ptr_->MulAA(aa, bb);

  return operators_ptr_->to(aa);
}

void AnonXGB::mul(std::vector<uint64_t> &inputs) {
  auto len = static_cast<int64_t>(inputs.size());
  FieldType field = FieldType::FM64;

  NdArrayRef inp = ring_rand(field, {len});

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto xinp = NdArrayView<ring2k_t>(inp);
    pforeach(0, xinp.numel(), [&](int64_t i) { xinp[i] = inputs[i]; });
  });

  auto mul_prot =
      operators_ptr_->getContext()->getState<CheetahMulState>()->get();

  NdArrayRef oup;
  if (role_ == psi::PsiRoleType::Sender) {
    oup = mul_prot->MulOLE(inp, true);
  } else {
    oup = mul_prot->MulOLE(inp, false);
  }

  DISPATCH_ALL_FIELDS(field, "", [&]() {
    auto xoup = NdArrayView<ring2k_t>(oup);
    pforeach(0, xoup.numel(), [&](int64_t i) { inputs[i] = xoup[i]; });
  });
}

std::vector<uint64_t> AnonXGB::square(const std::vector<uint64_t> &shares) {
  NdArrayRef aa = operators_ptr_->from(shares);
  aa = operators_ptr_->SquareA(aa);
  return operators_ptr_->to(aa);
}

void AnonXGB::truncate(std::vector<uint64_t> &ashr, size_t log2_scale,
                       SignType sign) {
  NdArrayRef aa = operators_ptr_->from(ashr);
  aa = operators_ptr_->asAShares(aa);
  aa = operators_ptr_->TruncA(aa, log2_scale, sign);
  ashr = operators_ptr_->to(aa);
}

std::vector<uint8_t> AnonXGB::is_left(const std::vector<uint64_t> &idx_ashr) {
  std::vector<uint64_t> temp(idx_ashr.begin(), idx_ashr.end());
  if (role_ == psi::PsiRoleType::Sender) {
    std::transform(temp.begin(), temp.end(), temp.begin(),
                   [&](uint64_t vv) { return vv - self_cols_; });
  }

  return msb(temp);
}

std::vector<uint64_t> AnonXGB::recoveryA(const std::vector<uint64_t> &ashr) {
  yacl::Buffer ashr_buf(ashr.size() * sizeof(uint64_t));
  for (size_t i = 0; i < ashr.size(); i++) {
    std::memcpy(
        reinterpret_cast<uint8_t *>(ashr_buf.data()) + i * sizeof(uint64_t),
        &(ashr[i]), sizeof(uint64_t));
  }

  link_ctx_->SendAsyncThrottled(
      link_ctx_->NextRank(), ashr_buf,
      fmt::format("AnonXGB: Recovery AShares of size {}", ashr_buf.size()));

  yacl::Buffer recv_buf = link_ctx_->Recv(
      link_ctx_->NextRank(), fmt::format("AnonXGB:Recovery AShares"));
  SPU_ENFORCE(static_cast<uint64_t>(recv_buf.size()) ==
              sizeof(uint64_t) * ashr.size());

  std::vector<uint64_t> recv;
  for (int64_t i = 0; i < recv_buf.size(); i += sizeof(uint64_t)) {
    uint64_t vv = 0;
    std::memcpy(&vv, reinterpret_cast<uint8_t *>(recv_buf.data()) + i,
                sizeof(uint64_t));
    recv.push_back(vv);
  }

  for (size_t i = 0; i < recv.size(); i++) {
    recv[i] = recv[i] + ashr[i];
  }
  return recv;
}

template <typename T>
std::vector<T> AnonXGB::recoveryB(const std::vector<T> &bshr) {
  yacl::Buffer bshr_buf(bshr.size() * sizeof(T));
  for (size_t i = 0; i < bshr.size(); i++) {
    std::memcpy(reinterpret_cast<uint8_t *>(bshr_buf.data()) + i * sizeof(T),
                &(bshr[i]), sizeof(T));
  }

  link_ctx_->SendAsyncThrottled(
      link_ctx_->NextRank(), bshr_buf,
      fmt::format("AnonXGB: Recovery BShares of size {}", bshr_buf.size()));

  yacl::Buffer recv_buf = link_ctx_->Recv(
      link_ctx_->NextRank(), fmt::format("AnonXGB:Recovery BShares"));
  SPU_ENFORCE(static_cast<uint64_t>(recv_buf.size()) ==
              sizeof(T) * bshr.size());

  std::vector<T> recv;
  for (int64_t i = 0; i < recv_buf.size(); i += sizeof(T)) {
    uint64_t vv = 0;
    std::memcpy(&vv, reinterpret_cast<uint8_t *>(recv_buf.data()) + i,
                sizeof(T));
    recv.push_back(vv);
  }

  for (size_t i = 0; i < recv.size(); i++) {
    recv[i] = recv[i] ^ bshr[i];
  }
  return recv;
}

std::vector<uint64_t> AnonXGB::sigmoid(const std::vector<uint64_t> &shares,
                                       uint8_t type) {
  return type == 2 ? sigmoid2(shares) : sigmoid1(shares);
}

std::vector<uint64_t> AnonXGB::sigmoid1(const std::vector<uint64_t> &shares) {
  //        link_ctx_->PrintStats();
  std::vector<double> coeffs = {
      1.642327, -1.070336, 0.5510985,
  };

  size_t len = shares.size();
  uint64_t bound = encode(2.8, log2_scale_);
  uint64_t half = encode(0.25, log2_scale_ + log2_scale_);
  uint64_t one = half << 1U;

  std::vector<uint64_t> values(len * 2);
  std::transform(shares.begin(), shares.end(), values.begin(),
                 [&](uint64_t vv) { return vv + bound; });
  std::transform(shares.begin(), shares.end(), values.begin() + len,
                 [&](uint64_t vv) { return vv - bound; });

  auto before_neg_sgns = link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;

  //        auto time_start = std::chrono::high_resolution_clock::now();
  auto neg_sgns = msb(values);
  //        auto time_end = std::chrono::high_resolution_clock::now();
  //        auto time_diff =
  //        std::chrono::duration_cast<std::chrono::microseconds>(time_end -
  //        time_start); std::cout << "msb takes " << time_diff.count() << " us"
  //        << std::endl;

  auto after_neg_sgns = link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;
  std::cout << "msb comm = " << (after_neg_sgns - before_neg_sgns) << std::endl;

  std::vector<double> sin_arr(len * coeffs.size());
  std::vector<double> cos_arr(len * coeffs.size());
  double pow = std::pow(2, log2_scale_ + 4);

  pforeach(0, len, [&](int64_t i) {
    double value = shares[i] / pow * M_PI;
    sin_arr[i] = std::sin(value);
    cos_arr[i] = std::cos(value);
  });

  for (size_t i = 1; i < coeffs.size(); i++) {
    size_t offset = i * len;
    pforeach(0, len, [&](int64_t j) {
      size_t posj = offset - len + j;
      sin_arr[offset + j] =
          sin_arr[j] * cos_arr[posj] + cos_arr[j] * sin_arr[posj];
      cos_arr[offset + j] =
          cos_arr[j] * cos_arr[posj] - sin_arr[j] * sin_arr[posj];
    });
  }

  for (size_t i = 0; i < coeffs.size(); i++) {
    pforeach(0, len, [&](int64_t j) { sin_arr[i * len + j] *= coeffs[i]; });
  }

  values.resize(sin_arr.size() + cos_arr.size());
  if (role_ == psi::PsiRoleType::Sender) {
    std::transform(sin_arr.begin(), sin_arr.end(), values.begin(),
                   [&](double vv) { return encode(vv, log2_scale_); });
    std::transform(cos_arr.begin(), cos_arr.end(),
                   values.begin() + sin_arr.size(),
                   [&](double vv) { return encode(vv, log2_scale_); });
  } else {
    std::transform(cos_arr.begin(), cos_arr.end(), values.begin(),
                   [&](double vv) { return encode(vv, log2_scale_); });
    std::transform(sin_arr.begin(), sin_arr.end(),
                   values.begin() + cos_arr.size(),
                   [&](double vv) { return encode(vv, log2_scale_); });
  }

  auto time_start = std::chrono::high_resolution_clock::now();
  mul(values);
  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
      time_end - time_start);

  auto after_mul = link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;
  std::cout << "mul takes " << time_diff.count() << " us, comm = " << (after_mul - after_neg_sgns) << std::endl;

  for (size_t i = 0; i < sin_arr.size(); i++) {
    values[i] = values[i] + values[i + sin_arr.size()];
  }
  values.resize(sin_arr.size());

  std::vector<uint64_t> res(shares.size(), half);
  for (size_t i = 0; i < coeffs.size(); i++) {
    for (size_t j = 0; j < len; j++) {
      res[j] += values[i * len + j];
    }
  }

  std::vector<uint64_t> temp(len + len);
  std::transform(res.begin(), res.end(), temp.begin(),
                 [&](uint64_t vv) { return -vv; });
  std::transform(res.begin(), res.end(), temp.begin() + len,
                 [&](uint64_t vv) { return vv - one; });

  temp = multiplex(neg_sgns, temp);

  for (size_t i = 0; i < len; i++) {
    res[i] = temp[i] + temp[i + len] + one;
  }

  truncate(res, log2_scale_ + 1, SignType::Positive);

  //        link_ctx_->PrintStats();

  return res;
}

std::vector<uint64_t> AnonXGB::sigmoid2(const std::vector<uint64_t> &shares) {
  // std::cout << "begin sigmoid: " << std::endl;
  // link_ctx_->PrintStats();

  std::vector<double> coeffs = {
      // 2.911534156270083,   -3.5932930569416737, 3.618659659303885,
      // -2.4606874754063313, 1.3075041795616118,  -0.45319919724799307,
      // 0.10037762183081642,
      5.351809e-01, -3.353619e-02, 5.362122e-02 // 4 term
      // 5.404022e-01, -4.181079e-02, 6.178210e-02, -6.451866e-03 // 5 term

      // 0.6172949043536653, -0.0341990021261339, 0.1693788502244572, -0.0460333847898619,
      // 0.0816712796122188, -0.0433475059227459, 0.0507073237098216, -0.0369643373243371 // squirrel 9 term

  };

  size_t len = shares.size();
  // uint64_t high_seg = encode(4.0, log2_scale_);
  // uint64_t low_seg = encode(-4.0, log2_scale_);

  // std::vector<uint64_t> values(len * 2);
  // std::transform(shares.begin(), shares.end(), values.begin(),
  //                [&](uint64_t vv) { return vv + high_seg; });
  // std::transform(shares.begin(), shares.end(), values.begin() + len,
  //                [&](uint64_t vv) { return vv + low_seg; });

  //        auto time_start = std::chrono::high_resolution_clock::now();
  // auto neg_sgns = msbsigmoid(shares, low_seg, high_seg, log2_scale_);
  //        auto time_end = std::chrono::high_resolution_clock::now();
  //        auto time_diff =
  //        std::chrono::duration_cast<std::chrono::microseconds>(time_end -
  //        time_start); std::cout << "msb takes " << time_diff.count() << " us"
  //        << std::endl;

  uint64_t high_seg = encode(5.6, log2_scale_);
  uint64_t low_seg = encode(-5.6, log2_scale_);

  auto neg_sgns = msbsigmoid(shares, low_seg, high_seg, log2_scale_-2);

  // std::cout << "after two msb in sigmoid: " << std::endl;
  // link_ctx_->PrintStats();

  std::vector<double> sin_arr(len * coeffs.size());
  std::vector<double> cos_arr(len * coeffs.size());
  double pow = std::pow(2, log2_scale_ + 3);

  pforeach(0, len, [&](int64_t i) {
    double value = shares[i] / pow * M_PI;
    sin_arr[i] = std::sin(value);
    cos_arr[i] = std::cos(value);
  });

  for (size_t i = 1; i < coeffs.size(); i++) {
    size_t offset = i * len;
    pforeach(0, len, [&](int64_t j) {
      size_t posj = offset - len + j;
      sin_arr[offset + j] =
          sin_arr[j] * cos_arr[posj] + cos_arr[j] * sin_arr[posj];
      cos_arr[offset + j] =
          cos_arr[j] * cos_arr[posj] - sin_arr[j] * sin_arr[posj];
    });
  }

  const double scale_factor = bfv_plain_mod_ / std::pow(2, 51);
  const uint64_t offset = (64 - 51) + 46 - log2_scale_;

  for (size_t i = 0; i < coeffs.size(); i++) {
    double vv = coeffs[i] * scale_factor;
    pforeach(0, len, [&](int64_t j) { sin_arr[i * len + j] *= vv; });
  }

  std::vector<double> reals(sin_arr.size() + cos_arr.size());
  if (role_ == psi::PsiRoleType::Sender) {
    std::copy_n(sin_arr.data(), sin_arr.size(), reals.data());
    std::copy_n(cos_arr.data(), cos_arr.size(), reals.data() + sin_arr.size());
  } else {
    std::copy_n(cos_arr.data(), cos_arr.size(), reals.data());
    std::copy_n(sin_arr.data(), sin_arr.size(), reals.data() + cos_arr.size());
  }

  // auto bgn_comm = link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;
  // auto time_start = std::chrono::high_resolution_clock::now();
  vector<uint64_t> values = mul(reals);
  // auto time_end = std::chrono::high_resolution_clock::now();
  // auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
      // time_end - time_start);
  // std::cout << "mul takes " << time_diff.count() << " us, comm bytes = " << 
  // (link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes - bgn_comm) << std::endl;

  // std::cout << "after mul in sigmoid: " << std::endl;
  // link_ctx_->PrintStats();

  for (size_t i = 0; i < sin_arr.size(); i++) {
    values[i] = (values[i] + values[i + sin_arr.size()]) % bfv_plain_mod_;
  }
  for (size_t i = 1; i < coeffs.size(); i++) {
    for (size_t j = 0; j < len; j++) {
      values[j] = (values[j] + values[i * len + j]) % bfv_plain_mod_;
    }
  }
  
  std::vector<uint64_t> res(values.begin(), values.begin() + len);

  auto half = encode(0.25, log2_scale_ + offset);
  auto one = half << 1U;

  double factor = std::pow(2, 64) / bfv_plain_mod_;
  for (auto &rr : res) {
    uint64_t vv = std::round(factor * rr);
    rr = vv + half;
  }

  std::vector<uint64_t> temp(len + len);
  std::transform(res.begin(), res.end(), temp.begin(),
                 [&](uint64_t vv) { return -vv; });
  std::transform(res.begin(), res.end(), temp.begin() + len,
                 [&](uint64_t vv) { return vv - one; });

  temp = multiplex(neg_sgns, temp);

  for (size_t i = 0; i < len; i++) {
    res[i] = temp[i] + temp[i + len] + one;
  }

  truncate(res, offset + 1, SignType::Positive);

  std::cout << "sigmoid end " << std::endl;
  // link_ctx_->PrintStats();

  return res;
}

// 要求values的取值范围为[-1, 1], output scale 2^46
std::vector<uint64_t> AnonXGB::mul(const std::vector<double> &inputs) {
  auto serialize =
      [&](const std::vector<seal::Ciphertext> &enc_vec) -> yacl::Buffer {
    std::vector<std::vector<uint8_t>> bytes(enc_vec.size());
    for (size_t i = 0; i < enc_vec.size(); i++) {
      SPU_ENFORCE(enc_vec[i].size() != 0, "empty ciphertext");
      bytes[i].resize(enc_vec[i].save_size());
      enc_vec[i].save(reinterpret_cast<byte *>(bytes[i].data()),
                      bytes[i].size());
    }

    size_t length = 0;
    for (const auto &v_bytes : bytes) {
      length += v_bytes.size();
    }

    yacl::Buffer buf(length);
    size_t pos = 0;
    for (const auto &v_bytes : bytes) {
      std::memcpy(reinterpret_cast<uint8_t *>(buf.data()) + pos, v_bytes.data(),
                  v_bytes.size());
      pos += v_bytes.size();
    }
    SPU_ENFORCE(pos == length);

    return buf;
  };

  auto deserialze = [&](yacl::Buffer &buf,
                        size_t num) -> std::vector<seal::Ciphertext> {
    SPU_ENFORCE(buf.size() % num == 0);
    size_t each_size = buf.size() / num;

    vector<vector<uint8_t>> bytes(num);
    for (auto &v_bytes : bytes) {
      v_bytes.resize(each_size);
    }

    std::vector<seal::Ciphertext> enc_vec(num);
    for (size_t i = 0; i < num; i++) {
      std::memcpy(bytes[i].data(),
                  reinterpret_cast<uint8_t *>(buf.data()) + i * each_size,
                  bytes[i].size());
      enc_vec[i].load(*bfv_context_,
                      reinterpret_cast<byte *>((uint8_t *)bytes[i].data()),
                      bytes[i].size());
    }

    return enc_vec;
  };

  double scale = std::pow(2, 23);

  std::vector<uint64_t> values(inputs.size());
  std::transform(inputs.begin(), inputs.end(), values.begin(), [&](double vv) {
    return (static_cast<int64_t>(std::round(vv * scale)) + bfv_plain_mod_) %
           bfv_plain_mod_;
  });

  size_t num = (values.size() + bfv_coeff_count_ - 1) / bfv_coeff_count_;

  std::vector<seal::Plaintext> plain_vec(num);
  for (size_t i = 0; i < num; i++) {
    std::vector<uint64_t> sub_vec(bfv_coeff_count_, 0);
    size_t sub_size = std::min(bfv_coeff_count_, values.size() - i * bfv_coeff_count_);
    std::copy_n(values.data() + i * bfv_coeff_count_, sub_size, sub_vec.data());
    bfv_encoder_->encode(sub_vec, plain_vec[i]);
  }

  if (role_ == psi::PsiRoleType::Sender) {
    std::vector<seal::Ciphertext> enc_vec(num);
    for (size_t i = 0; i < num; i++) {
      // bfv_encryptor_->encrypt_symmetric(plain_vec[i], enc_vec[i]);
      const auto &context_data = bfv_context_->first_context_data();
      seal::util::encrypt_zero_symmetric(secret_key_, *bfv_context_, context_data->parms_id(), false, true, enc_vec[i]);
      seal::util::multiply_add_plain_with_scaling_variant(plain_vec[i], *context_data, *seal::util::iter(enc_vec[i]));
    }

    yacl::Buffer buf = serialize(enc_vec);

    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf,
        fmt::format("send bfv ciphertexts of size {}", buf.size()));

    buf = link_ctx_->Recv(
        link_ctx_->NextRank(),
        fmt::format("recv bfv ciphertexts after homormorhpic operations"));

    enc_vec = deserialze(buf, num);

    for (size_t i = 0; i < num; i++) {
      //              std::cout << "send decrypt budget = " <<
      //              decryptor.invariant_noise_budget(enc_vec0[i /
      //              bfv_coeff_count_]) << std::endl;

      std::vector<uint64_t> sub_vec(bfv_coeff_count_);
      bfv_decryptor_->decrypt(enc_vec[i], plain_vec[i]);
      bfv_encoder_->decode(plain_vec[i], sub_vec);

      size_t sub_size =
          std::min(bfv_coeff_count_, values.size() - i * bfv_coeff_count_);
      std::copy_n(sub_vec.data(), sub_size,
                  values.data() + i * bfv_coeff_count_);
    }

  } else {
    auto bytes =
        yacl::crypto::SecureRandBytes(values.size() * sizeof(uint64_t));
    std::memcpy(reinterpret_cast<uint8_t *>(values.data()), bytes.data(),
                bytes.size());
    std::transform(values.begin(), values.end(), values.begin(),
                   [&](uint64_t vv) { return vv % bfv_plain_mod_; });

    std::vector<seal::Plaintext> rnd_plain_vec(num);
    for (size_t i = 0; i < num; i++) {
      size_t sub_size =
          std::min(bfv_coeff_count_, values.size() - i * bfv_coeff_count_);
      std::vector<uint64_t> sub_vec(bfv_coeff_count_);
      std::copy_n(values.data() + i * bfv_coeff_count_, sub_size,
                  sub_vec.data());
      bfv_encoder_->encode(sub_vec, rnd_plain_vec[i]);
    }

    yacl::Buffer buf = link_ctx_->Recv(link_ctx_->NextRank(),
                                       fmt::format("recv bfv ciphertexts"));

    std::vector<seal::Ciphertext> enc_vec = deserialze(buf, num);
    for (size_t i = 0; i < num; i++) {
      bfv_evaluator_->multiply_plain_inplace(enc_vec[i], plain_vec[i]);
      bfv_evaluator_->mod_switch_to_next_inplace(enc_vec[i]);
      bfv_evaluator_->sub_plain_inplace(enc_vec[i], rnd_plain_vec[i]);
    }

    buf = serialize(enc_vec);
    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf,
        fmt::format(
            "after homormorphic operations, send bfv ciphertexts of size {}",
            buf.size()));
  }

  return values;
}

//    std::vector<uint64_t> AnonXGB::mul(const std::vector<double> &inputs) {
//        auto serialize = [&](const std::vector<seal::Ciphertext> &enc_vec) ->
//        yacl::Buffer {
//
//            std::vector<std::vector<uint8_t>> bytes(enc_vec.size());
//            for (size_t i = 0; i < enc_vec.size(); i++) {
//                SPU_ENFORCE(enc_vec[i].size() != 0, "empty ciphertext");
//                bytes[i].resize(enc_vec[i].save_size());
//                enc_vec[i].save(reinterpret_cast<byte *>(bytes[i].data()),
//                bytes[i].size());
//            }
//
//            size_t length = 0;
//            for (const auto &v_bytes : bytes) {
//                length += v_bytes.size();
//            }
//
//            yacl::Buffer buf(length);
//            size_t pos = 0;
//            for (const auto &v_bytes : bytes) {
//                std::memcpy(reinterpret_cast<uint8_t *>(buf.data()) + pos,
//                            v_bytes.data(), v_bytes.size());
//                pos += v_bytes.size();
//            }
//            SPU_ENFORCE(pos == length);
//
//            return buf;
//        };
//
//        auto deserialze = [&](yacl::Buffer &buf, size_t num) ->
//        std::vector<seal::Ciphertext> {
//            SPU_ENFORCE(buf.size() % num == 0);
//            size_t each_size = buf.size() / num;
//
//            vector<vector<uint8_t>> bytes(num);
//            for (auto & v_bytes : bytes) {
//                v_bytes.resize(each_size);
//            }
//
//            std::vector<seal::Ciphertext> enc_vec(num);
//            for (size_t i = 0; i < num; i++) {
//                std::memcpy(bytes[i].data(), reinterpret_cast<uint8_t
//                *>(buf.data()) + i * each_size, bytes[i].size());
//                enc_vec[i].load(*bfv_context_, reinterpret_cast<byte
//                *>((uint8_t *)bytes[i].data()), bytes[i].size());
//            }
//
//            return enc_vec;
//        };
//
//        double scale = std::pow(2, 23);
//
//        std::vector<uint64_t> values(inputs.size());
//        std::transform(inputs.begin(), inputs.end(), values.begin(),
//        [&](double vv) {
//            return (static_cast<int64_t>(std::round(vv * scale)) +
//            bfv_plain_mod_) % bfv_plain_mod_;
//        });
//
//        size_t num = (values.size() + bfv_coeff_count_ - 1) /
//        bfv_coeff_count_; size_t sub_size = 0; size_t each_step = 64;
//
//        seal::Plaintext plain;
//        std::vector<uint64_t> sub_vec(bfv_coeff_count_, 0);
//
//        if (role_ == psi::PsiRoleType::Sender) {
//
//            for (size_t i = 0; i < num; i += each_step) {
//                size_t each_len = std::min(each_step, num - i);
//
//                std::vector<seal::Ciphertext> enc_vec(each_len);
//                for (size_t j = i; j < (i + each_len); j++) {
//                    sub_size = std::min(bfv_coeff_count_, values.size() - j *
//                    bfv_coeff_count_); std::copy_n(values.data() + j *
//                    bfv_coeff_count_, sub_size, sub_vec.data());
//
//                    bfv_encoder_->encode(sub_vec, plain);
//                    bfv_encryptor_->encrypt_symmetric(plain, enc_vec[j - i]);
//                }
//
//                yacl::Buffer buf = serialize(enc_vec);
//
//                link_ctx_->SendAsyncThrottled(
//                        link_ctx_->NextRank(), buf,
//                        fmt::format("send bfv ciphertexts of size {}",
//                        buf.size()));
//
//                buf = link_ctx_->Recv(
//                        link_ctx_->NextRank(), fmt::format("recv bfv
//                        ciphertexts after homormorhpic operations"));
//
//                enc_vec = deserialze(buf, enc_vec.size());
//
//                printf("send enc_vec size = %zu, coeff_modulus_size = %zu \n",
//                enc_vec.size(), enc_vec[0].coeff_modulus_size());
//
//                for (size_t j = i; j < (i + each_len); j++) {
////                        std::cout << "send decrypt budget = " <<
///decryptor.invariant_noise_budget(enc_vec0[i / bfv_coeff_count_]) <<
///std::endl;
//
//                    bfv_decryptor_->decrypt(enc_vec[j - i], plain);
//                    bfv_encoder_->decode(plain, sub_vec);
//
//                    sub_size = std::min(bfv_coeff_count_, values.size() - j *
//                    bfv_coeff_count_); std::copy_n(sub_vec.data(), sub_size,
//                    values.data() + j * bfv_coeff_count_);
//                }
//            }
//
//        } else {
//            for (size_t i = 0; i < num; i += each_step) {
//                yacl::Buffer buf = link_ctx_->Recv(
//                        link_ctx_->NextRank(), fmt::format("recv bfv
//                        ciphertexts"));
//
//                size_t each_len = std::min(each_step, num - i);
//                std::vector<seal::Ciphertext> enc_vec = deserialze(buf,
//                each_len); for (size_t j = i; j < (i + each_len); j++) {
//                    sub_size = std::min(bfv_coeff_count_, values.size() - j *
//                    bfv_coeff_count_); std::copy_n(values.data() + j *
//                    bfv_coeff_count_, sub_size, sub_vec.data());
//
//                    bfv_encoder_->encode(sub_vec, plain);
//                    bfv_evaluator_->multiply_plain_inplace(enc_vec[j - i],
//                    plain);
//
//                    for (auto &vv : sub_vec) {
//                        vv = seal::random_uint64() % bfv_plain_mod_;
//                    }
//                    std::copy_n(sub_vec.data(), sub_size, values.data() + j *
//                    bfv_coeff_count_);
//
//                    bfv_encoder_->encode(sub_vec, plain);
//                    bfv_evaluator_->sub_plain_inplace(enc_vec[j - i], plain);
//                    bfv_evaluator_->mod_switch_to_next_inplace(enc_vec[j -
//                    i]);
//                }
//
//                buf = serialize(enc_vec);
//                link_ctx_->SendAsyncThrottled(
//                        link_ctx_->NextRank(), buf,
//                        fmt::format("after homormorphic operations, send bfv
//                        ciphertexts of size {}", buf.size()));
//            }
//        }
//
//        return values;
//    }

//    std::vector<uint64_t> AnonXGB::square_and_divide(const
//    std::vector<uint64_t> &a_shr,
//                                                     const
//                                                     std::vector<uint64_t>
//                                                     &babs_shr) {
//        SPU_ENFORCE(a_shr.size() == babs_shr.size());
//
//        auto aa = recoveryA(a_shr);
//        auto bb = recoveryA(babs_shr);
//
//        std::vector<uint64_t> cc(a_shr.size());
//        for (size_t i = 0; i < cc.size(); i++) {
//            double avalue = decode(aa[i], log2_scale_);
//            double bvalue = decode(bb[i], log2_scale_);
//            double value = avalue * avalue / bvalue;
//            cc[i] = encode(value / 2.0, log2_scale_);
//        }
//
//        return cc;
//    }

// std::vector<uint64_t> AnonXGB::square_and_divide(const std::vector<uint64_t>
// &a_shr,
//                                                  const std::vector<uint64_t>
//                                                  &babs_shr) {
//     SPU_ENFORCE(a_shr.size() == babs_shr.size());

//     NdArrayRef aa = operators_ptr_->from(a_shr);
//     NdArrayRef bb_abs = operators_ptr_->from(babs_shr);

//     aa = operators_ptr_->asAShares(aa);
//     bb_abs = operators_ptr_->asAShares(bb_abs);

//     size_t fxp_goldschmidt_iters = 30;

//     auto bb_msb = operators_ptr_->HighestOneBit(bb_abs);

//     auto factor_bb = operators_ptr_->BitRevB(bb_msb, 0, 2 * log2_scale_);
//     auto cc = operators_ptr_->MulAB(bb_abs, factor_bb, log2_scale_);

//     auto k2_9142 = operators_ptr_->encodeD(2.9142F, cc.shape(), log2_scale_);
//     auto rr = operators_ptr_->SubPA(k2_9142, operators_ptr_->AddAA(cc, cc));

//     auto ee = operators_ptr_->MulAA(rr, cc);
//     ee = operators_ptr_->TruncA(ee, log2_scale_, SignType::Positive);
//     auto k1 = operators_ptr_->encodeD(1.0F, ee.shape(), log2_scale_);
//     ee = operators_ptr_->SubPA(k1, ee);

//     for (size_t itr = 0; itr < fxp_goldschmidt_iters; itr++) {
//         rr = operators_ptr_->MulAA(rr, operators_ptr_->AddAP(ee, k1));
//         rr = operators_ptr_->TruncA(rr, log2_scale_, SignType::Positive);

//         if (itr + 1 < fxp_goldschmidt_iters) {
//             ee = operators_ptr_->SquareA(ee);
//             ee = operators_ptr_->TruncA(ee, log2_scale_, SignType::Positive);
//         }
//     }

//     NdArrayRef aa_abs = operators_ptr_->AbsA(aa);
//     auto aa_msb = operators_ptr_->HighestOneBit(aa_abs);

//     auto factor_aa = operators_ptr_->BitRevB(aa_msb, 0, 2 * log2_scale_);
//     aa = operators_ptr_->MulAB(aa_abs, factor_aa, log2_scale_);
//     aa = operators_ptr_->SquareA(aa);
//     aa = operators_ptr_->TruncA(aa, log2_scale_, SignType::Positive);

//     rr = operators_ptr_->MulAA(aa, rr);
//     rr = operators_ptr_->TruncA(rr, log2_scale_, SignType::Positive);
//     rr = operators_ptr_->MulAB(rr, factor_bb, log2_scale_);

//     rr = operators_ptr_->MulAB(rr, aa_msb, log2_scale_);
//     rr = operators_ptr_->MulAB(rr, aa_msb, log2_scale_);

//     rr = operators_ptr_->AddAA(rr, rr);
//     rr = operators_ptr_->AddAA(rr, rr);

//     return operators_ptr_->to(rr);
// }

std::vector<uint64_t> AnonXGB::compute_gain(
    const std::vector<uint64_t> &g_shr, const std::vector<uint64_t> &h_shr) {
  SPU_ENFORCE(g_shr.size() == h_shr.size());

  NdArrayRef g_arr = operators_ptr_->asAShares(operators_ptr_->from(g_shr));
  NdArrayRef h_arr = operators_ptr_->asAShares(operators_ptr_->from(h_shr));

  auto rsqrt_h = operators_ptr_->RSqrtA(h_arr, log2_scale_);
  
  auto gh =  operators_ptr_->MulAA(g_arr, rsqrt_h);
  gh = operators_ptr_->TruncA(gh, log2_scale_, SignType::Positive);
  auto gain_arr =  operators_ptr_->SquareA(gh);
  gain_arr = operators_ptr_->TruncA(gain_arr, log2_scale_, SignType::Positive);
  auto gain = operators_ptr_->to(gain_arr);

  return gain;
}

std::vector<uint64_t> AnonXGB::square_and_divide(
    const std::vector<uint64_t> &a_shr, const std::vector<uint64_t> &babs_shr) {
  SPU_ENFORCE(a_shr.size() == babs_shr.size());

  NdArrayRef aa = operators_ptr_->from(a_shr);

  aa = operators_ptr_->SquareA(aa);
  aa = operators_ptr_->TruncA(aa, log2_scale_, SignType::Positive);
  auto rr = operators_ptr_->to(aa);

  return divide(rr, babs_shr, SignType::Positive);
}

std::vector<uint64_t> AnonXGB::divide(const std::vector<uint64_t> &a_shr,
                                      const std::vector<uint64_t> &babs_shr,
                                      SignType sign) {
  SPU_ENFORCE(a_shr.size() == babs_shr.size());

  NdArrayRef aa = operators_ptr_->from(a_shr);
  NdArrayRef bb_abs = operators_ptr_->from(babs_shr);

  aa = operators_ptr_->asAShares(aa);
  bb_abs = operators_ptr_->asAShares(bb_abs);

  size_t fxp_goldschmidt_iters = 10;

  auto bb_msb = operators_ptr_->HighestOneBit(bb_abs);

  auto factor_bb = operators_ptr_->BitRevB(bb_msb, 0, 2 * log2_scale_);
  auto cc = operators_ptr_->MulAB(bb_abs, factor_bb, log2_scale_);

  auto k2_9142 = operators_ptr_->encodeD(2.9142F, cc.shape(), log2_scale_);
  auto rr = operators_ptr_->SubPA(k2_9142, operators_ptr_->AddAA(cc, cc));

  auto ee = operators_ptr_->MulAA(rr, cc);
  ee = operators_ptr_->TruncA(ee, log2_scale_, SignType::Positive);
  auto k1 = operators_ptr_->encodeD(1.0F, ee.shape(), log2_scale_);
  ee = operators_ptr_->SubPA(k1, ee);

  for (size_t itr = 0; itr < fxp_goldschmidt_iters; itr++) {
    rr = operators_ptr_->MulAA(rr, operators_ptr_->AddAP(ee, k1));
    rr = operators_ptr_->TruncA(rr, log2_scale_, SignType::Positive);

    if (itr + 1 < fxp_goldschmidt_iters) {
      ee = operators_ptr_->SquareA(ee);
      ee = operators_ptr_->TruncA(ee, log2_scale_, SignType::Positive);
    }
  }

  NdArrayRef is_aa_neg;
  NdArrayRef aa_abs;
  if (sign == SignType::Positive) {
    aa_abs = aa;
  } else if (sign == SignType::Negative) {
    aa_abs = operators_ptr_->NegA(aa);
  } else {
    is_aa_neg = operators_ptr_->MSB(aa);
    aa_abs = operators_ptr_->MuxA(is_aa_neg, aa, operators_ptr_->NegA(aa));
  }

  rr = operators_ptr_->MulAA(aa_abs, rr);
  rr = operators_ptr_->TruncA(rr, log2_scale_, SignType::Positive);
  rr = operators_ptr_->MulAB(rr, factor_bb, log2_scale_);

  if (sign == SignType::Positive) {
  } else if (sign == SignType::Negative) {
    rr = operators_ptr_->NegA(rr);
  } else {
    rr = operators_ptr_->MuxA(is_aa_neg, rr, operators_ptr_->NegA(rr));
  }

  return operators_ptr_->to(rr);
}

std::vector<uint64_t> AnonXGB::arg_max(const std::vector<uint64_t> &a_shr,
                                       size_t cols) {
  NdArrayRef aa = operators_ptr_->from(a_shr, cols);
  aa = operators_ptr_->asAShares(aa);
  aa = operators_ptr_->ArgMax(aa);
  return operators_ptr_->to(aa);
}

std::vector<uint64_t> mulAB(const std::vector<uint64_t> &a_shr,
                            const std::vector<uint64_t> &b_shr);

void AnonXGB::gen_tree(const std::vector<std::vector<uint8_t>> &split_bools,
                       double lambda, int tree_level, bool is_last_tree) {
  auto time_start = std::chrono::high_resolution_clock::now();

  auto bgn_bytes =
      link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;

  size_t begin = 0;
  size_t end = 1;

  std::vector<GradHess> gh_ashr0_vec;
  std::vector<GradHess> gh_ashr1_vec;

  gh_ashr0_vec.push_back(grad_hess_ashr0_);
  gh_ashr1_vec.push_back(grad_hess_ashr1_);

  std::vector<std::vector<uint8_t>> mask_bshr0_vec;
  std::vector<std::vector<uint8_t>> mask_bshr1_vec;

  mask_bshr0_vec.push_back(mask0_);
  mask_bshr1_vec.push_back(mask1_);

  std::vector<GradHess> hist_ashr0_vec;
  std::vector<GradHess> hist_ashr1_vec;

  size_t leaf_nums = 1ULL << tree_level;
  while ((end - begin) < leaf_nums) {
    std::vector<GradHess> hist_ashr0_lchild;
    std::vector<GradHess> hist_ashr1_lchild;

    for (size_t i = begin; i < end; i += 2) {
      hist_ashr0_lchild.push_back(gh_ashr0_vec[i]);
      hist_ashr1_lchild.push_back(gh_ashr1_vec[i]);
    }

    compute_histogram(hist_ashr0_lchild, hist_ashr1_lchild, split_bools);
    SPU_ENFORCE(hist_ashr0_vec.size() == begin &&
                hist_ashr1_vec.size() == begin);

    if (begin == 0) {
      hist_ashr0_vec.push_back(hist_ashr0_lchild[0]);
      hist_ashr1_vec.push_back(hist_ashr1_lchild[0]);
    } else {
      for (size_t i = begin, j = 0; i < end; i += 2, j++) {
        hist_ashr0_vec.push_back(hist_ashr0_lchild[j]);
        hist_ashr0_vec.push_back(hist_ashr0_vec[i / 2] - hist_ashr0_lchild[j]);

        hist_ashr1_vec.push_back(hist_ashr1_lchild[j]);
        hist_ashr1_vec.push_back(hist_ashr1_vec[i / 2] - hist_ashr1_lchild[j]);
      }
    }

    std::vector<GradHess> gh_ashr0;
    std::vector<GradHess> gh_ashr1;
    std::vector<GradHess> sum_gh_ashr0;
    std::vector<GradHess> sum_gh_ashr1;
    for (size_t i = begin; i < end; i++) {
      gh_ashr0.push_back(gh_ashr0_vec[i]);
      gh_ashr1.push_back(gh_ashr1_vec[i]);

      sum_gh_ashr0.push_back(gh_ashr0_vec[i].sum());
      sum_gh_ashr1.push_back(gh_ashr1_vec[i].sum());
    }

    std::vector<GradHess> level_hist_ashr0(hist_ashr0_vec.data() + begin,
                                           hist_ashr0_vec.data() + end);
    std::vector<GradHess> level_hist_ashr1(hist_ashr1_vec.data() + begin,
                                           hist_ashr1_vec.data() + end);
    std::vector<size_t> opt_cols = find_optimal_cols(
        lambda, level_hist_ashr0, level_hist_ashr1, sum_gh_ashr0, sum_gh_ashr1);

    share_bool_splits_and_multiplex(gh_ashr0, gh_ashr1, split_bools, opt_cols);

    for (size_t i = begin, j = 0; i < end; i++, j++) {
      GradHess lgh_ashr0 = gh_ashr0[j];
      GradHess rgh_ashr0 = gh_ashr0_vec[i] - gh_ashr0[j];

      auto lgrads0 = lgh_ashr0.getGrad();
      auto rgrads0 = rgh_ashr0.getGrad();

      vector<uint8_t> lmask_ashr0(lgrads0.size());
      vector<uint8_t> rmask_ashr0(rgrads0.size());

      for (size_t k = 0; k < lmask_ashr0.size(); k++) {
        lmask_ashr0[k] = lgrads0[k] & 0x1;
        rmask_ashr0[k] = rgrads0[k] & 0x1;
      }

      gh_ashr0_vec.push_back(lgh_ashr0);
      gh_ashr0_vec.push_back(rgh_ashr0);

      mask_bshr0_vec.push_back(lmask_ashr0);
      mask_bshr0_vec.push_back(rmask_ashr0);

      GradHess lgh_ashr1 = gh_ashr1[j];
      GradHess rgh_ashr1 = gh_ashr1_vec[i] - gh_ashr1[j];

      auto lgrads1 = lgh_ashr1.getGrad();
      auto rgrads1 = rgh_ashr1.getGrad();

      vector<uint8_t> lmask_ashr1(lgrads1.size());
      vector<uint8_t> rmask_ashr1(rgrads1.size());

      for (size_t k = 0; k < lmask_ashr1.size(); k++) {
        lmask_ashr1[k] = lgrads1[k] & 0x1;
        rmask_ashr1[k] = rgrads1[k] & 0x1;
      }

      gh_ashr1_vec.push_back(lgh_ashr1);
      gh_ashr1_vec.push_back(rgh_ashr1);

      mask_bshr1_vec.push_back(lmask_ashr1);
      mask_bshr1_vec.push_back(rmask_ashr1);
    }

    //            for (size_t i = begin; i < end; i++) {
    //                    grad_hess_sum_test(gh_ashr0_vec[i], gh_ashr1_vec[i],
    //                    "gen_tree");
    //            }

    begin = end;
    end = gh_ashr0_vec.size();
  }

  std::vector<uint64_t> leaf_grad_ashr;
  std::vector<uint64_t> leaf_hess_ashr;
  for (size_t i = begin; i < end; i++) {
    GradHess gh_sum = gh_ashr0_vec[i].sum();
    leaf_grad_ashr.push_back(gh_sum.getGrad()[0]);
    leaf_hess_ashr.push_back(gh_sum.getHess()[0]);
  }

  negate(leaf_grad_ashr);
  add_plain(leaf_hess_ashr, lambda, std::pow(2.0, log2_scale_));
  auto leaf_wt_ashr = divide(leaf_grad_ashr, leaf_hess_ashr, SignType::Unknown);
  leaf_wt_ashr_vec_.push_back(leaf_wt_ashr);

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
    std::vector<std::vector<uint8_t>> leaf_mask_bshr0_vec(
        mask_bshr0_vec.data() + begin, mask_bshr0_vec.data() + end);
    std::vector<std::vector<uint8_t>> leaf_mask_bshr1_vec(
        mask_bshr1_vec.data() + begin, mask_bshr1_vec.data() + end);
    update_tree_eval(leaf_mask_bshr0_vec, leaf_mask_bshr1_vec,
                     leaf_wt_ashr_vec_.back());
    update_grad_hess_ashr();
  }

  auto end_bytes =
      link_ctx_->GetStats()->sent_bytes + link_ctx_->GetStats()->recv_bytes;

  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
      time_end - time_start);
  std::cout << "gen_tree of depth " << tree_level << " takes "
            << time_diff.count() << " us" << std::endl;
  std::cout << "gen_tree of depth " << tree_level << " takes "
            << (end_bytes - bgn_bytes) << " comm bytes" << std::endl;

  //            grad_hess_sum_test(grad_hess_ashr0_, grad_hess_ashr1_, "tree
  //            eval update");
}

void AnonXGB::update_grad_hess_ashr() {
  auto concat = [&](const std::vector<uint64_t> &aa,
                    const std::vector<uint64_t> &bb) {
    std::vector<uint64_t> res(aa.begin(), aa.end());
    res.insert(res.end(), bb.begin(), bb.end());
    return res;
  };

  auto tree_eval_ashr = concat(tree_eval_ashr0_, tree_eval_ashr1_);
  auto pp_ashr = sigmoid(tree_eval_ashr); // scale is 2^{log2_scale_-1}
  auto pp_square_ashr = square(pp_ashr);
  truncate(pp_square_ashr, log2_scale_ - 1, SignType::Positive);

  std::vector<uint64_t> pp_ashr0(pp_ashr.begin(),
                                 pp_ashr.begin() + label_ashr0_.size());
  std::vector<uint64_t> pp_ashr1(pp_ashr.begin() + label_ashr0_.size(),
                                 pp_ashr.end());

  std::vector<uint64_t> pp_square_ashr0(
      pp_square_ashr.begin(), pp_square_ashr.begin() + label_ashr0_.size());
  std::vector<uint64_t> pp_square_ashr1(
      pp_square_ashr.begin() + label_ashr0_.size(), pp_square_ashr.end());

  auto gg_ashr0 = sub(pp_ashr0, label_ashr0_);
  auto hh_ashr0 = sub(pp_ashr0, pp_square_ashr0);
  for (size_t i = 0; i < gg_ashr0.size(); i++) {
    gg_ashr0[i] = (role_ == psi::PsiRoleType::Sender) ? gg_ashr0[i] + pp_ashr0[i] + 1 : gg_ashr0[i] + pp_ashr0[i];
    hh_ashr0[i] = (role_ == psi::PsiRoleType::Sender) ? hh_ashr0[i] + hh_ashr0[i] + 1 : hh_ashr0[i] + hh_ashr0[i];
  }
  grad_hess_ashr0_ = GradHess(gg_ashr0, hh_ashr0);

  // auto pp_get = recoveryA(gg_ashr0);
  // if (role_ == psi::PsiRoleType::Sender) {
  //   printf("after sigmoid, grad: ");
  //   for (size_t i = 0; i < pp_get.size(); i++) {
  //     printf("%f ", decode(pp_get[i], log2_scale_));
  //   }
  //   printf("\n");
  // }

  auto gg_ashr1 = sub(pp_ashr1, label_ashr1_);
  auto hh_ashr1 = sub(pp_ashr1, pp_square_ashr1);
  for (size_t i = 0; i < gg_ashr1.size(); i++) {
    gg_ashr1[i] = (role_ == psi::PsiRoleType::Sender) ? gg_ashr1[i] + pp_ashr1[i] + 1 : gg_ashr1[i] + pp_ashr1[i];
    hh_ashr1[i] = (role_ == psi::PsiRoleType::Sender) ? hh_ashr1[i] + hh_ashr1[i] + 1 : hh_ashr1[i] + hh_ashr1[i];
  }
  grad_hess_ashr1_ = GradHess(gg_ashr1, hh_ashr1);

  grad_hess_ashr0_ = multiplex(grad_hess_ashr0_, mask0_);
  grad_hess_ashr1_ = multiplex(grad_hess_ashr1_, mask1_);

  GradHess gh0 = grad_hess_ashr0_.sum();
  GradHess gh1 = grad_hess_ashr1_.sum();

  std::vector<uint64_t> gh_get = {gh0.getGrad()[0], gh0.getHess()[0], gh1.getGrad()[0], gh1.getHess()[0]};
  gh_get = recoveryA(gh_get);
  // if (role_ == psi::PsiRoleType::Sender) {
  //   printf("comp sum grad hess: %f, %f, %f, %f\n", 
  //     decode(gh_get[0], log2_scale_), decode(gh_get[1], log2_scale_), 
  //     decode(gh_get[2], log2_scale_), decode(gh_get[3], log2_scale_));
  // }
}

std::vector<size_t> find_optimal_cols(
    std::vector<std::shared_ptr<AnonXGB>> &ptr_vec, double lambda,
    const std::vector<GradHess> &lhist_ashr_vec0,
    const std::vector<GradHess> &lhist_ashr_vec1,
    const std::vector<GradHess> &sum_gh_ashr_vec0,
    const std::vector<GradHess> &sum_gh_ashr_vec1) {
  auto time_start = std::chrono::high_resolution_clock::now();

  SPU_ENFORCE(lhist_ashr_vec0.size() == lhist_ashr_vec1.size());
  SPU_ENFORCE(sum_gh_ashr_vec0.size() == sum_gh_ashr_vec1.size());
  SPU_ENFORCE(lhist_ashr_vec0.size() == sum_gh_ashr_vec1.size());

  std::atomic<size_t> begin_send_bytes = 0;
  std::atomic<size_t> begin_recv_bytes = 0;
  for (const auto &ptr : ptr_vec) {
    const auto &status = ptr->get_link_context()->GetStats();
    begin_send_bytes += status->sent_bytes;
    begin_recv_bytes += status->recv_bytes;
  }

  std::vector<uint64_t> gl_ashr_vec;
  std::vector<uint64_t> hl_ashr_vec;
  std::vector<uint64_t> gr_ashr_vec;
  std::vector<uint64_t> hr_ashr_vec;

  for (size_t i = 0; i < lhist_ashr_vec0.size(); i++) {
    GradHess sum_gh_ashr0 = sum_gh_ashr_vec0[i];
    GradHess sum_gh_ashr1 = sum_gh_ashr_vec1[i];

    GradHess rhist_gh_ashr0 =
        sum_gh_ashr0.extend(lhist_ashr_vec0[i].size()) - lhist_ashr_vec0[i];
    GradHess rhist_gh_ashr1 =
        sum_gh_ashr1.extend(lhist_ashr_vec1[i].size()) - lhist_ashr_vec1[i];

    GradHess lhist_gh_ashr =
        GradHess::concat(lhist_ashr_vec0[i], lhist_ashr_vec1[i]);
    GradHess rhist_gh_ashr = GradHess::concat(rhist_gh_ashr0, rhist_gh_ashr1);

    auto gl_ashr = lhist_gh_ashr.getGrad();
    auto hl_ashr = lhist_gh_ashr.getHess();
    auto gr_ashr = rhist_gh_ashr.getGrad();
    auto hr_ashr = rhist_gh_ashr.getHess();

    ptr_vec[0]->add_plain(hl_ashr, lambda,
                          std::pow(2.0, ptr_vec[0]->get_log2_scale()));
    ptr_vec[0]->add_plain(hr_ashr, lambda,
                          std::pow(2.0, ptr_vec[0]->get_log2_scale()));

    gl_ashr_vec.insert(gl_ashr_vec.end(), gl_ashr.begin(), gl_ashr.end());
    hl_ashr_vec.insert(hl_ashr_vec.end(), hl_ashr.begin(), hl_ashr.end());
    gr_ashr_vec.insert(gr_ashr_vec.end(), gr_ashr.begin(), gr_ashr.end());
    hr_ashr_vec.insert(hr_ashr_vec.end(), hr_ashr.begin(), hr_ashr.end());
  }

  auto concat = [&](const std::vector<uint64_t> &aa,
                    const std::vector<uint64_t> &bb) {
    std::vector<uint64_t> res(aa.begin(), aa.end());
    res.insert(res.end(), bb.begin(), bb.end());
    return res;
  };

  auto half_add = [&](const std::vector<uint64_t> &aa) {
    SPU_ENFORCE(!aa.empty() && (aa.size() & 0x1) == 0);

    size_t half_size = aa.size() >> 1U;
    std::vector<uint64_t> res(aa.begin(), aa.begin() + half_size);
    std::transform(res.begin(), res.end(), aa.begin() + half_size, res.begin(),
                   [&](uint64_t a, uint64_t b) { return a + b; });
    return res;
  };

  auto gg_ashr_vec = concat(gl_ashr_vec, gr_ashr_vec);
  auto hh_ashr_vec = concat(hl_ashr_vec, hr_ashr_vec);
  std::vector<uint64_t> ss_ashr(gg_ashr_vec.size());

  size_t num_thread = ptr_vec.size();
  size_t each_size = (ss_ashr.size() + num_thread - 1) / num_thread;
#pragma omp parallel num_threads(num_thread)
  {
    size_t tid = omp_get_thread_num();
    size_t bgn = std::min(tid * each_size, gg_ashr_vec.size());
    size_t end = std::min(bgn + each_size, gg_ashr_vec.size());

    if (bgn != end) {
      std::vector<uint64_t> sub_gg_ashr_vec(gg_ashr_vec.data() + bgn,
                                            gg_ashr_vec.data() + end);
      std::vector<uint64_t> sub_hh_ashr_vec(hh_ashr_vec.data() + bgn,
                                            hh_ashr_vec.data() + end);
      auto sub_ashr =
          ptr_vec[tid]->compute_gain(sub_gg_ashr_vec, sub_hh_ashr_vec);
          // ptr_vec[tid]->square_and_divide(sub_gg_ashr_vec, sub_hh_ashr_vec);
      std::copy_n(sub_ashr.data(), sub_ashr.size(), ss_ashr.data() + bgn);
    }
  }

  auto rr_ashr = half_add(ss_ashr);
  auto idx_ashr_vec = ptr_vec[0]->arg_max(rr_ashr, ptr_vec[0]->get_cols());

  // for test
  size_t total_cols = ptr_vec[0]->get_cols();
  // size_t offset = 0;
  // for (size_t i = 0; i < lhist_ashr_vec0.size(); i++) {
  //   std::vector<uint64_t> ashr(rr_ashr.data() + offset,
  //                              rr_ashr.data() + offset + total_cols);

  //   auto test_values = ptr_vec[0]->recoveryA(ashr);
  //   if (ptr_vec[0]->get_role() == psi::PsiRoleType::Sender) {
  //     printf("comp values: ");
  //     for (size_t ss = 0; ss < test_values.size(); ss++) {
  //       printf("%f ", ptr_vec[0]->decode(test_values[ss],
  //                                        ptr_vec[0]->get_log2_scale()));
  //     }
  //     printf("\n");
  //   }

  //   offset += total_cols;
  // }

  auto is_left_vec = ptr_vec[0]->is_left(idx_ashr_vec);
  auto value = ptr_vec[0]->get_role() == psi::PsiRoleType::Sender
    ? static_cast<uint64_t>(-1) : 0ULL;

  std::vector<uint64_t> values(is_left_vec.size() + is_left_vec.size());
  each_size = (is_left_vec.size() + num_thread - 1) / num_thread;
#pragma omp parallel num_threads(num_thread)
  {
    size_t tid = omp_get_thread_num();
    size_t bgn = std::min(tid * each_size, is_left_vec.size());
    size_t end = std::min(bgn + each_size, is_left_vec.size());

    if (bgn != end) {
      std::vector<uint64_t> sub_values(end - bgn, value);
      std::vector<uint64_t> sub_idx_ashr_vec(idx_ashr_vec.data() + bgn,
                                             idx_ashr_vec.data() + end);
      std::vector<uint8_t> sub_mask(is_left_vec.data() + bgn,
                                    is_left_vec.data() + end);

      GradHess v0 = GradHess(sub_values, sub_idx_ashr_vec);
      GradHess v1 = GradHess(sub_idx_ashr_vec, sub_values);

      GradHess vv = ptr_vec[tid]->multiplex(v1 - v0, sub_mask);
      vv = vv + v0;

      sub_values = concat(vv.getGrad(), vv.getHess());
      sub_values = ptr_vec[tid]->recoveryA(sub_values);

      std::copy_n(sub_values.data(), end - bgn, values.data() + bgn);
      std::copy_n(sub_values.data() + (end - bgn), end - bgn,
                  values.data() + is_left_vec.size() + bgn);
    }
  }

  std::vector<size_t> res(is_left_vec.size());
  for (size_t i = 0; i < res.size(); i++) {
    size_t j = i + res.size();
    bool bi = values[i] == static_cast<uint64_t>(-1);
    bool bj = values[j] == static_cast<uint64_t>(-1);

    SPU_ENFORCE(bi || (values[i] < (total_cols)), "value = {}", values[i]);
    SPU_ENFORCE(bj || (values[j] < (total_cols)), "value = {}", values[j]);
    SPU_ENFORCE(((!bi) && bj) || (bi && (!bj)));

    if (ptr_vec[0]->get_role() == psi::PsiRoleType::Sender) {
      res[i] = values[i];
    } else {
      res[i] = bj ? values[j] : values[j] - ptr_vec[0]->get_peer_cols();
    }
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
      time_end - time_start);

  std::atomic<size_t> end_send_bytes = 0;
  std::atomic<size_t> end_recv_bytes = 0;
  for (const auto &ptr : ptr_vec) {
    const auto &status = ptr->get_link_context()->GetStats();
    end_send_bytes += status->sent_bytes;
    end_recv_bytes += status->recv_bytes;
  }
  end_send_bytes -= begin_send_bytes;
  end_recv_bytes -= begin_recv_bytes;

  // if (ptr_vec[0]->get_role() == psi::PsiRoleType::Sender) {
  //   std::cout << "after find_optimal_cols: takes " << time_diff.count()
  //             << " us "
  //             << "send bytes = " << end_send_bytes
  //             << " recv bytes = " << end_recv_bytes << std::endl;
  //   ptr_vec[0]->get_link_context()->PrintStats();
  // }

  return res;
}

std::vector<double> AnonXGB::recoveryScoreInInference(
  const size_t &n, 
  std::vector<uint64_t> &scores) {
  vector<double> real_scores(n);
  auto plaintext_score_after_cuckoo_hash = recoveryA(scores);
  if (role_ == psi::PsiRoleType::Sender) {
    for (size_t i = 0; i < vole_cpsi_ptr0_->getNumBin(); i++) {
      auto item_index = vole_cpsi_ptr0_->getItemIndex(i);
      if (!item_index.empty()) {
        real_scores[item_index[0]] = decode(
          plaintext_score_after_cuckoo_hash[i], (enable_sigmoid_ ? log2_scale_ - 1 : log2_scale_)
        );
      }
    }
  }
  return real_scores;
}

std::vector<uint64_t> AnonXGB::bitwise_and(const std::vector<uint64_t> &values, size_t bitwidth) {
  size_t numel = values.size();
  bitwidth = bitwidth == 0 ? 64 : bitwidth;
  SPU_ENFORCE(bitwidth <= 64);
  std::vector<uint8_t> xx(numel * bitwidth);
  std::vector<uint8_t> yy(numel * bitwidth);

  constexpr size_t ot_msg_width = 1;

  auto *context = operators_ptr_->getContext();
  auto *comm = context->getState<Communicator>();
  auto *ot_state = context->getState<cheetah::CheetahOTState>();
  ot_state->LazyInit(comm, base_ot_, 0);
  auto ot_sender = ot_state->get(0)->GetSenderCOT();
  auto ot_receiver = ot_state->get(0)->GetReceiverCOT();

  if (role_ == psi::PsiRoleType::Sender) {
    ot_receiver->RecvRMRC(absl::MakeSpan(xx), absl::MakeSpan(yy),
      ot_msg_width);
  } else {
    ot_sender->SendRMRC(absl::MakeSpan(yy), absl::MakeSpan(xx),
      ot_msg_width);
    ot_sender->Flush();
    for (size_t i = 0; i < xx.size(); i++) {
      xx[i] = xx[i] ^ yy[i];
    }
  }

  std::vector<uint64_t> and_xs(numel);
  std::vector<uint64_t> and_ys(numel);

  for (size_t i = 0; i < numel; i++) {
    size_t bgn = i * bitwidth;
    size_t end = bgn + bitwidth;
    for (size_t j = bgn; j < end; j++) {
      and_xs[i] = (and_xs[i] << 1) | (xx[j] & 1);
      and_ys[i] = (and_ys[i] << 1) | (yy[j] & 1);
    }
  }

  auto ys = recoveryB(and_ys);
  if (role_ == psi::PsiRoleType::Sender) {
    std::vector<uint64_t> recv_xs(numel);
    yacl::Buffer buf = link_ctx_->Recv(link_ctx_->NextRank(), fmt::format("recv bytes"));
    std::memcpy(recv_xs.data(), reinterpret_cast<uint8_t *>(buf.data()), numel * sizeof(uint64_t));
    for (size_t i = 0; i < numel; i++) {
      SPU_ENFORCE((recv_xs[i] & and_xs[i]) == ys[i], "andTriple has problem");
    }
  } else {
    yacl::Buffer buf(numel * sizeof(uint64_t));
    std::memcpy(reinterpret_cast<uint8_t *>(buf.data()), and_xs.data(), numel * sizeof(uint64_t));
    link_ctx_->SendAsyncThrottled(link_ctx_->NextRank(), buf, fmt::format("send bytes {}", numel));
  }

  std::vector<uint64_t> open_vs(numel);
  for (size_t i = 0; i < numel; i++) {
    open_vs[i] = and_xs[i] ^ values[i];
  }

  std::vector<uint64_t> recv(numel);
  if (role_ == psi::PsiRoleType::Sender) {
    yacl::Buffer buf(numel * sizeof(uint64_t));
    std::memcpy(reinterpret_cast<uint8_t *>(buf.data()), open_vs.data(), numel * sizeof(uint64_t));
    link_ctx_->SendAsyncThrottled(link_ctx_->NextRank(), buf, fmt::format("send bytes {}", numel));

    buf = link_ctx_->Recv(link_ctx_->NextRank(), fmt::format("recv bytes"));
    std::memcpy(recv.data(), reinterpret_cast<uint8_t *>(buf.data()), numel * sizeof(uint64_t));
  } else {
    yacl::Buffer buf = link_ctx_->Recv(link_ctx_->NextRank(), fmt::format("recv bytes"));
    std::memcpy(recv.data(), reinterpret_cast<uint8_t *>(buf.data()), numel * sizeof(uint64_t));

    std::memcpy(reinterpret_cast<uint8_t *>(buf.data()), open_vs.data(), numel * sizeof(uint64_t));
    link_ctx_->SendAsyncThrottled(link_ctx_->NextRank(), buf, fmt::format("send bytes {}", numel));
  }
  
  uint64_t mask = bitwidth == 64 ? 0xFFFFFFFFFFFFFFFF : ((1ULL) << bitwidth) - 1;
  if (role_ == psi::PsiRoleType::Sender) {
    for (size_t i = 0; i < numel; i++) {
      recv[i] = ((recv[i] & values[i]) ^ and_ys[i]) & mask;
    }
  } else {
    for (size_t i = 0; i < numel; i++) {
      recv[i] = ((recv[i] & and_xs[i]) ^ and_ys[i]) & mask;
    }
  }

  return recv;
}

void AnonXGB::fill_tables_ot_sender_for_infer(
  size_t num_bins,
  vector<vector<uint8_t>> &indicators_receiver, 
  vector<vector<uint8_t>> &indicators_sender,
  vector<uint64_t> &tables, 
  vector<vector<uint64_t>> &rands, 
  std::vector<uint64_t> &weight,
  vector<uint8_t> &sel,
  size_t node_ind
) {
  for(size_t i = 0; i < num_bins; i++) { 
    if (indicators_receiver[node_ind][i] == 0) {
      tables[i*2] = -rands[node_ind][i];
      tables[i*2+1] = -rands[node_ind][i];
      sel[i] = 2;
    } else {
      sel[i] = indicators_sender[node_ind][i];
      if (indicators_sender[node_ind][i] == 0) {
        tables[i*2] = -rands[node_ind][i];
        tables[i*2+1] = weight[node_ind]-rands[node_ind][i];
      } else {
        tables[i*2] = weight[node_ind]-rands[node_ind][i];
        tables[i*2+1] = -rands[node_ind][i];
      }
    }
  }
}

void AnonXGB::fill_tables_ot_receiver_for_infer(
  size_t num_bins,
  vector<vector<uint8_t>> &indicators_sender,
  vector<uint64_t> tables,
  vector<vector<uint64_t>> &rands, 
  std::vector<uint64_t> &weight,
  size_t node_ind
) {
  for(size_t i = 0; i < num_bins; i++) { 
    if (indicators_sender[node_ind][i] == 0) {
      tables[i*3] = -rands[node_ind][i];
      tables[i*3+1] = weight[node_ind]-rands[node_ind][i];
      tables[i*3+2] = -rands[node_ind][i];
    } else {
      tables[i*3] = weight[node_ind]-rands[node_ind][i];
      tables[i*3+1] = -rands[node_ind][i];
      tables[i*3+2] = -rands[node_ind][i];
    }
  }
}

void AnonXGB::infer_acc_weights_with_lookup_table(
  size_t num_leaves, 
  size_t num_bins,
  vector<uint64_t> &packed_indicators_sender,
  vector<vector<uint8_t>> &indicators_receiver,
  std::vector<uint64_t> &weight,
  vector<uint64_t> &scores
) {
  NdArrayRef scores_a = operators_ptr_->from(scores);
  scores_a = operators_ptr_->asAShares(scores_a);
  vector<vector<uint8_t>> indicators_sender(num_leaves, vector<uint8_t>(num_bins, 0));
  for (size_t i = 0; i < num_leaves; i++) {
    for(size_t j = 0; j < num_bins; j++) {
      indicators_sender[i][j] = (packed_indicators_sender[j] >> (num_leaves-i-1)) & 1;
    }
  }
  auto *context = operators_ptr_->getContext();
  auto *comm = context->getState<Communicator>();
  auto *ot_state = context->getState<cheetah::CheetahOTState>();
  ot_state->LazyInit(comm, base_ot_, 0);
  auto ot_sender = ot_state->get(0)->GetSenderCOT();
  auto ot_receiver = ot_state->get(0)->GetReceiverCOT();
  
  vector<vector<uint64_t>> leaf_weights;
  vector<vector<uint64_t>> rands(num_leaves, vector<uint64_t>(num_bins));
  vector<uint64_t> temp_rands(num_leaves*num_bins);
  yacl::crypto::FillRand((char *)&temp_rands[0], temp_rands.size() * sizeof(uint64_t), false);

  for (size_t node_ind = 0; node_ind < num_leaves; node_ind++)
      for(size_t i = 0; i < num_bins; i++)
        rands[node_ind][i] = temp_rands[node_ind*num_bins+i];
  SPDLOG_INFO("secure rands finish");
  for (size_t node_ind = 0; node_ind < num_leaves; node_ind++) {
    if (role_ == psi::PsiRoleType::Sender) {
      vector<uint64_t> tables(num_bins*2);
      vector<uint8_t> sel(num_bins);
      vector<uint64_t> leaf_weight(num_bins);
      fill_tables_ot_sender_for_infer(num_bins, indicators_receiver,
        indicators_sender, tables, rands, weight, sel, node_ind);
      ot_sender->SendCMCC(absl::MakeSpan(tables), 2, 64);
      ot_receiver->RecvCMCC(absl::MakeSpan(sel), 3, absl::MakeSpan(leaf_weight), 64);
      ot_sender->Flush();
      for (size_t i = 0; i < num_bins; i++) {
        leaf_weight[i] += rands[node_ind][i];
      }
      leaf_weights.emplace_back(leaf_weight);
    } else {
      vector<uint64_t> tables(num_bins*3);
      fill_tables_ot_receiver_for_infer(num_bins, indicators_sender, tables, rands, weight, node_ind);
      vector<uint64_t> leaf_weight(num_bins);
      ot_receiver->RecvCMCC(absl::MakeSpan(indicators_sender[node_ind]), 2, absl::MakeSpan(leaf_weight), 64);
      ot_sender->SendCMCC(absl::MakeSpan(tables), 3, 64);
      ot_sender->Flush();
      for (size_t i = 0; i < num_bins; i++) {
        leaf_weight[i] += rands[node_ind][i];
      }
      leaf_weights.emplace_back(leaf_weight);
    }
  }

  for(size_t i = 0; i < num_leaves; i++) {
    NdArrayRef weight_a = operators_ptr_->from(leaf_weights[i]);
    weight_a = operators_ptr_->asAShares(weight_a);
    scores_a = operators_ptr_->AddAA(weight_a, scores_a);
  }

  scores = operators_ptr_->to(scores_a);
}

void AnonXGB::infer_acc_weights_with_mux(
  size_t num_leaves, 
  size_t num_bins,
  vector<uint64_t> &packed_indicators_sender,
  vector<vector<uint8_t>> &indicators_receiver,
  std::vector<uint64_t> &weight,
  vector<uint64_t> &scores) 
{
  vector<uint64_t> packed_indicators_receiver(num_bins,0);
  vector<uint64_t> packed_leaf_indicators(num_bins);

  if (role_ == psi::PsiRoleType::Sender) {
    for (size_t i = 0; i < num_bins; i++) {
      for (size_t j = 0; j < num_leaves; j++) {
        packed_indicators_receiver[i] = (packed_indicators_receiver[i] << 1) + indicators_receiver[j][i];
      }
    }
    packed_leaf_indicators = bitwise_and(packed_indicators_receiver, num_leaves);
    for (size_t i = 0; i < packed_indicators_receiver.size(); i++) {
      packed_leaf_indicators[i] ^= (packed_indicators_receiver[i] & packed_indicators_sender[i]);
    }
  } else {
    packed_leaf_indicators = bitwise_and(packed_indicators_sender, num_leaves);
  }

  vector<vector<uint8_t>> leaf_indicators(num_leaves, vector<uint8_t>(num_bins));
  for (size_t i = 0; i < num_bins; i++) {
    for(size_t j = 0; j < num_leaves; j++) {
      leaf_indicators[j][i] = (packed_leaf_indicators[i] >> (num_leaves-j-1)) & 1;
    }
  }

  auto current_score = multiplex_single(weight, leaf_indicators);
  for(size_t i = 0; i < current_score.size(); i++) {
    scores[i] += current_score[i];
  }
}

void AnonXGB::infer_acc_weights(
  size_t num_leaves, 
  size_t num_bins,
  vector<uint64_t> &packed_indicators_sender,
  vector<vector<uint8_t>> &indicators_receiver,
  std::vector<uint64_t> &weight,
  vector<uint64_t> &scores,
  bool enable_lookup_table_cal_weight) 
{
  if (enable_lookup_table_cal_weight) {
    infer_acc_weights_with_lookup_table(num_leaves, num_bins, packed_indicators_sender,
      indicators_receiver, weight, scores);
  } else { // mux(sum{mux(and(indicator_sender, indicator_sender_receiver), weigth)},cpsi_result)
    infer_acc_weights_with_mux(num_leaves, num_bins, packed_indicators_sender,
      indicators_receiver, weight, scores);
  }
}

std::vector<uint64_t> AnonXGB::infer(const size_t num_tree, 
                                const size_t depth,
                                std::vector<std::pair<std::vector<std::vector<std::tuple<int, double, double>>>,std::vector<uint64_t>>> &model,
                                std::vector<std::vector<double>> &feats,
                                bool enable_lookup_table_cal_weight) {
  SPU_ENFORCE(this->is_inference_==true, "only support inference");
  size_t n = feats.size();
  size_t num_bins = mask0_.size();
  vector<vector<vector<vector<uint8_t>>>> indicators(num_tree, vector<vector<vector<uint8_t>>>(depth + 1));
  for (size_t t = 0; t < num_tree; t++) {
    for (size_t d = 0; d < depth + 1; d++) {
      indicators[t][d] = vector<vector<uint8_t>>((int)pow(2,d), vector<uint8_t>(n,1));
    }
  }
  vector<uint64_t> scores(num_bins, encode(0, log2_scale_));


  for (size_t t = 0; t < num_tree; t++) {
    SPDLOG_INFO("infer for tree: {}", t);
    auto split_value = model[t].first;
    auto weight = model[t].second;
    for (size_t d = 0; d < depth; d++) {
      size_t num_nodes = 1 << d;
      for(size_t node_ind = 0; node_ind < num_nodes; node_ind++) {
        vector<uint8_t> &current_indicator = indicators[t][d][node_ind];
        vector<uint8_t> &left_child_indicator = indicators[t][d+1][node_ind*2];
        vector<uint8_t> &right_child_indicator = indicators[t][d+1][node_ind*2+1];
        const auto &current_split = split_value[d][node_ind];
        for (size_t i = 0; i < n; i++) {
          int feat_ind = get<0>(current_split);
          if (feat_ind != -1) {
            if (feats[i][feat_ind] < (get<2>(current_split) - 1e-10)) {
              left_child_indicator[i] = current_indicator[i];
              right_child_indicator[i] = 0;
            } else {
              left_child_indicator[i] = 0;
              right_child_indicator[i] = current_indicator[i];
            }
          } else {
            left_child_indicator[i] = current_indicator[i];
            right_child_indicator[i] = current_indicator[i];
          }
        }
      }
    }
    SPDLOG_INFO("local leaf indicator generating finish");

    // opprf share receiver's indicators of leaves
    size_t num_leaves = 1 << depth;
    vector<uint64_t> packed_indicators_sender(num_bins,0);
    vector<vector<uint8_t>> indicators_receiver(num_leaves);
    if (role_ == psi::PsiRoleType::Receiver) {
      vector<uint64_t> packed_indicators(n, 0);
      for(size_t i = 0; i < n; i++) {
        for (size_t node_ind = 0; node_ind < num_leaves; node_ind++) {
          packed_indicators[i] = (packed_indicators[i] << 1) + indicators[t][depth][node_ind][i];
        }
      }
      packed_indicators_sender = vole_cpsi_ptr0_->OpprfShareSender(packed_indicators, true);
    } else {
      for (size_t node_ind = 0; node_ind < num_leaves; node_ind++) {
        indicators_receiver[node_ind] = vole_cpsi_ptr0_->adjustOrder(indicators[t][depth][node_ind]);
      }
      packed_indicators_sender = vole_cpsi_ptr0_->OpprfShareReceiver(true);
    }
    
    SPDLOG_INFO("opprfshare sender's leaf indicator generating finish");

    // cal weight
    infer_acc_weights(num_leaves, num_bins, packed_indicators_sender, 
      indicators_receiver, weight, scores, enable_lookup_table_cal_weight);
    SPDLOG_INFO("calculating weight finish");
  }
  
  scores = multiplex(mask0_, scores);
  // sigmoid
  if (enable_sigmoid_){
    scores = sigmoid(scores);
    SPDLOG_INFO("sigmoid finish");
  }
  return scores;
}

std::vector<std::shared_ptr<AnonXGB>> initialize(
    std::vector<std::shared_ptr<yacl::link::Context>> &ctxs,
    const std::vector<std::string> &items, const std::vector<double> &labels,
    size_t cols, uint64_t log2_scale) {
  size_t items_size = items.size();
  size_t num_thread = ctxs.size();
  size_t each_size = (items_size + num_thread - 1) / num_thread;

  std::vector<std::shared_ptr<AnonXGB>> ptr_vec(num_thread);

  auto time_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(num_thread)
  {
    size_t tid = omp_get_thread_num();
    //            for (size_t tid = 0; tid < num_thread; tid++) {
    size_t bgn = tid * each_size;
    size_t end = std::min(bgn + each_size, items_size);

    auto role = labels.size() == items_size ? psi::PsiRoleType::Sender
                                            : psi::PsiRoleType::Receiver;
    ptr_vec[tid] = std::make_shared<AnonXGB>(ctxs[tid], role, log2_scale);

    std::vector<std::string> sub_items(items.data() + bgn, items.data() + end);
    std::vector<double> sub_labels(end - bgn);

    if (role == psi::PsiRoleType::Sender) {
      std::copy_n(labels.data() + bgn, sub_labels.size(), sub_labels.data());
    }

    ptr_vec[tid]->initialize(sub_items, sub_labels, cols, false);
  }

  ptr_vec[0]->init_histogram();
  auto [histogram_ptr0, histogram_ptr1] = ptr_vec[0]->get_histograms();
  for (size_t tid = 1; tid < num_thread; tid++) {
    ptr_vec[tid]->init_histogram(histogram_ptr0, histogram_ptr1);
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
      time_end - time_start);

  std::cout << "init takes " << time_diff.count() << " us and "
            << time_diff.count() << " us " << std::endl;

  return ptr_vec;
}

std::vector<std::shared_ptr<AnonXGB>> initialize_inference(
    std::vector<std::shared_ptr<yacl::link::Context>> &ctxs,
    const std::vector<std::string> &items, 
    psi::PsiRoleType role,
    size_t cols, uint64_t log2_scale) {
  size_t items_size = items.size();
  size_t num_thread = ctxs.size();
  size_t each_size = (items_size + num_thread - 1) / num_thread;

  std::vector<std::shared_ptr<AnonXGB>> ptr_vec(num_thread);

  auto time_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(num_thread)
  {
    size_t tid = omp_get_thread_num();
    //            for (size_t tid = 0; tid < num_thread; tid++) {
    size_t bgn = tid * each_size;
    size_t end = std::min(bgn + each_size, items_size);

    ptr_vec[tid] = std::make_shared<AnonXGB>(ctxs[tid], role, log2_scale, true, false);

    std::vector<std::string> sub_items(items.data() + bgn, items.data() + end);


    ptr_vec[tid]->initialize_inference(sub_items, cols);
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(
      time_end - time_start);

  std::cout << "init takes " << time_diff.count() << " us and "
            << time_diff.count() << " us " << std::endl;

  return ptr_vec;
}

void initialize_gh_and_mask(
        std::vector<std::shared_ptr<AnonXGB>> &ptr_vec,
        std::vector<std::vector<GradHess>> &gh_ashr0_vec,
        std::vector<std::vector<GradHess>> &gh_ashr1_vec,
        std::vector<std::vector<std::vector<uint8_t>>> &mask_bshr0_vec,
        std::vector<std::vector<std::vector<uint8_t>>> &mask_bshr1_vec) {
    size_t num_thread = ptr_vec.size();
#pragma omp parallel num_threads(num_thread)
    {
        size_t tid = omp_get_thread_num();
        auto anon_xgb_ptr = ptr_vec[tid];
        gh_ashr0_vec[tid].push_back(anon_xgb_ptr->get_grad_hess_ashr0());
        gh_ashr1_vec[tid].push_back(anon_xgb_ptr->get_grad_hess_ashr1());
        mask_bshr0_vec[tid].push_back(anon_xgb_ptr->get_mask0());
        mask_bshr1_vec[tid].push_back(anon_xgb_ptr->get_mask1());
    }
}

void process_tree_levels(
        std::vector<std::shared_ptr<AnonXGB>> &ptr_vec,
        const std::vector<std::vector<uint8_t>> &split_bools,
        double lambda, int tree_level,
        size_t num_thread, size_t each_size,
        std::vector<std::vector<GradHess>> &gh_ashr0_vec,
        std::vector<std::vector<GradHess>> &gh_ashr1_vec,
        std::vector<std::vector<std::vector<uint8_t>>> &mask_bshr0_vec,
        std::vector<std::vector<std::vector<uint8_t>>> &mask_bshr1_vec,
        std::vector<std::vector<size_t>> &opt_cols_vec,
        std::atomic<size_t> &histogram_comm_bytes,
        std::atomic<size_t> &find_best_split_comm_bytes,
        std::atomic<size_t> &sync_split_comm_bytes,
        std::chrono::milliseconds &histogram_time,
        std::chrono::milliseconds &find_best_split_time,
        std::chrono::milliseconds &sync_split_time,
        size_t &begin,
        size_t &end) {
    size_t leaf_nums = 1ULL << tree_level;
    auto time_start_each = std::chrono::high_resolution_clock::now();
    auto time_end_each = time_start_each;

    std::vector<GradHess> hist_ashr0_vec;
    std::vector<GradHess> hist_ashr1_vec;

    auto get_comm_bytes = [&](std::atomic<size_t> &send_bytes, std::atomic<size_t> &recv_bytes) {
        send_bytes = 0;
        recv_bytes = 0;
        for (const auto &ptr : ptr_vec) {
            send_bytes += ptr->get_link_context()->GetStats()->sent_bytes;
            recv_bytes += ptr->get_link_context()->GetStats()->recv_bytes;
        }
    };

    std::atomic<size_t> init_sent_bytes = 0;
    std::atomic<size_t> init_recv_bytes = 0;
    std::atomic<size_t> sent_bytes = 0;
    std::atomic<size_t> recv_bytes = 0;
    std::atomic<size_t> run_sent_bytes = 0;
    std::atomic<size_t> run_recv_bytes = 0;

    while ((end - begin) < leaf_nums) {
        std::vector<std::vector<GradHess>> hist_ashr0_lchilds(num_thread);
        std::vector<std::vector<GradHess>> hist_ashr1_lchilds(num_thread);

        get_comm_bytes(init_sent_bytes, init_recv_bytes);
#pragma omp parallel num_threads(num_thread)
        {
            size_t tid = omp_get_thread_num();
            //            for (size_t tid = 0; tid < num_thread; tid++) {

            size_t start = tid * each_size;
            size_t finish = std::min(start + each_size, split_bools.size());
            std::vector<std::vector<uint8_t>> sub_split_bools(
                    split_bools.data() + start, split_bools.data() + finish);

            for (size_t i = begin; i < end; i += 2) {
                hist_ashr0_lchilds[tid].push_back(gh_ashr0_vec[tid][i]);
                hist_ashr1_lchilds[tid].push_back(gh_ashr1_vec[tid][i]);
            }

            ptr_vec[tid]->compute_histogram(hist_ashr0_lchilds[tid],
                                            hist_ashr1_lchilds[tid], sub_split_bools);
        }
        SPU_ENFORCE(hist_ashr0_vec.size() == begin &&
                    hist_ashr1_vec.size() == begin);
        get_comm_bytes(sent_bytes, recv_bytes);
        run_sent_bytes = (sent_bytes - init_sent_bytes);
        run_recv_bytes = (recv_bytes - init_recv_bytes);

        time_end_each = std::chrono::high_resolution_clock::now();
        histogram_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                time_end_each - time_start_each);
        time_start_each = time_end_each;

        std::vector<GradHess> hist_ashr0_lchild(hist_ashr0_lchilds[0].begin(),
                                                hist_ashr0_lchilds[0].end());
        std::vector<GradHess> hist_ashr1_lchild(hist_ashr1_lchilds[0].begin(),
                                                hist_ashr1_lchilds[0].end());

        for (size_t tid = 1; tid < num_thread; tid++) {
            std::transform(hist_ashr0_lchilds[tid].begin(),
                           hist_ashr0_lchilds[tid].end(), hist_ashr0_lchild.begin(),
                           hist_ashr0_lchild.begin(),
                           [&](GradHess &v1, GradHess &v2) { return v1 + v2; });

            std::transform(hist_ashr1_lchilds[tid].begin(),
                           hist_ashr1_lchilds[tid].end(), hist_ashr1_lchild.begin(),
                           hist_ashr1_lchild.begin(),
                           [&](GradHess &v1, GradHess &v2) { return v1 + v2; });
        }

        if (begin == 0) {
            hist_ashr0_vec.push_back(hist_ashr0_lchild[0]);
            hist_ashr1_vec.push_back(hist_ashr1_lchild[0]);
        } else {
            for (size_t i = begin, j = 0; i < end; i += 2, j++) {
                hist_ashr0_vec.push_back(hist_ashr0_lchild[j]);
                hist_ashr0_vec.push_back(hist_ashr0_vec[i / 2] - hist_ashr0_lchild[j]);

                hist_ashr1_vec.push_back(hist_ashr1_lchild[j]);
                hist_ashr1_vec.push_back(hist_ashr1_vec[i / 2] - hist_ashr1_lchild[j]);
            }
        }

        std::vector<GradHess> sum_gh_ashr0;
        std::vector<GradHess> sum_gh_ashr1;

        for (size_t i = begin; i < end; i++) {
            GradHess gh_ashr0_sum = GradHess(0, 0);  // gh_ashr0_vec[0][i].sum();
            GradHess gh_ashr1_sum = GradHess(0, 0);  // gh_ashr1_vec[0][i].sum();

            for (size_t tid = 0; tid < num_thread; tid++) {
                gh_ashr0_sum = gh_ashr0_sum + gh_ashr0_vec[tid][i].sum();
                gh_ashr1_sum = gh_ashr1_sum + gh_ashr1_vec[tid][i].sum();
            }

            sum_gh_ashr0.push_back(gh_ashr0_sum);
            sum_gh_ashr1.push_back(gh_ashr1_sum);
        }

        std::vector<GradHess> level_hist_ashr0(hist_ashr0_vec.data() + begin,
                                               hist_ashr0_vec.data() + end);
        std::vector<GradHess> level_hist_ashr1(hist_ashr1_vec.data() + begin,
                                               hist_ashr1_vec.data() + end);
        get_comm_bytes(init_sent_bytes, init_recv_bytes);
        std::vector<size_t> opt_cols =
                find_optimal_cols(ptr_vec, lambda, level_hist_ashr0, level_hist_ashr1,
                                  sum_gh_ashr0, sum_gh_ashr1);
        opt_cols_vec.push_back(opt_cols);

        get_comm_bytes(sent_bytes, recv_bytes);
        run_sent_bytes = (sent_bytes - init_sent_bytes);
        run_recv_bytes = (recv_bytes - init_recv_bytes);
        find_best_split_comm_bytes = find_best_split_comm_bytes + run_sent_bytes + run_recv_bytes;

        time_end_each = std::chrono::high_resolution_clock::now();
        find_best_split_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                time_end_each - time_start_each);
        time_start_each = time_end_each;

        std::vector<std::vector<GradHess>> gh_ashr0_splits(num_thread);
        std::vector<std::vector<GradHess>> gh_ashr1_splits(num_thread);

        for (size_t i = begin; i < end; i++)
            for (size_t tid = 0; tid < num_thread; tid++) {
                gh_ashr0_splits[tid].push_back(gh_ashr0_vec[tid][i]);
                gh_ashr1_splits[tid].push_back(gh_ashr1_vec[tid][i]);
            }
        get_comm_bytes(init_sent_bytes, init_recv_bytes);
#pragma omp parallel num_threads(num_thread)
        {
            size_t tid = omp_get_thread_num();
            //            for (size_t tid = 0; tid < num_thread; tid++) {
            size_t start = tid * each_size;
            size_t finish = std::min(start + each_size, split_bools.size());
            std::vector<std::vector<uint8_t>> sub_split_bools(
                    split_bools.data() + start, split_bools.data() + finish);

            ptr_vec[tid]->share_bool_splits_and_multiplex(gh_ashr0_splits[tid], gh_ashr1_splits[tid],
                                                          sub_split_bools, opt_cols);

            for (size_t i = begin, j = 0; i < end; i++, j++) {
                GradHess lgh_ashr0 = gh_ashr0_splits[tid][j];
                GradHess rgh_ashr0 = gh_ashr0_vec[tid][i] - gh_ashr0_splits[tid][j];

                auto lgrads0 = lgh_ashr0.getGrad();
                auto rgrads0 = rgh_ashr0.getGrad();

                vector<uint8_t> lmask_ashr0(lgrads0.size());
                vector<uint8_t> rmask_ashr0(rgrads0.size());
                for (size_t k = 0; k < lmask_ashr0.size(); k++) {
                    lmask_ashr0[k] = lgrads0[k] & 0x1;
                    rmask_ashr0[k] = rgrads0[k] & 0x1;
                }

                gh_ashr0_vec[tid].push_back(lgh_ashr0);
                gh_ashr0_vec[tid].push_back(rgh_ashr0);
                mask_bshr0_vec[tid].push_back(lmask_ashr0);
                mask_bshr0_vec[tid].push_back(rmask_ashr0);

                GradHess lgh_ashr1 = gh_ashr1_splits[tid][j];
                GradHess rgh_ashr1 = gh_ashr1_vec[tid][i] - gh_ashr1_splits[tid][j];

                auto lgrads1 = lgh_ashr1.getGrad();
                auto rgrads1 = rgh_ashr1.getGrad();

                vector<uint8_t> lmask_ashr1(lgrads1.size());
                vector<uint8_t> rmask_ashr1(rgrads1.size());
                for (size_t k = 0; k < lmask_ashr1.size(); k++) {
                    lmask_ashr1[k] = lgrads1[k] & 0x1;
                    rmask_ashr1[k] = rgrads1[k] & 0x1;
                }

                gh_ashr1_vec[tid].push_back(lgh_ashr1);
                gh_ashr1_vec[tid].push_back(rgh_ashr1);
                mask_bshr1_vec[tid].push_back(lmask_ashr1);
                mask_bshr1_vec[tid].push_back(rmask_ashr1);
            }
        }
        get_comm_bytes(sent_bytes, recv_bytes);
        run_sent_bytes = (sent_bytes - init_sent_bytes);
        run_recv_bytes = (recv_bytes - init_recv_bytes);
        sync_split_comm_bytes = sync_split_comm_bytes + run_sent_bytes + run_recv_bytes;
        time_end_each = std::chrono::high_resolution_clock::now();
        sync_split_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                time_end_each - time_start_each);
        time_start_each = time_end_each;


        begin = end;
        end = gh_ashr0_vec[0].size();
    }
}

std::vector<uint64_t> compute_leaf_weights(
        std::vector<std::shared_ptr<AnonXGB>> &ptr_vec,
        size_t &begin, size_t &end, size_t num_thread,
        std::vector<std::vector<GradHess>> &gh_ashr0_vec,
        double lambda,
        std::atomic<size_t> &cal_weight_comm_bytes,
        std::chrono::milliseconds &cal_weight_time) {
    auto time_start = std::chrono::high_resolution_clock::now();
    std::vector<uint64_t> leaf_grad_ashr;
    std::vector<uint64_t> leaf_hess_ashr;

    for (size_t i = begin; i < end; i++) {
        GradHess gh_sum = gh_ashr0_vec[0][i].sum();
        for (size_t tid = 1; tid < num_thread; tid++) {
            gh_sum = gh_sum + gh_ashr0_vec[tid][i].sum();
        }
        leaf_grad_ashr.push_back(gh_sum.getGrad()[0]);
        leaf_hess_ashr.push_back(gh_sum.getHess()[0]);
    }

    std::atomic<size_t> init_sent_bytes = 0;
    std::atomic<size_t> init_recv_bytes = 0;
    std::atomic<size_t> sent_bytes = 0;
    std::atomic<size_t> recv_bytes = 0;
    std::atomic<size_t> run_sent_bytes = 0;
    std::atomic<size_t> run_recv_bytes = 0;

    auto get_comm_bytes = [&](std::atomic<size_t> &send_bytes, std::atomic<size_t> &recv_bytes) {
        send_bytes = 0;
        recv_bytes = 0;
        for (const auto &ptr : ptr_vec) {
            send_bytes += ptr->get_link_context()->GetStats()->sent_bytes;
            recv_bytes += ptr->get_link_context()->GetStats()->recv_bytes;
        }
    };

    get_comm_bytes(init_sent_bytes, init_recv_bytes);
    ptr_vec[0]->negate(leaf_grad_ashr);
    ptr_vec[0]->add_plain(leaf_hess_ashr, lambda,
                          std::pow(2.0, ptr_vec[0]->get_log2_scale()));
    auto leaf_wt_ashr =
            ptr_vec[0]->divide(leaf_grad_ashr, leaf_hess_ashr, SignType::Unknown);

    get_comm_bytes(sent_bytes, recv_bytes);
    run_sent_bytes = (sent_bytes - init_sent_bytes);
    run_recv_bytes = (recv_bytes - init_recv_bytes);
    cal_weight_comm_bytes = run_sent_bytes + run_recv_bytes;

    auto time_end = std::chrono::high_resolution_clock::now();
    cal_weight_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);

    return leaf_wt_ashr;
}

void update_tree_and_gradients(
        std::vector<std::shared_ptr<AnonXGB>> &ptr_vec,
        bool is_last_tree,
        size_t &begin, size_t &end, size_t num_thread,
        std::vector<std::vector<std::vector<uint8_t>>> &mask_bshr0_vec,
        std::vector<std::vector<std::vector<uint8_t>>> &mask_bshr1_vec,
        const std::vector<uint64_t> &leaf_wt_ashr,
        std::atomic<size_t> &cal_gradients_comm_bytes,
        std::chrono::milliseconds &cal_gradients_time) {

    if (is_last_tree) return;

    auto time_start = std::chrono::high_resolution_clock::now();
    std::atomic<size_t> init_sent_bytes = 0;
    std::atomic<size_t> init_recv_bytes = 0;
    std::atomic<size_t> sent_bytes = 0;
    std::atomic<size_t> recv_bytes = 0;
    std::atomic<size_t> run_sent_bytes = 0;
    std::atomic<size_t> run_recv_bytes = 0;

    auto get_comm_bytes = [&](std::atomic<size_t> &send_bytes, std::atomic<size_t> &recv_bytes) {
        send_bytes = 0;
        recv_bytes = 0;
        for (const auto &ptr : ptr_vec) {
            send_bytes += ptr->get_link_context()->GetStats()->sent_bytes;
            recv_bytes += ptr->get_link_context()->GetStats()->recv_bytes;
        }
    };

    get_comm_bytes(init_sent_bytes, init_recv_bytes);
#pragma omp parallel num_threads(num_thread)
    {
        size_t tid = omp_get_thread_num();
        std::vector<std::vector<uint8_t>> leaf_mask_bshr0_vec(
                mask_bshr0_vec[tid].data() + begin, mask_bshr0_vec[tid].data() + end);
        std::vector<std::vector<uint8_t>> leaf_mask_bshr1_vec(
                mask_bshr1_vec[tid].data() + begin, mask_bshr1_vec[tid].data() + end);
        ptr_vec[tid]->update_tree_eval(leaf_mask_bshr0_vec, leaf_mask_bshr1_vec,
                                       leaf_wt_ashr);
        ptr_vec[tid]->update_grad_hess_ashr();
    }

    get_comm_bytes(sent_bytes, recv_bytes);
    run_sent_bytes = (sent_bytes - init_sent_bytes);
    run_recv_bytes = (recv_bytes - init_recv_bytes);
    cal_gradients_comm_bytes = run_sent_bytes + run_recv_bytes;

    auto time_end = std::chrono::high_resolution_clock::now();
    cal_gradients_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
}

std::pair<std::vector<std::vector<size_t>>, std::vector<uint64_t>>
gen_tree_parallel(std::vector<std::shared_ptr<AnonXGB>> &ptr_vec,
                  const std::vector<std::vector<uint8_t>> &split_bools,
                  double lambda, int tree_level, bool is_last_tree) {
    size_t num_thread = ptr_vec.size();
    size_t each_size = (split_bools.size() + num_thread - 1) / num_thread;
    size_t begin = 0;
    size_t end = 1;

    // init
    std::atomic<size_t> histogram_comm_bytes = 0;
    std::atomic<size_t> find_best_split_comm_bytes = 0;
    std::atomic<size_t> sync_split_comm_bytes = 0;
    std::atomic<size_t> cal_gradients_comm_bytes = 0;
    std::atomic<size_t> cal_weight_comm_bytes = 0;
    std::atomic<size_t> sync_indicator_comm_bytes = 0;
    std::atomic<size_t> update_indicator_comm_bytes = 0;

    std::chrono::milliseconds histogram_time = 0ms;
    std::chrono::milliseconds find_best_split_time = 0ms;
    std::chrono::milliseconds sync_split_time = 0ms;
    std::chrono::milliseconds cal_gradients_time = 0ms;
    std::chrono::milliseconds cal_weight_time = 0ms;

    // init gradient and mask
    std::vector<std::vector<GradHess>> gh_ashr0_vec(num_thread);
    std::vector<std::vector<GradHess>> gh_ashr1_vec(num_thread);
    std::vector<std::vector<std::vector<uint8_t>>> mask_bshr0_vec(num_thread);
    std::vector<std::vector<std::vector<uint8_t>>> mask_bshr1_vec(num_thread);
    initialize_gh_and_mask(ptr_vec, gh_ashr0_vec, gh_ashr1_vec,
                           mask_bshr0_vec, mask_bshr1_vec);

    // generate intermidiate tree nodes
    std::cout << "begin generate intermidiate tree nodes" << std::endl;
    std::vector<std::vector<size_t>> opt_cols_vec;
    process_tree_levels(ptr_vec, split_bools, lambda, tree_level,
                        num_thread, each_size, gh_ashr0_vec, gh_ashr1_vec,
                        mask_bshr0_vec, mask_bshr1_vec, opt_cols_vec,
                        histogram_comm_bytes, find_best_split_comm_bytes,
                        sync_split_comm_bytes, histogram_time,
                        find_best_split_time, sync_split_time,
                        begin, end);

    // compute leaf weights
    std::cout << "compute leaf weights" << std::endl;
    auto leaf_wt_ashr = compute_leaf_weights(ptr_vec, begin, end, num_thread,
                                             gh_ashr0_vec, lambda,
                                             cal_weight_comm_bytes,
                                             cal_weight_time);

    // update tree and gradients
    std::cout << "begin update tree and gradients" << std::endl;
    update_tree_and_gradients(ptr_vec, is_last_tree, begin, end, num_thread,
                              mask_bshr0_vec, mask_bshr1_vec, leaf_wt_ashr,
                              cal_gradients_comm_bytes, cal_gradients_time);

    for (const auto &ptr : ptr_vec) {
        sync_indicator_comm_bytes += ptr->sync_indicator_comm_bytes;
        update_indicator_comm_bytes += ptr->update_indicator_comm_bytes;
    }

    if (ptr_vec[0]->get_role() == psi::PsiRoleType::Sender) {
        std::cout << "=== Evaluation Result of Each Components in one Tree ===\n";
        std::cout << "run histogram takes " << histogram_time.count() << " ms, costs "
                  << histogram_comm_bytes << " bytes\n";
        std::cout << "find best split takes " << find_best_split_time.count() << " ms, costs "
                  << find_best_split_comm_bytes << " bytes\n";
        std::cout << "sync split takes " << sync_split_time.count() << " ms, costs "
                  << sync_split_comm_bytes << " bytes\n";
        std::cout << " - sync indicator cost " << sync_indicator_comm_bytes << " bytes\n";
        std::cout << " - update indicator cost " << update_indicator_comm_bytes << " bytes\n";
        std::cout << "cal gradients takes " << cal_gradients_time.count() << " ms, costs "
                  << cal_gradients_comm_bytes << " bytes\n";
        std::cout << "cal weight takes " << cal_weight_time.count() << " ms, costs "
                  << cal_weight_comm_bytes << " bytes\n";
        std::cout << "========================================================\n";
    }

    return std::make_pair(opt_cols_vec, leaf_wt_ashr);
}


std::vector<double> infer_parallel(std::vector<std::shared_ptr<AnonXGB>> &ptr_vec,
                                    const size_t &num_tree, 
                                    const size_t &depth,
                                    std::vector<std::pair<std::vector<std::vector<std::tuple<int, double, double>>>,std::vector<uint64_t>>> &model,
                                    std::vector<std::vector<double>> &feats,
                                    bool enable_lookup_table_cal_weight) {
  size_t num_samples = feats.size();
  size_t num_thread = ptr_vec.size();
  size_t each_size = (num_samples + num_thread - 1) / num_thread;
  std::vector<double> plaintext_scores(num_samples);

#pragma omp parallel num_threads(num_thread)
  {
    size_t tid = omp_get_thread_num();
    size_t bgn = tid * each_size;
    size_t end = std::min(bgn + each_size, num_samples);
    vector<vector<double>> sub_feats(feats.begin()+bgn, feats.begin()+end);
    for (size_t i = bgn; i < end; i++) {
      sub_feats[i-bgn] = feats[i];
    }
    vector<uint64_t> sub_scores = ptr_vec[tid]->infer(num_tree, depth, model, sub_feats, enable_lookup_table_cal_weight);
    vector<double> sub_plaintext_scores = ptr_vec[tid]->recoveryScoreInInference(end-bgn, sub_scores);


    std::copy_n(sub_plaintext_scores.begin(), end-bgn, plaintext_scores.begin()+bgn);
  }
  return plaintext_scores;
}

}  // namespace spu::mpc::cheetah
