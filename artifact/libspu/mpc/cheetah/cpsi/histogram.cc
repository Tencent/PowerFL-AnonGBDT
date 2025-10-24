
#include "libspu/mpc/cheetah/cpsi/histogram.h"

#include "yacl/crypto/rand/rand.h"
#include "yacl/utils/parallel.h"

#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

using namespace std;

HistoGram::HistoGram(std::shared_ptr<yacl::link::Context> link_ctx,
                     psi::PsiRoleType role)
    : link_ctx_(std::move(link_ctx)), role_(role) {
  //        auto conn = std::make_shared<Communicator>(link_ctx_);
  //        base_ot_ = std::make_shared<BasicOTProtocols>(conn);
}

void HistoGram::initialize(bool use_ckks) {
  use_ckks_ = use_ckks;

  if (role_ == psi::PsiRoleType::Sender) {
    yacl::Buffer buf = link_ctx_->Recv(link_ctx_->NextRank(),
                                       fmt::format("recv public params"));

    vector<vector<uint8_t>> bytes = deserialze(buf);

    if (use_ckks_) {
      SPU_ENFORCE(bytes.size() == 4, "initialize communication of CKKS failed");

      seal::EncryptionParameters parms(seal::scheme_type::ckks);

      uint64_t lwe_coeff_count = 4096;
      uint64_t coeff_count = 8192;

      parms.set_poly_modulus_degree(lwe_coeff_count);
      parms.set_coeff_modulus(
          seal::CoeffModulus::Create(lwe_coeff_count, {60, 49}));
      seal::SEALContext context0(parms);

      parms.set_poly_modulus_degree(coeff_count);
      parms.set_coeff_modulus(
          seal::CoeffModulus::Create(coeff_count, {60, 49, 60}));
      seal::SEALContext context1(parms);

      vector<uint8_t> rk{};
      vector<uint8_t> gk{};
      ckks_ptr0_ = make_shared<CKKSParams>(context0, bytes[0], rk, gk);
      ckks_ptr1_ = make_shared<CKKSParams>(context1, bytes[1], rk, bytes[2]);

      ckks_ptr1_->bytes_to_switchKeys(bytes[3], switch_keys_);
    } else {
      SPU_THROW("only support ckks");
    }

  } else {
    vector<vector<uint8_t>> bytes;

    if (use_ckks_) {
      seal::EncryptionParameters parms(seal::scheme_type::ckks);

      uint64_t lwe_coeff_count = 4096;
      uint64_t coeff_count = 8192;

      parms.set_poly_modulus_degree(lwe_coeff_count);
      parms.set_coeff_modulus(
          seal::CoeffModulus::Create(lwe_coeff_count, {60, 49}));
      seal::SEALContext context0(parms);

      parms.set_poly_modulus_degree(coeff_count);
      parms.set_coeff_modulus(
          seal::CoeffModulus::Create(coeff_count, {60, 49, 60}));
      seal::SEALContext context1(parms);

      auto ckks_ptr0 = make_shared<CKKSParams>(context0, false, false);

      ckks_ptr0_ = make_shared<CKKSParams>(context0, false, false);
      ckks_ptr1_ = make_shared<CKKSParams>(context1, false, true);

      vector<uint64_t> lwe_sk;
      ckks_ptr0_->get_secret_key(lwe_sk);
      ckks_ptr1_->gen_switch_keys(lwe_sk, switch_keys_);

      bytes.resize(4);

      ckks_ptr0_->pk_to_bytes(bytes[0]);
      ckks_ptr1_->pk_to_bytes(bytes[1]);
      ckks_ptr1_->gk_to_bytes(bytes[2]);
      ckks_ptr1_->switchKeys_to_bytes(switch_keys_, bytes[3]);
    } else {
      SPU_THROW("only support ckks");
    }

    yacl::Buffer buf = serialize(bytes);
    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf,
        fmt::format("send public params of size {} bytes", buf.size()));
  }
}

void HistoGram::set_ckks_params(std::shared_ptr<HistoGram> &histogram_ptr) {
  this->ckks_ptr0_ = std::make_shared<CKKSParams>(histogram_ptr->ckks_ptr0_);
  this->ckks_ptr1_ = std::make_shared<CKKSParams>(histogram_ptr->ckks_ptr1_);

  this->switch_keys_ = histogram_ptr->switch_keys_;
}

yacl::Buffer HistoGram::serialize(const vector<vector<uint8_t>> &bytes,
                                  bool sameSize) {
  size_t len = sameSize ? 0 : sizeof(size_t) * bytes.size() + sizeof(size_t);

  if (sameSize) {
    for (size_t i = 1; i < bytes.size(); i++) {
      SPU_ENFORCE(bytes[i].size() == bytes[0].size());
    }
  }

  for (size_t i = 0; i < bytes.size(); i++) {
    len += bytes[i].size();
  }

  yacl::Buffer buf(len);
  size_t pos = 0;
  for (size_t i = 0; i < bytes.size(); i++) {
    std::memcpy(reinterpret_cast<uint8_t *>(buf.data()) + pos, bytes[i].data(),
                bytes[i].size());
    pos += bytes[i].size();
  }

  if (!sameSize) {
    for (size_t i = 0; i < bytes.size(); i++) {
      size_t each_size = bytes[i].size();
      std::memcpy(reinterpret_cast<uint8_t *>(buf.data()) + pos, &(each_size),
                  sizeof(size_t));
      pos += sizeof(size_t);
    }

    size_t size = bytes.size();
    std::memcpy(reinterpret_cast<uint8_t *>(buf.data()) + pos, &(size),
                sizeof(size_t));
    pos += sizeof(size_t);
  }

  SPU_ENFORCE(pos == len);

  return buf;
}

vector<vector<uint8_t>> HistoGram::deserialze(yacl::Buffer &buf,
                                              size_t sameSize) {
  vector<vector<uint8_t>> bytes;
  if (sameSize == 0) {
    size_t pos = buf.size() - sizeof(size_t);
    size_t bytes_size = 0;
    std::memcpy(&(bytes_size), reinterpret_cast<uint8_t *>(buf.data()) + pos,
                sizeof(size_t));
    bytes.resize(bytes_size);

    pos = pos - bytes_size * sizeof(size_t);
    for (size_t i = 0; i < bytes.size(); i++) {
      size_t each_size = 0;
      std::memcpy(&(each_size), reinterpret_cast<uint8_t *>(buf.data()) + pos,
                  sizeof(size_t));
      pos += sizeof(size_t);
      bytes[i].resize(each_size);
    }
  } else {
    SPU_ENFORCE(buf.size() % sameSize == 0);
    bytes.resize(buf.size() / sameSize);

    for (size_t i = 0; i < bytes.size(); i++) {
      bytes[i].resize(sameSize);
    }
  }

  size_t pos = 0;
  for (size_t i = 0; i < bytes.size(); i++) {
    std::memcpy(bytes[i].data(), reinterpret_cast<uint8_t *>(buf.data()) + pos,
                bytes[i].size());
    pos += bytes[i].size();
  }

  return bytes;
}

double HistoGram::ComputeHistoGram(
    std::vector<uint64_t> &gshares, std::vector<uint64_t> &hshares,
    double scale, const std::vector<std::vector<uint8_t>> &splitBools) {
  if (use_ckks_) {
    return CKKSHistoGram(gshares, hshares, scale, splitBools);
  } else {
    SPU_THROW("only support ckks");
  }
}

double HistoGram::ComputeHistoGram(
    std::vector<std::vector<uint64_t>> &gshare_vec,
    std::vector<std::vector<uint64_t>> &hshare_vec, double scale,
    const std::vector<std::vector<uint8_t>> &split_bools) {
  SPU_ENFORCE(use_ckks_ == true);
  //        return CKKSHistoGram(gshare_vec, hshare_vec, scale, split_bools);

  double new_scale = 0.0;
  if (role_ == psi::PsiRoleType::Sender) {
    yacl::Buffer buf = link_ctx_->Recv(
        link_ctx_->NextRank(), fmt::format("recv encrypted grad and hess"));

    new_scale =
        HomCompHistoGram(gshare_vec, hshare_vec, buf, scale, split_bools);

    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf,
        fmt::format("send summed grad and hess of size {} bytes", buf.size()));

  } else {
    yacl::Buffer buf = CKKSEncrypt(gshare_vec, hshare_vec, scale);

    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf,
        fmt::format("send encrypted grad and hess of size {} bytes",
                    buf.size()));

    buf = link_ctx_->Recv(link_ctx_->NextRank(),
                          fmt::format("recv summed grad and hess"));

    size_t cols = split_bools[0].size();
    new_scale = PostProcess(gshare_vec, hshare_vec, buf, cols);
  }

  return new_scale;
}

double HistoGram::CKKSHistoGram(vector<uint64_t> &gshares,
                                vector<uint64_t> &hshares, double scale,
                                const vector<vector<uint8_t>> &splitBools) {
  SPU_ENFORCE(gshares.size() == hshares.size());
  SPU_ENFORCE(!splitBools.empty());
  size_t rows = gshares.size();
  size_t cols = splitBools[0].size();
  for (const auto &splitBool : splitBools) {
    SPU_ENFORCE(splitBool.size() == cols);
  }

  size_t lwe_coeff_count = ckks_ptr0_->slot_count() << 1U;
  size_t coeff_count = ckks_ptr1_->slot_count() << 1U;

  SPU_ENFORCE(cols <= coeff_count / 4,
              "only support number of lwes packing <= {}", coeff_count / 4);

  uint64_t lwe_mod = ckks_ptr0_->get_coeff_modulus(0);
  uint64_t rlwe_mod = ckks_ptr1_->get_coeff_modulus(0);
  uint64_t pow60 = 1ULL << 60;
  uint64_t mask60 = pow60 - 1;

  vector<uint64_t> gg(rows);
  vector<uint64_t> hh(rows);
  std::transform(gshares.begin(), gshares.end(), gg.begin(), [&](uint64_t xx) {
    uint128_t ss = static_cast<uint128_t>(lwe_mod) * (xx & mask60);
    uint64_t yy = ss >> 60;
    if ((ss & mask60) * 2 >= pow60) {
      yy = yy + 1;
    }
    return yy % lwe_mod;
  });
  std::transform(hshares.begin(), hshares.end(), hh.begin(), [&](uint64_t xx) {
    uint128_t ss = static_cast<uint128_t>(lwe_mod) * (xx & mask60);
    uint64_t yy = ss >> 60;
    if ((ss & mask60) * 2 >= pow60) {
      yy = yy + 1;
    }
    return yy % lwe_mod;
  });

  seal::Ciphertext grad_enc;
  seal::Ciphertext hess_enc;

  vector<seal::Ciphertext> enc_grad_list;
  vector<seal::Ciphertext> enc_hess_list;

  vector<vector<uint8_t>> ciphers;

  if (role_ == psi::PsiRoleType::Sender) {
    size_t enc_num = (rows + lwe_coeff_count - 1) / lwe_coeff_count;
    yacl::Buffer buf = link_ctx_->Recv(
        link_ctx_->NextRank(), fmt::format("recv encrypted grad and hess"));
    SPU_ENFORCE((buf.size() % (enc_num * 2)) == 0);

    ciphers = deserialze(buf, buf.size() / (enc_num * 2));

    enc_grad_list.resize(ciphers.size() / 2);
    enc_hess_list.resize(ciphers.size() / 2);
    for (size_t i = 0; i < enc_grad_list.size(); i++) {
      ckks_ptr0_->bytes_to_ciphertext(ciphers[i], enc_grad_list[i]);
    }
    for (size_t i = 0; i < enc_hess_list.size(); i++) {
      ckks_ptr0_->bytes_to_ciphertext(ciphers[i + enc_grad_list.size()],
                                      enc_hess_list[i]);
    }

    for (size_t i = 0; i < rows; i += lwe_coeff_count) {
      size_t n = min(lwe_coeff_count, rows - i);

      vector<uint64_t> sub_grad_vec(gg.data() + i, gg.data() + i + n);
      ckks_ptr0_->add_int_inplace(enc_grad_list[i / lwe_coeff_count],
                                  sub_grad_vec);

      vector<uint64_t> sub_hess_vec(hh.data() + i, hh.data() + i + n);
      ckks_ptr0_->add_int_inplace(enc_hess_list[i / lwe_coeff_count],
                                  sub_hess_vec);
    }

    vector<vector<int>> index_vec(cols);
    for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
        if (splitBools[i][j]) {
          index_vec[j].push_back(i);
        }
      }
    }

    vector<vector<uint64_t>> grad_lwe_coeffs;
    vector<uint64_t> grad_lwe_values;
    ckks_ptr0_->sum(enc_grad_list, index_vec, grad_lwe_coeffs, grad_lwe_values);

    ckks_ptr1_->lwes_to_rlwe(grad_lwe_coeffs, grad_lwe_values, lwe_mod,
                             enc_grad_list[0].scale(), switch_keys_, grad_enc);

    vector<vector<uint64_t>> hess_lwe_coeffs;
    vector<uint64_t> hess_lwe_values;
    ckks_ptr0_->sum(enc_hess_list, index_vec, hess_lwe_coeffs, hess_lwe_values);

    ckks_ptr1_->lwes_to_rlwe(hess_lwe_coeffs, hess_lwe_values, lwe_mod,
                             enc_hess_list[0].scale(), switch_keys_, hess_enc);

    gshares.resize(cols);
    hshares.resize(cols);

    auto bytes = yacl::crypto::SecureRandBytes(cols * 2 * sizeof(uint64_t));
    std::memcpy(reinterpret_cast<uint8_t *>(gshares.data()), bytes.data(),
                cols * sizeof(uint64_t));
    std::memcpy(reinterpret_cast<uint8_t *>(hshares.data()),
                bytes.data() + cols * sizeof(uint64_t),
                cols * sizeof(uint64_t));

    for (auto &vv : gshares) {
      vv = vv % rlwe_mod;
    }
    for (auto &vv : hshares) {
      vv = vv % rlwe_mod;
    }

    vector<uint64_t> grad_mask;
    ckks_ptr1_->expand(gshares, grad_mask);
    ckks_ptr1_->sub_int_inplace(grad_enc, grad_mask);

    vector<uint64_t> hess_mask;
    ckks_ptr1_->expand(hshares, hess_mask);
    ckks_ptr1_->sub_int_inplace(hess_enc, hess_mask);

    ciphers.resize(2);
    ckks_ptr1_->ciphertext_to_bytes(grad_enc, ciphers[0]);
    ckks_ptr1_->ciphertext_to_bytes(hess_enc, ciphers[1]);

    buf = serialize(ciphers, true);
    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf,
        fmt::format("send summed grad and hess of size {} bytes", buf.size()));

  } else {
    double new_scale = (scale * lwe_mod) / pow60;
    ;

    for (size_t i = 0; i < rows; i += lwe_coeff_count) {
      size_t n = min(lwe_coeff_count, rows - i);

      seal::Ciphertext enc_grad_tmp;
      vector<uint64_t> sub_grad_vec(gg.data() + i, gg.data() + i + n);

      ckks_ptr0_->encrypt_big_int(sub_grad_vec, enc_grad_tmp);
      enc_grad_tmp.scale() = new_scale;

      enc_grad_list.push_back(enc_grad_tmp);

      seal::Ciphertext enc_hess_tmp;
      vector<uint64_t> sub_hess_vec(hh.data() + i, hh.data() + i + n);

      ckks_ptr0_->encrypt_big_int(sub_hess_vec, enc_hess_tmp);
      enc_hess_tmp.scale() = new_scale;

      enc_hess_list.push_back(enc_hess_tmp);
    }

    ciphers.resize(enc_grad_list.size() + enc_hess_list.size());
    for (size_t i = 0; i < enc_grad_list.size(); i++) {
      ckks_ptr0_->ciphertext_to_bytes(enc_grad_list[i], ciphers[i]);
    }
    for (size_t i = 0; i < enc_hess_list.size(); i++) {
      ckks_ptr0_->ciphertext_to_bytes(enc_hess_list[i],
                                      ciphers[i + enc_grad_list.size()]);
    }

    yacl::Buffer buf = serialize(ciphers, true);
    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf,
        fmt::format("send encrypted grad and hess of size {} bytes",
                    buf.size()));

    buf = link_ctx_->Recv(link_ctx_->NextRank(),
                          fmt::format("recv summed grad and hess"));
    SPU_ENFORCE((buf.size() % 2) == 0);
    ciphers = deserialze(buf, buf.size() / 2);

    ckks_ptr1_->bytes_to_ciphertext(ciphers[0], grad_enc);
    ckks_ptr1_->bytes_to_ciphertext(ciphers[1], hess_enc);

    ckks_ptr1_->decrypt_aggregate_lwes(grad_enc, gshares, cols);
    ckks_ptr1_->decrypt_aggregate_lwes(hess_enc, hshares, cols);
  }

  double pow64 = std::pow(2, 64);
  for (size_t j = 0; j < cols; j++) {
    gshares[j] = (pow64 * gshares[j]) / rlwe_mod;
    hshares[j] = (pow64 * hshares[j]) / rlwe_mod;
  }

  return grad_enc.scale() * pow64 / rlwe_mod;
}

std::vector<uint64_t> HistoGram::uint64_to_lwe_mod(const vector<uint64_t> &value, uint64_t lwe_mod)
{
  uint64_t pow60 = 1ULL << 60;
  uint64_t mask60 = pow60 - 1;
  std::vector<uint64_t> res(value.size());
  std::transform(value.begin(), value.end(), res.begin(), [&](uint64_t vv) {
    uint128_t ss = static_cast<uint128_t>(lwe_mod) * (vv & mask60);
    uint64_t tt = ss >> 60;
    if ((ss & mask60) * 2 >= pow60) {
      tt = tt + 1;
    }
    return tt % lwe_mod;
  });
  return res;
};

yacl::Buffer HistoGram::CKKSEncrypt(
    const std::vector<std::vector<uint64_t>> &gshare_vec,
    const std::vector<std::vector<uint64_t>> &hshare_vec, double scale) {
  SPU_ENFORCE(gshare_vec.size() == hshare_vec.size());
  size_t count = gshare_vec.size();
  size_t rows = gshare_vec[0].size();
  for (size_t i = 0; i < count; i++) {
    SPU_ENFORCE(gshare_vec[i].size() == rows && hshare_vec[i].size() == rows);
  }

  SPU_ENFORCE(role_ == psi::PsiRoleType::Receiver,
              "only receiver executes CKKSEncrypt");

  size_t lwe_coeff_count = ckks_ptr0_->slot_count() << 1U;
  uint64_t lwe_mod = ckks_ptr0_->get_coeff_modulus(0);

  uint64_t pow60 = 1ULL << 60;

  double new_scale = (scale * lwe_mod) / pow60;

  vector<vector<uint8_t>> cipher_stream;
  for (size_t cnt = 0; cnt < count; cnt++) {
    vector<uint64_t> gg = uint64_to_lwe_mod(gshare_vec[cnt], lwe_mod);
    vector<uint64_t> hh = uint64_to_lwe_mod(hshare_vec[cnt], lwe_mod);

    vector<uint8_t> cipher;
    // encrypt
    for (size_t i = 0; i < rows; i += lwe_coeff_count) {
      size_t n = min(lwe_coeff_count, rows - i);

      seal::Ciphertext enc_grad_tmp;
      vector<uint64_t> sub_grad_vec(gg.data() + i, gg.data() + i + n);

      ckks_ptr0_->encrypt_big_int(sub_grad_vec, enc_grad_tmp);
      enc_grad_tmp.scale() = new_scale;

      seal::Ciphertext enc_hess_tmp;
      vector<uint64_t> sub_hess_vec(hh.data() + i, hh.data() + i + n);

      ckks_ptr0_->encrypt_big_int(sub_hess_vec, enc_hess_tmp);
      enc_hess_tmp.scale() = new_scale;

      ckks_ptr0_->ciphertext_to_bytes(enc_grad_tmp, cipher);
      cipher_stream.push_back(cipher);

      ckks_ptr0_->ciphertext_to_bytes(enc_hess_tmp, cipher);
      cipher_stream.push_back(cipher);
    }
  }

  return serialize(cipher_stream, true);
}

double HistoGram::HomCompHistoGram(
    std::vector<std::vector<uint64_t>> &gshare_vec,
    std::vector<std::vector<uint64_t>> &hshare_vec, yacl::Buffer &buf,
    double scale, const std::vector<std::vector<uint8_t>> &split_bools) {
  SPU_ENFORCE(gshare_vec.size() == hshare_vec.size());
  SPU_ENFORCE(role_ == psi::PsiRoleType::Sender, "only sender execute");

  SPU_ENFORCE(!split_bools.empty());
  size_t rows = split_bools.size();
  size_t cols = split_bools[0].size();
  for (const auto &split_bool : split_bools) {
    SPU_ENFORCE(split_bool.size() == cols);
  }
  for (const auto &gshare : gshare_vec) {
    SPU_ENFORCE(gshare.size() == rows);
  }
  for (const auto &hshare : hshare_vec) {
    SPU_ENFORCE(hshare.size() == rows);
  }

  size_t lwe_coeff_count = ckks_ptr0_->slot_count() << 1U;
  size_t coeff_count = ckks_ptr1_->slot_count() << 1U;
  size_t enc_num = (rows + lwe_coeff_count - 1) / lwe_coeff_count;
  size_t count = gshare_vec.size();

  SPU_ENFORCE(cols <= coeff_count / 4,
              "only support number of lwes packing <= {}", coeff_count / 4);
  SPU_ENFORCE((buf.size() % (enc_num * 2 * count)) == 0);
  vector<vector<uint8_t>> cipher_stream =
      deserialze(buf, buf.size() / (enc_num * 2 * count));

  uint64_t lwe_mod = ckks_ptr0_->get_coeff_modulus(0);
  uint64_t rlwe_mod = ckks_ptr1_->get_coeff_modulus(0);
  uint64_t pow60 = 1ULL << 60;

  vector<vector<int>> index_vec(cols);
  for (size_t j = 0; j < cols; j++) {
    for (size_t i = 0; i < rows; i++) {
      if (split_bools[i][j] != 0u) {
        index_vec[j].push_back(i);
      }
    }
  }

  double new_scale = (scale * lwe_mod) / pow60;

  vector<vector<uint8_t>> cipher_stream_other;
  for (size_t cnt = 0; cnt < count; cnt++) {
    seal::Ciphertext grad_enc;
    seal::Ciphertext hess_enc;

    vector<seal::Ciphertext> enc_grad_list;
    vector<seal::Ciphertext> enc_hess_list;

    size_t offset = cnt * (2 * enc_num);
    for (size_t i = 0; i < enc_num; i++) {
      ckks_ptr0_->bytes_to_ciphertext(cipher_stream[offset + 2 * i], grad_enc);
      ckks_ptr0_->bytes_to_ciphertext(cipher_stream[offset + 2 * i + 1],
                                      hess_enc);

      enc_grad_list.push_back(grad_enc);
      enc_hess_list.push_back(hess_enc);
    }

    vector<uint64_t> gg = uint64_to_lwe_mod(gshare_vec[cnt], lwe_mod);
    vector<uint64_t> hh = uint64_to_lwe_mod(hshare_vec[cnt], lwe_mod);

    for (size_t i = 0; i < rows; i += lwe_coeff_count) {
      size_t n = min(lwe_coeff_count, rows - i);

      vector<uint64_t> sub_grad_vec(gg.data() + i, gg.data() + i + n);
      ckks_ptr0_->add_int_inplace(enc_grad_list[i / lwe_coeff_count],
                                  sub_grad_vec);

      vector<uint64_t> sub_hess_vec(hh.data() + i, hh.data() + i + n);
      ckks_ptr0_->add_int_inplace(enc_hess_list[i / lwe_coeff_count],
                                  sub_hess_vec);
    }

    vector<vector<uint64_t>> grad_lwe_coeffs;
    vector<uint64_t> grad_lwe_values;
    ckks_ptr0_->sum(enc_grad_list, index_vec, grad_lwe_coeffs, grad_lwe_values);

    ckks_ptr1_->lwes_to_rlwe(grad_lwe_coeffs, grad_lwe_values, lwe_mod,
                             enc_grad_list[0].scale(), switch_keys_, grad_enc);

    vector<vector<uint64_t>> hess_lwe_coeffs;
    vector<uint64_t> hess_lwe_values;
    ckks_ptr0_->sum(enc_hess_list, index_vec, hess_lwe_coeffs, hess_lwe_values);

    //            auto time_start = std::chrono::high_resolution_clock::now();
    ckks_ptr1_->lwes_to_rlwe(hess_lwe_coeffs, hess_lwe_values, lwe_mod,
                             enc_hess_list[0].scale(), switch_keys_, hess_enc);
    //            auto time_end = std::chrono::high_resolution_clock::now();
    //            auto time_diff =
    //            std::chrono::duration_cast<std::chrono::microseconds>(time_end
    //            - time_start); std::cout << hess_lwe_values.size() << " lwes
    //            to rlwes takes " << time_diff.count() << " us" << std::endl;

    gshare_vec[cnt].resize(cols);
    hshare_vec[cnt].resize(cols);

    auto bytes = yacl::crypto::SecureRandBytes(cols * 2 * sizeof(uint64_t));
    std::memcpy(reinterpret_cast<uint8_t *>(gshare_vec[cnt].data()),
                bytes.data(), cols * sizeof(uint64_t));
    std::memcpy(reinterpret_cast<uint8_t *>(hshare_vec[cnt].data()),
                bytes.data() + cols * sizeof(uint64_t),
                cols * sizeof(uint64_t));

    for (auto &vv : gshare_vec[cnt]) {
      vv = vv % rlwe_mod;
    }
    for (auto &vv : hshare_vec[cnt]) {
      vv = vv % rlwe_mod;
    }

    vector<uint64_t> grad_mask;
    ckks_ptr1_->expand(gshare_vec[cnt], grad_mask);
    ckks_ptr1_->sub_int_inplace(grad_enc, grad_mask);

    vector<uint64_t> hess_mask;
    ckks_ptr1_->expand(hshare_vec[cnt], hess_mask);
    ckks_ptr1_->sub_int_inplace(hess_enc, hess_mask);

    vector<uint8_t> cipher;
    ckks_ptr1_->ciphertext_to_bytes(grad_enc, cipher);
    cipher_stream_other.push_back(cipher);
    ckks_ptr1_->ciphertext_to_bytes(hess_enc, cipher);
    cipher_stream_other.push_back(cipher);

    new_scale = grad_enc.scale();
  }

  buf = serialize(cipher_stream_other, true);

  double pow64 = std::pow(2, 64);
  for (size_t cnt = 0; cnt < count; cnt++) {
    for (size_t j = 0; j < cols; j++) {
      gshare_vec[cnt][j] = (pow64 * gshare_vec[cnt][j]) / rlwe_mod;
      hshare_vec[cnt][j] = (pow64 * hshare_vec[cnt][j]) / rlwe_mod;
    }
  }

  return new_scale * pow64 / rlwe_mod;
}

double HistoGram::PostProcess(std::vector<std::vector<uint64_t>> &gshare_vec,
                              std::vector<std::vector<uint64_t>> &hshare_vec,
                              yacl::Buffer &buf, size_t cols) {
  size_t count = gshare_vec.size();
  SPU_ENFORCE(gshare_vec.size() == hshare_vec.size());
  SPU_ENFORCE((buf.size() % (2 * count)) == 0);

  uint64_t rlwe_mod = ckks_ptr1_->get_coeff_modulus(0);
  auto cipher_stream = deserialze(buf, buf.size() / (2 * count));

  double new_scale = 0.0;
  for (size_t cnt = 0; cnt < count; cnt++) {
    seal::Ciphertext grad_enc;
    seal::Ciphertext hess_enc;

    ckks_ptr1_->bytes_to_ciphertext(cipher_stream[2 * cnt + 0], grad_enc);
    ckks_ptr1_->bytes_to_ciphertext(cipher_stream[2 * cnt + 1], hess_enc);

    ckks_ptr1_->decrypt_aggregate_lwes(grad_enc, gshare_vec[cnt], cols);
    ckks_ptr1_->decrypt_aggregate_lwes(hess_enc, hshare_vec[cnt], cols);

    new_scale = grad_enc.scale();
  }

  double pow64 = std::pow(2, 64);
  for (size_t cnt = 0; cnt < count; cnt++) {
    for (size_t j = 0; j < cols; j++) {
      gshare_vec[cnt][j] = (pow64 * gshare_vec[cnt][j]) / rlwe_mod;
      hshare_vec[cnt][j] = (pow64 * hshare_vec[cnt][j]) / rlwe_mod;
    }
  }

  return new_scale * pow64 / rlwe_mod;
}

double HistoGram::CKKSHistoGram(
    std::vector<std::vector<uint64_t>> &gshare_vec,
    std::vector<std::vector<uint64_t>> &hshare_vec, double scale,
    const std::vector<std::vector<uint8_t>> &split_bools) {
  SPU_ENFORCE(gshare_vec.size() == hshare_vec.size());
  size_t count = gshare_vec.size();

  SPU_ENFORCE(!split_bools.empty());
  size_t rows = split_bools.size();
  size_t cols = split_bools[0].size();
  for (const auto &split_bool : split_bools) {
    SPU_ENFORCE(split_bool.size() == cols);
  }
  for (const auto &gshare : gshare_vec) {
    SPU_ENFORCE(gshare.size() == rows);
  }
  for (const auto &hshare : hshare_vec) {
    SPU_ENFORCE(hshare.size() == rows);
  }

  size_t lwe_coeff_count = ckks_ptr0_->slot_count() << 1U;
  size_t coeff_count = ckks_ptr1_->slot_count() << 1U;

  SPU_ENFORCE(cols <= coeff_count / 4,
              "only support number of lwes packing <= {}", coeff_count / 4);

  uint64_t lwe_mod = ckks_ptr0_->get_coeff_modulus(0);
  uint64_t rlwe_mod = ckks_ptr1_->get_coeff_modulus(0);
  uint64_t pow60 = 1ULL << 60;

  double new_scale = (scale * lwe_mod) / pow60;

  if (role_ == psi::PsiRoleType::Receiver) {
    auto buf = CKKSEncrypt(gshare_vec, hshare_vec, scale);

    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf,
        fmt::format("send encrypted grad and hess of size {} bytes",
                    buf.size()));

    buf = link_ctx_->Recv(link_ctx_->NextRank(),
                          fmt::format("recv summed grad and hess"));

    return PostProcess(gshare_vec, hshare_vec, buf, cols);

  } else {
    vector<vector<int>> index_vec(cols);
    for (size_t j = 0; j < cols; j++) {
      for (size_t i = 0; i < rows; i++) {
        if (split_bools[i][j]) {
          index_vec[j].push_back(i);
        }
      }
    }

    size_t enc_num = (rows + lwe_coeff_count - 1) / lwe_coeff_count;
    yacl::Buffer buf = link_ctx_->Recv(
        link_ctx_->NextRank(), fmt::format("recv encrypted grad and hess"));
    SPU_ENFORCE((buf.size() % (enc_num * 2 * count)) == 0);

    vector<vector<uint8_t>> cipher_stream =
        deserialze(buf, buf.size() / (enc_num * 2 * count));

    vector<vector<uint8_t>> cipher_stream_other;
    for (size_t cnt = 0; cnt < count; cnt++) {
      seal::Ciphertext grad_enc;
      seal::Ciphertext hess_enc;

      vector<seal::Ciphertext> enc_grad_list;
      vector<seal::Ciphertext> enc_hess_list;

      size_t offset = cnt * (2 * enc_num);
      for (size_t i = 0; i < enc_num; i++) {
        ckks_ptr0_->bytes_to_ciphertext(cipher_stream[offset + 2 * i],
                                        grad_enc);
        ckks_ptr0_->bytes_to_ciphertext(cipher_stream[offset + 2 * i + 1],
                                        hess_enc);

        enc_hess_list.push_back(hess_enc);
        enc_grad_list.push_back(grad_enc);
      }

      vector<uint64_t> gg = uint64_to_lwe_mod(gshare_vec[cnt], lwe_mod);
      vector<uint64_t> hh = uint64_to_lwe_mod(hshare_vec[cnt], lwe_mod);

      for (size_t i = 0; i < rows; i += lwe_coeff_count) {
        size_t n = min(lwe_coeff_count, rows - i);

        vector<uint64_t> sub_grad_vec(gg.data() + i, gg.data() + i + n);
        ckks_ptr0_->add_int_inplace(enc_grad_list[i / lwe_coeff_count],
                                    sub_grad_vec);

        vector<uint64_t> sub_hess_vec(hh.data() + i, hh.data() + i + n);
        ckks_ptr0_->add_int_inplace(enc_hess_list[i / lwe_coeff_count],
                                    sub_hess_vec);
      }

      vector<vector<uint64_t>> grad_lwe_coeffs;
      vector<uint64_t> grad_lwe_values;
      ckks_ptr0_->sum(enc_grad_list, index_vec, grad_lwe_coeffs,
                      grad_lwe_values);

      ckks_ptr1_->lwes_to_rlwe(grad_lwe_coeffs, grad_lwe_values, lwe_mod,
                               enc_grad_list[0].scale(), switch_keys_,
                               grad_enc);

      vector<vector<uint64_t>> hess_lwe_coeffs;
      vector<uint64_t> hess_lwe_values;
      ckks_ptr0_->sum(enc_hess_list, index_vec, hess_lwe_coeffs,
                      hess_lwe_values);

      ckks_ptr1_->lwes_to_rlwe(hess_lwe_coeffs, hess_lwe_values, lwe_mod,
                               enc_hess_list[0].scale(), switch_keys_,
                               hess_enc);

      gshare_vec[cnt].resize(cols);
      hshare_vec[cnt].resize(cols);

      auto bytes = yacl::crypto::SecureRandBytes(cols * 2 * sizeof(uint64_t));
      std::memcpy(reinterpret_cast<uint8_t *>(gshare_vec[cnt].data()),
                  bytes.data(), cols * sizeof(uint64_t));
      std::memcpy(reinterpret_cast<uint8_t *>(hshare_vec[cnt].data()),
                  bytes.data() + cols * sizeof(uint64_t),
                  cols * sizeof(uint64_t));

      for (auto &vv : gshare_vec[cnt]) {
        vv = vv % rlwe_mod;
      }
      for (auto &vv : hshare_vec[cnt]) {
        vv = vv % rlwe_mod;
      }

      vector<uint64_t> grad_mask;
      ckks_ptr1_->expand(gshare_vec[cnt], grad_mask);
      ckks_ptr1_->sub_int_inplace(grad_enc, grad_mask);

      vector<uint64_t> hess_mask;
      ckks_ptr1_->expand(hshare_vec[cnt], hess_mask);
      ckks_ptr1_->sub_int_inplace(hess_enc, hess_mask);

      vector<uint8_t> cipher;
      ckks_ptr1_->ciphertext_to_bytes(grad_enc, cipher);
      cipher_stream_other.push_back(cipher);
      ckks_ptr1_->ciphertext_to_bytes(hess_enc, cipher);
      cipher_stream_other.push_back(cipher);

      new_scale = grad_enc.scale();
    }

    buf = serialize(cipher_stream_other, true);
    link_ctx_->SendAsyncThrottled(
        link_ctx_->NextRank(), buf,
        fmt::format("send summed grad and hess of size {} bytes", buf.size()));
  }

  double pow64 = std::pow(2, 64);
  for (size_t cnt = 0; cnt < count; cnt++) {
    for (size_t j = 0; j < cols; j++) {
      gshare_vec[cnt][j] = (pow64 * gshare_vec[cnt][j]) / rlwe_mod;
      hshare_vec[cnt][j] = (pow64 * hshare_vec[cnt][j]) / rlwe_mod;
    }
  }

  return new_scale * pow64 / rlwe_mod;
}

//    double HistoGram::CKKSHistoGram(std::vector<std::vector<uint64_t>>
//    &gshare_vec, std::vector<std::vector<uint64_t>> &hshare_vec, double scale,
//                         const std::vector<std::vector<uint8_t>> &split_bools)
//                         {
//        SPU_ENFORCE(gshare_vec.size() == hshare_vec.size());
//        size_t count = gshare_vec.size();
//
//        SPU_ENFORCE(!split_bools.empty());
//        size_t rows = split_bools.size();
//        size_t cols = split_bools[0].size();
//        for (const auto & split_bool : split_bools) {
//            SPU_ENFORCE(split_bool.size() == cols);
//        }
//        for (const auto & gshare : gshare_vec) {
//            SPU_ENFORCE(gshare.size() == rows);
//        }
//        for (const auto & hshare : hshare_vec) {
//            SPU_ENFORCE(hshare.size() == rows);
//        }
//
//        size_t lwe_coeff_count = ckks_ptr0_->slot_count() << 1U;
//        size_t coeff_count = ckks_ptr1_->slot_count() << 1U;
//
//        SPU_ENFORCE(cols <= coeff_count / 4, "only support number of lwes
//        packing <= {}", coeff_count / 4);
//
//        uint64_t lwe_mod = ckks_ptr0_->get_coeff_modulus(0);
//        uint64_t rlwe_mod = ckks_ptr1_->get_coeff_modulus(0);
//        uint64_t pow60 = 1ULL << 60;
//        uint64_t mask60 = pow60 - 1;
//
//        double new_scale = (scale * lwe_mod) / pow60;
//
//        if (role_ == psi::PsiRoleType::Receiver) {
//            vector<vector<uint8_t>> cipher_stream;
//            for (size_t cnt = 0; cnt < count; cnt++) {
//                vector<uint64_t> gg = uint64_to_lwe_mod(gshare_vec[cnt], lwe_mod);
//                vector<uint64_t> hh = uint64_to_lwe_mod(hshare_vec[cnt], lwe_mod);
//
//                seal::Ciphertext grad_enc;
//                seal::Ciphertext hess_enc;
//
//                vector<seal::Ciphertext> enc_grad_list;
//                vector<seal::Ciphertext> enc_hess_list;
//
//                // encrypt
//                for (size_t i = 0; i < rows; i += lwe_coeff_count) {
//                    size_t n = min(lwe_coeff_count, rows - i);
//
//                    seal::Ciphertext enc_grad_tmp;
//                    vector<uint64_t> sub_grad_vec(gg.data() + i, gg.data() + i
//                    + n);
//
//                    ckks_ptr0_->encrypt_big_int(sub_grad_vec, enc_grad_tmp);
//                    enc_grad_tmp.scale() = new_scale;
//
//                    enc_grad_list.push_back(enc_grad_tmp);
//
//                    seal::Ciphertext enc_hess_tmp;
//                    vector<uint64_t> sub_hess_vec(hh.data() + i, hh.data() + i
//                    + n);
//
//                    ckks_ptr0_->encrypt_big_int(sub_hess_vec, enc_hess_tmp);
//                    enc_hess_tmp.scale() = new_scale;
//
//                    enc_hess_list.push_back(enc_hess_tmp);
//                }
//
//                for (const auto & enc_grad : enc_grad_list) {
//                    vector<uint8_t> cipher;
//                    ckks_ptr0_->ciphertext_to_bytes(enc_grad, cipher);
//                    cipher_stream.push_back(cipher);
//                }
//                for (const auto & enc_hess : enc_hess_list) {
//                    vector<uint8_t> cipher;
//                    ckks_ptr0_->ciphertext_to_bytes(enc_hess, cipher);
//                    cipher_stream.push_back(cipher);
//                }
//            }
//
//            yacl::Buffer buf = serialize(cipher_stream, true);
//            link_ctx_->SendAsyncThrottled(
//                    link_ctx_->NextRank(), buf,
//                    fmt::format("send encrypted grad and hess of size {}
//                    bytes", buf.size()));
//
//            for (size_t cnt = 0; cnt < count; cnt++) {
//                buf = link_ctx_->Recv(
//                        link_ctx_->NextRank(), fmt::format("recv summed grad
//                        and hess"));
//                SPU_ENFORCE((buf.size() % 2) == 0);
//
//                cipher_stream = deserialze(buf, buf.size() / 2);
//
//                seal::Ciphertext grad_enc;
//                seal::Ciphertext hess_enc;
//
//                ckks_ptr1_->bytes_to_ciphertext(cipher_stream[0], grad_enc);
//                ckks_ptr1_->bytes_to_ciphertext(cipher_stream[1], hess_enc);
//
//                ckks_ptr1_->decrypt_aggregate_lwes(grad_enc, gshare_vec[cnt],
//                cols); ckks_ptr1_->decrypt_aggregate_lwes(hess_enc,
//                hshare_vec[cnt], cols);
//
//                new_scale = grad_enc.scale();
//            }
//
//        } else {
//            vector<vector<int>> index_vec(cols);
//            for (size_t j = 0; j < cols; j++) {
//                for (size_t i = 0; i < rows; i++) {
//                    if (split_bools[i][j]) {
//                        index_vec[j].push_back(i);
//                    }
//                }
//            }
//
//            size_t enc_num = (rows + lwe_coeff_count - 1) / lwe_coeff_count;
//            yacl::Buffer buf = link_ctx_->Recv(
//                    link_ctx_->NextRank(), fmt::format("recv encrypted grad
//                    and hess"));
//            SPU_ENFORCE((buf.size() % (enc_num * 2 * count)) == 0);
//
//            vector<vector<uint8_t>> cipher_stream = deserialze(buf, buf.size()
//            / (enc_num * 2 * count));
//
//            for (size_t cnt = 0; cnt < count; cnt++) {
//                seal::Ciphertext grad_enc;
//                seal::Ciphertext hess_enc;
//
//                vector<seal::Ciphertext> enc_grad_list;
//                vector<seal::Ciphertext> enc_hess_list;
//
//                size_t offset = cnt * (2 * enc_num);
//                for (size_t i = 0; i < enc_num; i++) {
//                    ckks_ptr0_->bytes_to_ciphertext(cipher_stream[offset + i],
//                    grad_enc); enc_grad_list.push_back(grad_enc);
//                }
//                for (size_t i = 0; i < enc_num; i++) {
//                    ckks_ptr0_->bytes_to_ciphertext(cipher_stream[offset +
//                    enc_num + i], hess_enc);
//                    enc_hess_list.push_back(hess_enc);
//                }
//
//                vector<uint64_t> gg = uint64_to_lwe_mod(gshare_vec[cnt], lwe_mod);
//                vector<uint64_t> hh = uint64_to_lwe_mod(hshare_vec[cnt], lwe_mod);
//
//                for (size_t i = 0; i < rows; i += lwe_coeff_count) {
//                    size_t n = min(lwe_coeff_count, rows - i);
//
//                    vector<uint64_t> sub_grad_vec(gg.data() + i, gg.data() + i
//                    + n); ckks_ptr0_->add_int_inplace(enc_grad_list[i /
//                    lwe_coeff_count], sub_grad_vec);
//
//                    vector<uint64_t> sub_hess_vec(hh.data() + i, hh.data() + i
//                    + n); ckks_ptr0_->add_int_inplace(enc_hess_list[i /
//                    lwe_coeff_count], sub_hess_vec);
//                }
//
//                vector<vector<uint64_t>> grad_lwe_coeffs;
//                vector<uint64_t> grad_lwe_values;
//                ckks_ptr0_->sum(enc_grad_list, index_vec, grad_lwe_coeffs,
//                grad_lwe_values);
//
//                ckks_ptr1_->lwes_to_rlwe(grad_lwe_coeffs, grad_lwe_values,
//                lwe_mod, enc_grad_list[0].scale(),
//                                         switch_keys_, grad_enc);
//
//                vector<vector<uint64_t>> hess_lwe_coeffs;
//                vector<uint64_t> hess_lwe_values;
//                ckks_ptr0_->sum(enc_hess_list, index_vec, hess_lwe_coeffs,
//                hess_lwe_values);
//
//                ckks_ptr1_->lwes_to_rlwe(hess_lwe_coeffs, hess_lwe_values,
//                lwe_mod, enc_hess_list[0].scale(),
//                                         switch_keys_, hess_enc);
//
//                gshare_vec[cnt].resize(cols);
//                hshare_vec[cnt].resize(cols);
//                for (auto &vv : gshare_vec[cnt]) {
//                    vv = seal::random_uint64() % rlwe_mod;
//                }
//                for (auto &vv : hshare_vec[cnt]) {
//                    vv = seal::random_uint64() % rlwe_mod;
//                }
//
//                vector<uint64_t> grad_mask;
//                ckks_ptr1_->expand(gshare_vec[cnt], grad_mask);
//                ckks_ptr1_->sub_int_inplace(grad_enc, grad_mask);
//
//                vector<uint64_t> hess_mask;
//                ckks_ptr1_->expand(hshare_vec[cnt], hess_mask);
//                ckks_ptr1_->sub_int_inplace(hess_enc, hess_mask);
//
//                vector<vector<uint8_t>> cipher_stream_other;
//                vector<uint8_t> cipher;
//                ckks_ptr1_->ciphertext_to_bytes(grad_enc, cipher);
//                cipher_stream_other.push_back(cipher);
//                ckks_ptr1_->ciphertext_to_bytes(hess_enc, cipher);
//                cipher_stream_other.push_back(cipher);
//
//                new_scale = grad_enc.scale();
//
//                buf = serialize(cipher_stream_other, true);
//                link_ctx_->SendAsyncThrottled(
//                        link_ctx_->NextRank(), buf,
//                        fmt::format("send summed grad and hess of size {}
//                        bytes", buf.size()));
//            }
//        }
//
//        double pow64 = std::pow(2, 64);
//        for (size_t cnt = 0; cnt < count; cnt++) {
//            for (size_t j = 0; j < cols; j++) {
//                gshare_vec[cnt][j] = (pow64 * gshare_vec[cnt][j]) / rlwe_mod;
//                hshare_vec[cnt][j] = (pow64 * hshare_vec[cnt][j]) / rlwe_mod;
//            }
//        }
//
//        return new_scale * pow64 / rlwe_mod;
//    }

void HistoGram::getSplitBools(const vector<vector<double>> &features,
                              size_t numBuckets,
                              vector<vector<uint8_t>> &splitBools,
                              vector<vector<double>> &splitPoints) {
  SPU_ENFORCE(!features.empty(), "input feature data should not be empty");
  SPU_ENFORCE(numBuckets > 1, "bucket number should be > 1");

  size_t rows = features.size();
  size_t cols = features[0].size();
  for (size_t i = 0; i < rows; i++) {
    SPU_ENFORCE(features[i].size() == cols);
  }

  splitBools.resize(cols * numBuckets);
  for (size_t j = 0; j < splitBools.size(); j++) {
    splitPoints[j].assign(rows, 0);
  }

  if (splitPoints.empty()) {
    splitPoints.resize(cols);
    for (size_t j = 0; j < cols; j++) {
      vector<double> feat_col(rows);
      for (size_t i = 0; i < rows; i++) {
        feat_col[i] = features[i][j];
      }
      std::sort(feat_col.begin(), feat_col.end());

      size_t step = (rows + numBuckets - 1) / numBuckets;
      splitPoints[j].resize(numBuckets - 1);
      for (size_t i = 0; i < numBuckets - 1; i++) {
        size_t pos = (i + 1) * step;
        splitPoints[j][i] = (feat_col[pos - 1] + feat_col[pos + 1]) / 2.0;
      }
    }
  } else {
    SPU_ENFORCE(splitPoints.size() == cols);
    for (size_t j = 0; j < cols; j++) {
      SPU_ENFORCE(splitPoints[j].size() == (numBuckets - 1));
      std::sort(splitPoints[j].begin(), splitPoints[j].end());
    }
  }

  for (size_t j = 0; j < cols; j++) {
    for (size_t i = 0; i < rows; i++) {
      auto iter = std::find(splitPoints[j].begin(), splitPoints[j].end(),
                            features[i][j]);
      splitBools[j * numBuckets + iter - splitPoints[j].begin()][i] = 1;
    }
  }
}

//    void HistoGram::multiplex(std::vector<uint64_t> &gshares,
//    std::vector<uint64_t> &hshares, const std::vector<uint8_t> &masks) {
//        SPU_ENFORCE(gshares.size() == hshares.size() && gshares.size() ==
//        masks.size(),
//                    "input length should be equal");
//
//        for (size_t i = 0; i < masks.size(); i++) {
//            SPU_ENFORCE(masks[i] == 1 || masks[i] == 0, "invalid select
//            bits");
//        }
//
//        size_t size = gshares.size();
//        vector<uint128_t> rand(size);
//        yacl::crypto::FastRandU128(absl::MakeSpan(rand), true);
//
//        const auto mask64 = static_cast<uint64_t>(-1);
//
//        // Compute (b0 ^ b1) ? (x0 + x1) : 0
//        vector<uint128_t> sent2(size * 2);
//        pforeach(0, size, [&](int64_t i) {
//
//            uint64_t high0 = (gshares[i] * masks[i]) - (rand[i] >> 64U);
//            uint64_t low0  = (hshares[i] * masks[i]) - (rand[i] & mask64);
//            sent2[i * 2] = (static_cast<uint128_t>(high0) << 64U) | low0;
//
//            uint64_t high1 = (gshares[i] * (masks[i] ^ 1)) - (rand[i] >> 64U);
//            uint64_t low1  = (hshares[i] * (masks[i] ^ 1)) - (rand[i] &
//            mask64); sent2[i * 2 + 1] = (static_cast<uint128_t>(high1) << 64U)
//            | low1;
//        });
//
//        vector<uint128_t> recv(size);
//
//        auto sender = base_ot_->GetSenderCOT();
//        auto receiver = base_ot_->GetReceiverCOT();
//
//        if (role_ == psi::PsiRoleType::Sender) {
//            sender->SendCMCC(absl::MakeSpan(sent2), 2);
//            sender->Flush();
//
//            receiver->RecvCMCC(absl::MakeSpan(masks), 2,
//            absl::MakeSpan(recv));
//        } else {
//            receiver->RecvCMCC(absl::MakeSpan(masks), 2,
//            absl::MakeSpan(recv));
//
//            sender->SendCMCC(absl::MakeSpan(sent2), 2);
//            sender->Flush();
//        }
//
//        pforeach(0, size, [&](int64_t i) {
//            gshares[i] = (rand[i] >> 64U) + (recv[i] >> 64U);
//            hshares[i] = (rand[i] + recv[i]) & mask64;
//        });
//    }
//
//    void HistoGram::multiplex(std::vector<uint64_t> &shares, const
//    std::vector<uint8_t> &masks) {
//        SPU_ENFORCE(shares.size() == masks.size(), "input length should be
//        equal");
//
//        for (size_t i = 0; i < masks.size(); i++) {
//            SPU_ENFORCE(masks[i] == 1 || masks[i] == 0, "invalid select
//            bits");
//        }
//
//        size_t size = shares.size();
//        auto rand = yacl::crypto::RandVec<uint64_t>(size, true);
//
//        // Compute (b0 ^ b1) ? (x0 + x1) : 0
//        vector<uint64_t> sent2(size * 2);
//        pforeach(0, size, [&](int64_t i) {
//            sent2[i * 2] = (shares[i] * masks[i]) - rand[i];
//            sent2[i * 2 + 1] = (shares[i] * (masks[i] ^ 1)) - rand[i];
//        });
//
//        vector<uint64_t> recv(size);
//
//        auto sender = base_ot_->GetSenderCOT();
//        auto receiver = base_ot_->GetReceiverCOT();
//
//        if (role_ == psi::PsiRoleType::Sender) {
//            sender->SendCMCC(absl::MakeSpan(sent2), 2);
//            sender->Flush();
//
//            receiver->RecvCMCC(absl::MakeSpan(masks), 2,
//            absl::MakeSpan(recv));
//        } else {
//            receiver->RecvCMCC(absl::MakeSpan(masks), 2,
//            absl::MakeSpan(recv));
//
//            sender->SendCMCC(absl::MakeSpan(sent2), 2);
//            sender->Flush();
//        }
//
//        pforeach(0, size, [&](int64_t i) {
//            shares[i] = rand[i] + recv[i];
//        });
//    }
//
//    void HistoGram::smultiplex(std::vector<uint64_t> &shares, const
//    std::vector<uint8_t> &masks) {
//        SPU_ENFORCE(shares.size() == masks.size(), "input length should be
//        equal"); for (size_t i = 0; i < masks.size(); i++) {
//            SPU_ENFORCE(masks[i] == 1 || masks[i] == 0, "invalid select
//            bits");
//        }
//
//        size_t size = shares.size();
//        auto rand = yacl::crypto::RandVec<uint64_t>(size, true);
//
//        // Compute (b0 ^ b1) ? -(x0 + x1) : (x0 + x1)
//        vector<uint64_t> sent2(size * 2);
//        pforeach(0, size, [&](int64_t i) {
//            uint64_t value0 = masks[i] != 0 ? -shares[i] : shares[i];
//            uint64_t value1 = (masks[i] ^ 1) != 0 ? -shares[i] : shares[i];
//            sent2[i * 2] = value0 - rand[i];
//            sent2[i * 2 + 1] = value1 - rand[i];
//        });
//
//        vector<uint64_t> recv(size);
//
//        auto sender = base_ot_->GetSenderCOT();
//        auto receiver = base_ot_->GetReceiverCOT();
//
//        if (role_ == psi::PsiRoleType::Sender) {
//            sender->SendCMCC(absl::MakeSpan(sent2), 2);
//            sender->Flush();
//
//            receiver->RecvCMCC(absl::MakeSpan(masks), 2,
//            absl::MakeSpan(recv));
//        } else {
//            receiver->RecvCMCC(absl::MakeSpan(masks), 2,
//            absl::MakeSpan(recv));
//
//            sender->SendCMCC(absl::MakeSpan(sent2), 2);
//            sender->Flush();
//        }
//
//        pforeach(0, size, [&](int64_t i) {
//            shares[i] = rand[i] + recv[i];
//        });
//    }
//
//    std::vector<uint8_t> HistoGram::msb(const std::vector<uint64_t> &shares,
//    size_t bit_width) {
//        size_t size = shares.size();
//
//        vector<uint8_t> msbs(shares.size());
//        std::transform(shares.begin(), shares.end(), msbs.begin(), [&](const
//        uint64_t vv) {
//            return (vv >> 63U) & 0x1;
//        });
//
//        FieldType field = FieldType::FM64;
//        bool greater_than = true;
//        size_t radix = 4;
//        const uint64_t mask63 = ((1ULL << 63) - 1);
//
//        NdArrayRef inp = ring_rand(field, {static_cast<int64_t>(size)});
//        DISPATCH_ALL_FIELDS(field, "", [&]() {
//            auto xinp = NdArrayView<ring2k_t>(inp);
//            if (role_ == psi::PsiRoleType::Sender) {
//                pforeach(0, xinp.numel(), [&](int64_t i) {
//                    xinp[i] = shares[i] & mask63;
//                });
//            } else {
//                pforeach(0, xinp.numel(), [&](int64_t i) {
//                    xinp[i] = mask63 - (shares[i] & mask63);
//                });
//            }
//        });
//
//        if (comp_prot_ == nullptr) {
//            comp_prot_ = make_shared<CompareProtocol>(base_ot_, radix);
//        }
//        auto flag = comp_prot_->Compute(inp, greater_than, bit_width);
//
//        DISPATCH_ALL_FIELDS(field, "", [&]() {
//            auto xinp = NdArrayView<ring2k_t>(flag);
//            pforeach(0, xinp.numel(), [&](int64_t i) {
//                msbs[i] = msbs[i] ^ (xinp[i] & 0x1);
//            });
//        });
//
//        return msbs;
//    }
//
//    void HistoGram::abs(vector<uint64_t> &shares) {
//        auto neg_sgns = msb(shares);
//        smultiplex(shares, neg_sgns);
//    }
//
//    void HistoGram::truncate(std::vector<uint64_t> &shares, size_t shift,
//    SignType sign) {
//        FieldType field = FieldType::FM64;
//        TruncateProtocol::Meta meta;
//        meta.sign = sign;
//        meta.signed_arith = false;
//        meta.shift_bits = shift;
//
//        if (trunc_prot_ == nullptr) {
//            trunc_prot_ = make_shared<TruncateProtocol>(base_ot_);
//        }
//
//        NdArrayRef inp = ring_rand(field,
//        {static_cast<int64_t>(shares.size())}); DISPATCH_ALL_FIELDS(field, "",
//        [&]() {
//            auto xinp = NdArrayView<ring2k_t>(inp);
//            pforeach(0, xinp.numel(), [&](int64_t i) {
//                xinp[i] = shares[i];
//            });
//        });
//
//        NdArrayRef oup = trunc_prot_->Compute(inp, meta);
//
//        DISPATCH_ALL_FIELDS(field, "", [&]() {
//            auto xoup = NdArrayView<ring2k_t>(oup);
//            pforeach(0, xoup.numel(), [&](int64_t i) {
//                shares[i] = xoup[i];
//            });
//        });
//    }
//
//    void HistoGram::multiply(std::vector<uint64_t> &shares1, const
//    std::vector<uint64_t> &shares2) {
//        SPU_ENFORCE(shares1.size() == shares2.size());
//
//        int64_t len = static_cast<int64_t>(shares1.size());
//        FieldType field = FieldType::FM64;
//
//        NdArrayRef inp = ring_rand(field, {len * 2});
//
//        DISPATCH_ALL_FIELDS(field, "", [&]() {
//            auto xinp = NdArrayView<ring2k_t>(inp);
//            const uint64_t *ptr1 = (role_ == psi::PsiRoleType::Sender) ?
//            shares1.data() : shares2.data(); const uint64_t *ptr2 = (role_ ==
//            psi::PsiRoleType::Sender) ? shares2.data() : shares1.data();
//
//            pforeach(0, len, [&](int64_t i) {
//                xinp[i] = ptr1[i];
//            });
//            pforeach(len, xinp.numel(), [&](int64_t i) {
//                xinp[i] = ptr2[i - len];
//            });
//        });
//
//        if (mul_ptr_ == nullptr) {
//            mul_ptr_ = std::make_shared<CheetahMul>(link_ctx_);
//        }
//
//        NdArrayRef oup;
//        if (role_ == psi::PsiRoleType::Sender) {
//            oup = mul_ptr_->MulOLE(inp, true);
//        } else {
//            oup = mul_ptr_->MulOLE(inp, false);
//        }
//
//        vector<uint64_t> cross0(len);
//        vector<uint64_t> cross1(len);
//        DISPATCH_ALL_FIELDS(field, "", [&]() {
//            auto xoup = NdArrayView<ring2k_t>(oup);
//
//            pforeach(0, len, [&](int64_t i) {
//                cross0[i] = xoup[i];
//            });
//            pforeach(len, xoup.numel(), [&](int64_t i) {
//                cross1[i - len] = xoup[i];
//            });
//        });
//
//        pforeach(0, len, [&](int64_t i) {
//            shares1[i] = shares1[i] * shares2[i] + cross0[i] + cross1[i];
//        });
//    }
//
//    void HistoGram::multiply(std::vector<uint64_t> &inputs) {
//        auto len = static_cast<int64_t>(inputs.size());
//        FieldType field = FieldType::FM64;
//
//        NdArrayRef inp = ring_rand(field, {len});
//
//        DISPATCH_ALL_FIELDS(field, "", [&]() {
//            auto xinp = NdArrayView<ring2k_t>(inp);
//            pforeach(0, xinp.numel(), [&](int64_t i) {
//                xinp[i] = inputs[i];
//            });
//        });
//
//        if (mul_ptr_ == nullptr) {
//            mul_ptr_ = std::make_shared<CheetahMul>(link_ctx_);
//        }
//
//        NdArrayRef oup;
//        if (role_ == psi::PsiRoleType::Sender) {
//            oup = mul_ptr_->MulOLE(inp, true);
//        } else {
//            oup = mul_ptr_->MulOLE(inp, false);
//        }
//
//        DISPATCH_ALL_FIELDS(field, "", [&]() {
//            auto xoup = NdArrayView<ring2k_t>(oup);
//            pforeach(0, xoup.numel(), [&](int64_t i) {
//                inputs[i] = xoup[i];
//            });
//        });
//    }
//
//    void HistoGram::square(std::vector<uint64_t> &shares) {
//        vector<uint64_t> temp(shares.begin(), shares.end());
//        multiply(temp);
//        pforeach(0, shares.size(), [&](int64_t i) {
//            shares[i] = shares[i] * shares[i] + temp[i] + temp[i];
//        });
//    }
//
//    void HistoGram::sinPI(std::vector<uint64_t> &shares, size_t log2_scale,
//    size_t w_bits) {
//        size_t len = shares.size();
//
//        vector<double> sin_arr(len);
//        vector<double> cos_arr(len);
//        double pow = std::pow(2, log2_scale + w_bits);
//
//        pforeach(0, len, [&](int64_t i) {
//            double value = shares[i] / pow * M_PI;
//            sin_arr[i] = std::sin(value);
//            cos_arr[i] = std::cos(value);
//        });
//
//        vector<uint64_t> values(len * 2);
//
//        const double *ptr0 = (role_ == psi::PsiRoleType::Sender) ?
//        sin_arr.data() : cos_arr.data(); const double *ptr1 = (role_ ==
//        psi::PsiRoleType::Sender) ? cos_arr.data() : sin_arr.data();
//        pforeach(0, len, [&](int64_t i) {
//            values[i] = encode(ptr0[i], log2_scale);
//        });
//        pforeach(len, values.size(), [&](int64_t i) {
//            values[i] = encode(ptr1[i - len], log2_scale);
//        });
//
//        multiply(values);
//
//        for (size_t i = 0; i < len; i++) {
//            shares[i] = values[i] + values[i + len];
//        }
//    }
//
//    void HistoGram::cosPI(std::vector<uint64_t> &shares, size_t log2_scale,
//    size_t w_bits) {
//        size_t len = shares.size();
//
//        vector<double> sin_arr(len);
//        vector<double> cos_arr(len);
//        double pow = std::pow(2, log2_scale + w_bits);
//
//        pforeach(0, len, [&](int64_t i) {
//            double value = shares[i] / pow * M_PI;
//            sin_arr[i] = std::sin(value);
//            cos_arr[i] = std::cos(value);
//        });
//
//        vector<uint64_t> values(len * 2);
//
//        const double *ptr0 = cos_arr.data();
//        const double *ptr1 = sin_arr.data();
//        pforeach(0, len, [&](int64_t i) {
//            values[i] = encode(ptr0[i], log2_scale);
//        });
//        pforeach(len, values.size(), [&](int64_t i) {
//            values[i] = encode(ptr1[i - len], log2_scale);
//        });
//
//        multiply(values);
//
//        for (size_t i = 0; i < len; i++) {
//            shares[i] = values[i] - values[i + len];
//        }
//    }
//
//    void HistoGram::sigmoid(std::vector<uint64_t> &shares, size_t log2_scale)
//    {
//        vector<double> coeffs = {
//                2.911534156270083, -3.5932930569416737, 3.618659659303885,
//                -2.4606874754063313, 1.3075041795616118, -0.45319919724799307,
//                0.10037762183081642
//        };
//
//        size_t len = shares.size();
//        uint64_t eight = encode(4.0, log2_scale);
//        uint64_t half = encode(0.25, log2_scale + log2_scale);
//        uint64_t one = half << 1U;
//
//        vector<uint64_t> values(len * 2);
//        std::transform(shares.begin(), shares.end(), values.begin(),
//        [&](uint64_t vv) {
//            return vv + eight;
//        });
//        std::transform(shares.begin(), shares.end(), values.begin() + len,
//        [&](uint64_t vv) {
//            return vv - eight;
//        });
//
//        auto time_start = chrono::high_resolution_clock::now();
//        auto neg_sgns = msb(values);
//        auto time_end = chrono::high_resolution_clock::now();
//        auto time_diff = chrono::duration_cast<chrono::microseconds>(time_end
//        - time_start); cout << "msb takes " << time_diff.count() << " us" <<
//        endl;
//
//        vector<double> sin_arr(len * coeffs.size());
//        vector<double> cos_arr(len * coeffs.size());
//        double pow = std::pow(2, log2_scale + 4);
//
//        pforeach(0, len, [&](int64_t i) {
//            double value = shares[i] / pow * M_PI;
//            sin_arr[i] = std::sin(value);
//            cos_arr[i] = std::cos(value);
//        });
//
//        for (size_t i = 1; i < coeffs.size(); i++) {
//            size_t offset = i * len;
//            pforeach(0, len, [&](int64_t j) {
//                size_t posj = offset - len + j;
//                sin_arr[offset + j] = sin_arr[j] * cos_arr[posj] + cos_arr[j]
//                * sin_arr[posj]; cos_arr[offset + j] = cos_arr[j] *
//                cos_arr[posj] - sin_arr[j] * sin_arr[posj];
//            });
//        }
//
//        for (size_t i = 0; i < coeffs.size(); i++) {
//            pforeach(0, len, [&](int64_t j) {
//                sin_arr[i * len + j] *= coeffs[i];
//            });
//        }
//
//        values.resize(sin_arr.size() + cos_arr.size());
//        if (role_ == psi::PsiRoleType::Sender) {
//            std::transform(sin_arr.begin(), sin_arr.end(), values.begin(),
//            [&](double vv) {
//                return encode(vv, log2_scale);
//            });
//            std::transform(cos_arr.begin(), cos_arr.end(), values.begin() +
//            sin_arr.size(), [&](double vv) {
//                return encode(vv, log2_scale);
//            });
//        } else {
//            std::transform(cos_arr.begin(), cos_arr.end(), values.begin(),
//            [&](double vv) {
//                return encode(vv, log2_scale);
//            });
//            std::transform(sin_arr.begin(), sin_arr.end(), values.begin() +
//            cos_arr.size(), [&](double vv) {
//                return encode(vv, log2_scale);
//            });
//        }
//
//        time_start = chrono::high_resolution_clock::now();
//        multiply(values);
//        time_end = chrono::high_resolution_clock::now();
//        time_diff = chrono::duration_cast<chrono::microseconds>(time_end -
//        time_start); cout << "mul takes " << time_diff.count() << " us" <<
//        endl;
//
//        for (size_t i = 0; i < sin_arr.size(); i++) {
//            values[i] = values[i] + values[i + sin_arr.size()];
//        }
//        values.resize(sin_arr.size());
//
//        std::fill_n(shares.data(), shares.size(), half);
//        for (size_t i = 0; i < coeffs.size(); i++) {
//            for (size_t j = 0; j < len; j++) {
//                shares[j] += values[i * len + j];
//            }
//        }
//
//        vector<uint64_t> temp(len + len);
//        std::transform(shares.begin(), shares.end(), temp.begin(),
//        [&](uint64_t vv) {
//            return -vv;
//        });
//        std::transform(shares.begin(), shares.end(), temp.begin() + len,
//        [&](uint64_t vv) {
//            return vv - one;
//        });
//
//        multiplex(temp, neg_sgns);
//
//        for (size_t i = 0; i < len; i++) {
//            shares[i] = temp[i] + temp[i + len] + one;
//        }
//
//        truncate(shares, log2_scale, SignType::Positive);
//    }
}  // namespace spu::mpc::cheetah
