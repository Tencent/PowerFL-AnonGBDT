
#include "libspu/mpc/cheetah/rlwe/fast_pack.h"

#include "gtest/gtest.h"
#include "yacl/crypto/rand/rand.h"  // RandSeed

#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {
using namespace std;
using namespace seal;
class FastPackTest : public testing::TestWithParam<size_t> {};

INSTANTIATE_TEST_SUITE_P(Instances, FastPackTest, testing::Values(1000000));

void print(const vector<double> &vec) {
  for (const auto &vv : vec) {
    if (abs(vv) > 1e-6) {
      cout << vv << " ";
    } else {
      cout << 0.0 << " ";
    }
  }
  cout << endl;
};

void print_error(const vector<double> &vec1, const vector<double> &vec2) {
  SPU_ENFORCE(vec1.size() == vec2.size());

  double max_error = 0.0;
  double avg_error = 0.0;
  for (size_t i = 0; i < vec1.size(); i++) {
    double diff = std::abs(vec1[i] - vec2[i]);
    max_error = max_error < diff ? diff : max_error;
    avg_error = avg_error + diff;
  }

  cout << "max error: " << max_error
       << " avg error: " << (avg_error / vec1.size()) << endl;
}

TEST_P(FastPackTest, TestEvalAuto) {
  EncryptionParameters parms(scheme_type::ckks);

  mt19937_64 gen(random_device{}());
  auto dist_double = uniform_real_distribution<double>(-1, 1);
  auto scale = std::pow(2, 40);

  uint64_t coeff_count = 8192;
  parms.set_poly_modulus_degree(coeff_count);
  parms.set_coeff_modulus(CoeffModulus::Create(coeff_count, {60, 60, 60}));

  SEALContext context(parms, true, sec_level_type::tc128);

  auto ckks_ptr = new CKKSParams(context, true, true);

  vector<double> vec(coeff_count);
  for (auto &xx : vec) {
    xx = dist_double(gen);
  }

  print(vec);

  Ciphertext enc;
  ckks_ptr->encrypt_non_ntt(vec, enc, scale);

  Ciphertext enc_tmp;
  for (size_t i = 1; i < ckks_ptr->slot_count(); i = i << 1U) {
    ckks_ptr->eval_auto(enc, i, enc_tmp);
    ckks_ptr->add_inplace(enc, enc_tmp);
  }

  ckks_ptr->decrypt_non_ntt(enc, vec);
  print(vec);
}

TEST_P(FastPackTest, HistoGram) {
  EncryptionParameters parms(scheme_type::ckks);

  mt19937_64 gen(random_device{}());
  auto dist_double = uniform_real_distribution<double>(-1, 1);
  // auto dist_int = uniform_int_distribution(0, 2);
  auto scale = std::pow(2, 35);

  size_t lwe_coeff_count = 4096;

  parms.set_poly_modulus_degree(lwe_coeff_count);
  parms.set_coeff_modulus(CoeffModulus::Create(lwe_coeff_count, {60, 49}));

  SEALContext context0(parms);
  auto ckks_ptr0 = new CKKSParams(context0, false, false);

  vector<uint64_t> lwe_sk;
  ckks_ptr0->get_secret_key(lwe_sk);
  uint64_t lwe_mod = ckks_ptr0->get_coeff_modulus(0);

  size_t split_num = 5;
  size_t cols = 1000;
  size_t rows = 100000;

  vector<double> xValue(rows);
  for (auto &xx : xValue) {
    xx = dist_double(gen) * 10;
  }

  vector<Ciphertext> enc_vec;
  for (size_t i = 0; i < rows; i += lwe_coeff_count) {
    vector<double> vec(lwe_coeff_count, 0.0);

    size_t n = min(lwe_coeff_count, rows - i);
    std::copy_n(xValue.data() + i, n, vec.data());

    Ciphertext enc_tmp;
    ckks_ptr0->encrypt_non_ntt(vec, enc_tmp, scale);
    enc_vec.push_back(enc_tmp);
  }

  vector<vector<int>> index_vec(cols);
  for (size_t l = 0; l < cols; l += split_num) {
    vector<double> vec(rows);
    for (auto &xx : vec) {
      xx = dist_double(gen) * 10;
    }

    vector<double> splits(split_num - 1);
    for (auto &xx : splits) {
      xx = dist_double(gen) * 10;
    }
    std::sort(splits.begin(), splits.end());

    for (size_t i = 0; i < rows; i++) {
      if (vec[i] < splits.front()) {
        index_vec[l].push_back(static_cast<int>(i));
      }
    }

    for (size_t j = 1; j < splits.size(); j++) {
      for (size_t i = 0; i < rows; i++) {
        if (splits[j - 1] <= vec[i] && vec[i] < splits[j]) {
          index_vec[l + j].push_back(static_cast<int>(i));
        }
      }
    }

    for (size_t i = 0; i < rows; i++) {
      if (vec[i] >= splits.back()) {
        index_vec[l + split_num - 1].push_back(static_cast<int>(i));
      }
    }
  }

  vector<double> real(cols, 0.0);
  for (size_t l = 0; l < cols; l++) {
    const auto indexs = index_vec[l];
    for (auto index : indexs) {
      real[l] += xValue[index];
    }
  }

  vector<vector<uint64_t>> coeffs;
  vector<uint64_t> values;
  auto time_start = chrono::high_resolution_clock::now();
  ckks_ptr0->sum(enc_vec, index_vec, coeffs, values);
  auto time_end = chrono::high_resolution_clock::now();
  auto time_diff =
      chrono::duration_cast<chrono::microseconds>(time_end - time_start);
  cout << "hom sum takes: " << time_diff.count() << " us" << endl;

  size_t coeff_count = 8192;
  parms.set_poly_modulus_degree(coeff_count);
  parms.set_coeff_modulus(CoeffModulus::Create(coeff_count, {60, 49, 60}));

  SEALContext context1(parms);
  auto ckks_ptr1 = new CKKSParams(context1, false, true);

  KSwitchKeys switchKeys;
  ckks_ptr1->gen_switch_keys(lwe_sk, switchKeys);

  time_start = chrono::high_resolution_clock::now();
  Ciphertext enc;
  ckks_ptr1->lwes_to_rlwe(coeffs, values, lwe_mod, scale, switchKeys, enc);
  time_end = chrono::high_resolution_clock::now();
  time_diff =
      chrono::duration_cast<chrono::microseconds>(time_end - time_start);
  cout << "lwes to single rlwe takes " << time_diff.count() << " us" << endl;

  vector<uint64_t> res(cols);
  ckks_ptr1->decrypt_aggregate_lwes(enc, res);

  uint64_t rlwe_mod = ckks_ptr1->get_coeff_modulus(0);
  vector<double> comp(cols);
  for (size_t i = 0; i < cols; i++) {
    comp[i] = res[i] * 2 >= rlwe_mod
                  ? static_cast<int64_t>(res[i] - rlwe_mod) / enc.scale()
                  : res[i] / enc.scale();
  }

  print_error(comp, real);
}

TEST_P(FastPackTest, HistoGramSS) {
  mt19937_64 gen(random_device{}());
  auto dist_dbl = uniform_real_distribution<double>(-1, 1);
  // auto dist_int = uniform_int_distribution(0, 2);
  auto scale = std::pow(2, 35);

  size_t lwe_coeff_count = 4096;
  size_t coeff_count = 8192;

  size_t split_num = 5;
  size_t cols = 2000;
  size_t rows = 100000;

  vector<double> xValue(rows);
  for (auto &xx : xValue) {
    xx = dist_dbl(gen) * 10;
  }

  vector<uint64_t> arr1(rows);
  vector<uint64_t> arr2(rows);

  for (size_t i = 0; i < rows; i++) {
    arr1[i] = random_uint64();
    auto value = static_cast<uint64_t>(std::round(xValue[i] * scale));
    arr2[i] = value - arr1[i];
  }

  auto time_start = chrono::high_resolution_clock::now();
  EncryptionParameters parms(scheme_type::ckks);

  parms.set_poly_modulus_degree(lwe_coeff_count);
  parms.set_coeff_modulus(CoeffModulus::Create(lwe_coeff_count, {60, 49}));
  SEALContext context0(parms);

  parms.set_poly_modulus_degree(coeff_count);
  parms.set_coeff_modulus(CoeffModulus::Create(coeff_count, {60, 49, 60}));
  SEALContext context1(parms);

  auto key_ptr0 = make_shared<CKKSParams>(context0, false, false);
  auto key_ptr1 = make_shared<CKKSParams>(context1, false, true);

  vector<uint64_t> lwe_sk;
  key_ptr0->get_secret_key(lwe_sk);

  KSwitchKeys switchKeys;
  key_ptr1->gen_switch_keys(lwe_sk, switchKeys);

  vector<uint8_t> publicKey0_bytes;
  vector<uint8_t> publicKey1_bytes;
  vector<uint8_t> galoisKey_bytes;
  vector<uint8_t> switchKey_bytes;

  key_ptr0->pk_to_bytes(publicKey0_bytes);
  key_ptr1->pk_to_bytes(publicKey1_bytes);
  key_ptr1->gk_to_bytes(galoisKey_bytes);
  key_ptr1->switchKeys_to_bytes(switchKeys, switchKey_bytes);

  auto time_end = chrono::high_resolution_clock::now();
  auto time_diff =
      chrono::duration_cast<chrono::microseconds>(time_end - time_start);
  cout << "guest initialize takes " << time_diff.count() << " us" << endl;

  cout << "guest send "
       << (publicKey0_bytes.size() + publicKey1_bytes.size() +
           galoisKey_bytes.size() + switchKey_bytes.size())
       << " bytes to host" << endl;

  uint64_t lwe_mod = key_ptr0->get_coeff_modulus(0);
  uint64_t pow60 = 1ULL << 60;
  uint64_t mask60 = pow60 - 1;

  vector<uint64_t> xvalue1(rows);
  std::transform(arr1.begin(), arr1.end(), xvalue1.begin(), [&](uint64_t xx) {
    uint128_t ss = static_cast<uint128_t>(lwe_mod) * (xx & mask60);
    uint64_t yy = ss >> 60;
    if ((ss & mask60) * 2 >= pow60) {
      yy = yy + 1;
    }
    return yy % lwe_mod;
  });

  vector<uint64_t> xvalue2(rows);
  std::transform(arr2.begin(), arr2.end(), xvalue2.begin(), [&](uint64_t xx) {
    uint128_t ss = static_cast<uint128_t>(lwe_mod) * (xx & mask60);
    uint64_t yy = ss >> 60;
    if ((ss & mask60) * 2 >= pow60) {
      yy = yy + 1;
    }
    return yy % lwe_mod;
  });

  //        vector<double> tt(rows);
  //        for (size_t i = 0; i < rows; i++) {
  //            uint64_t vv = (xvalue1[i] + xvalue2[i]) % lwe_mod;
  //            tt[i] = vv * 2 >= lwe_mod ? static_cast<int64_t>(vv - lwe_mod) /
  //            scale : vv / scale;
  //        }
  //        print_error(xValue, tt);

  time_start = chrono::high_resolution_clock::now();

  vector<Ciphertext> enc_list;
  for (size_t i = 0; i < rows; i += lwe_coeff_count) {
    size_t n = min(lwe_coeff_count, rows - i);

    Ciphertext enc_tmp;
    vector<uint64_t> sub_vec(xvalue1.data() + i, xvalue1.data() + i + n);

    key_ptr0->encrypt_big_int(sub_vec, enc_tmp);
    enc_tmp.scale() = (scale * lwe_mod) / pow60;

    enc_list.push_back(enc_tmp);
  }

  //        vector<uint64_t> tt(rows);
  //        for (size_t i = 0; i < rows; i += lwe_coeff_count) {
  //            size_t n = min(lwe_coeff_count, rows - i);
  //
  //            vector<uint64_t> sub_vec(n);
  //            key_ptr0->decrypt_big_int(enc_list[i / lwe_coeff_count],
  //            sub_vec); std::copy_n(sub_vec.data(), n, tt.data() + i);
  //        }
  //
  //        for (size_t i = 0; i < rows; i++) {
  //            uint64_t vv = xvalue1[i] >= tt[i] ? xvalue1[i] - tt[i] : tt[i] -
  //            xvalue1[i]; if (((0.0 + vv) / tt[i]) > 1e-5) {
  //                cout << "有问题: " << xvalue1[i] << " " << tt[i] << endl;
  ////                break;
  //            }
  //        }

  time_end = chrono::high_resolution_clock::now();
  time_diff =
      chrono::duration_cast<chrono::microseconds>(time_end - time_start);
  cout << "guest encryption takes " << time_diff.count() << " us" << endl;

  vector<vector<int>> index_vec(cols);
  for (size_t l = 0; l < cols; l += split_num) {
    vector<double> vec(rows);
    for (auto &xx : vec) {
      xx = dist_dbl(gen) * 10;
    }

    vector<double> splits(split_num - 1);
    for (auto &xx : splits) {
      xx = dist_dbl(gen) * 10;
    }
    std::sort(splits.begin(), splits.end());

    for (size_t i = 0; i < rows; i++) {
      if (vec[i] < splits.front()) {
        index_vec[l].push_back(static_cast<int>(i));
      }
    }

    for (size_t j = 1; j < splits.size(); j++) {
      for (size_t i = 0; i < rows; i++) {
        if (splits[j - 1] <= vec[i] && vec[i] < splits[j]) {
          index_vec[l + j].push_back(static_cast<int>(i));
        }
      }
    }

    for (size_t i = 0; i < rows; i++) {
      if (vec[i] >= splits.back()) {
        index_vec[l + split_num - 1].push_back(static_cast<int>(i));
      }
    }
  }

  vector<double> real(cols, 0.0);
  for (size_t l = 0; l < cols; l++) {
    const auto indexs = index_vec[l];
    for (auto index : indexs) {
      real[l] += xValue[index];
    }
  }

  time_start = chrono::high_resolution_clock::now();

  vector<uint8_t> rk{}, gk{};
  auto ckks_ptr0 = make_shared<CKKSParams>(context0, publicKey0_bytes, rk, gk);
  auto ckks_ptr1 =
      make_shared<CKKSParams>(context1, publicKey1_bytes, rk, galoisKey_bytes);

  time_start = chrono::high_resolution_clock::now();

  for (size_t i = 0; i < rows; i += lwe_coeff_count) {
    size_t n = min(lwe_coeff_count, rows - i);

    vector<uint64_t> sub_vec(xvalue2.data() + i, xvalue2.data() + i + n);
    ckks_ptr0->add_int_inplace(enc_list[i / lwe_coeff_count], sub_vec);
  }

  //        vector<double> yy(rows);
  //        for (size_t i = 0; i < rows; i += lwe_coeff_count) {
  //            size_t n = min(lwe_coeff_count, rows - i);
  //
  //            vector<uint64_t> sub_vec(n);
  //            key_ptr0->decrypt_big_int(enc_list[i / lwe_coeff_count],
  //            sub_vec);
  //
  //            std::transform(sub_vec.begin(), sub_vec.end(), yy.data() + i,
  //            [&](int64_t xx) {
  //                auto vv = static_cast<uint64_t>(xx);
  //                return vv * 2 >= lwe_mod ? static_cast<int64_t>(vv -
  //                lwe_mod) / enc_list[i / lwe_coeff_count].scale()
  //                                         : vv / enc_list[i /
  //                                         lwe_coeff_count].scale();
  //            });
  //
  //            for (size_t j = 0; j < n; j++) {
  //                cout << yy[i + j] << " ";
  //            }
  //            cout << endl;
  //        }
  //        print(xValue);
  //        print(yy);
  //        print_error(xValue, yy);

  vector<vector<uint64_t>> coeffs;
  vector<uint64_t> values;
  ckks_ptr0->sum(enc_list, index_vec, coeffs, values);

  time_end = chrono::high_resolution_clock::now();
  time_diff =
      chrono::duration_cast<chrono::microseconds>(time_end - time_start);
  cout << "host hom sum takes: " << time_diff.count() << " us" << endl;

  time_start = chrono::high_resolution_clock::now();

  Ciphertext enc;
  ckks_ptr1->lwes_to_rlwe(coeffs, values, lwe_mod, enc_list[0].scale(),
                          switchKeys, enc);

  uint64_t rlwe_mod = ckks_ptr1->get_coeff_modulus(0);
  vector<uint64_t> yvalue2(cols);
  for (auto &vv : yvalue2) {
    vv = random_uint64() % rlwe_mod;
  }
  vector<uint64_t> mask;
  ckks_ptr1->expand(yvalue2, mask);
  ckks_ptr1->sub_int_inplace(enc, mask);

  time_end = chrono::high_resolution_clock::now();
  time_diff =
      chrono::duration_cast<chrono::microseconds>(time_end - time_start);
  cout << "host pack lwes takes: " << time_diff.count() << " us" << endl;

  time_start = chrono::high_resolution_clock::now();
  vector<uint64_t> yvalue1;
  key_ptr1->decrypt_aggregate_lwes(enc, yvalue1, cols);
  time_end = chrono::high_resolution_clock::now();
  time_diff =
      chrono::duration_cast<chrono::microseconds>(time_end - time_start);
  cout << "guest decryption takes: " << time_diff.count() << " us" << endl;

  double pow64 = std::pow(2, 64);
  for (size_t i = 0; i < cols; i++) {
    yvalue1[i] = std::round((pow64 * yvalue1[i]) / rlwe_mod);
    yvalue2[i] = std::round((pow64 * yvalue2[i]) / rlwe_mod);
  }

  vector<double> comp(cols);
  //        for (size_t i = 0; i < cols; i++) {
  //            uint64_t value = (yvalue1[i] + yvalue2[i]) % rlwe_mod;
  //            comp[i] = value * 2 >= rlwe_mod ? static_cast<int64_t>(value -
  //            rlwe_mod) / enc.scale() : value / enc.scale();
  //        }

  double new_scale = enc.scale() * pow64 / rlwe_mod;
  for (size_t i = 0; i < cols; i++) {
    comp[i] = static_cast<int64_t>(yvalue1[i] + yvalue2[i]) / new_scale;
  }

  print_error(comp, real);
}
}  // namespace spu::mpc::cheetah