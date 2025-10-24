
#ifndef SPU_FAST_PACK_H
#define SPU_FAST_PACK_H

#include "seal/seal.h"
#include "seal/plaintext.h"
#include "seal/secretkey.h"
#include "seal/publickey.h"
#include "seal/serialization.h"
#include "seal/util/uintarith.h"
#include "seal/util/rlwe.h"
#include "seal/util/scalingvariant.h"
#include "libspu/core/prelude.h"
#include "spdlog/spdlog.h"
#include <vector>

namespace spu::mpc::cheetah {

    void sample_poly_normal(
            std::shared_ptr<seal::UniformRandomGenerator> rng,
            const seal::EncryptionParameters &parms, const double stddev,
            uint64_t *destination);

    void encrypt_symmetric_zero(const seal::SecretKey &secret_key, const seal::SEALContext &context, seal::parms_id_type parms_id,
                                bool is_ntt_form, double stddev, seal::Ciphertext &destination);

    /// acc += op0* op1 using double-width precision
    /// |acc| = 2 * |op0| = 2 * |op1|
    inline void FMAU128(uint64_t *acc, const uint64_t *op0, const uint64_t *op1, const size_t len) {
        unsigned long long wide[2];
        for (size_t i = 0; i < len; ++i, acc += 2) {
            seal::util::multiply_uint64(*op0++, *op1++, wide);
            uint64_t _wp[2] = {static_cast<uint64_t>(wide[0]),
                               static_cast<uint64_t>(wide[1])};
            auto carry = seal::util::add_uint(acc, _wp, 2, acc);
            SPU_ENFORCE(carry == 0, "FMAU128: overflow!");
        }
    }

    inline uint64_t barrett_reduce_64_lazy(const uint64_t input,
                                           const seal::Modulus &modulus) {
        // Reduces input using base 2^64 Barrett reduction
        // input must be at most 63 bits

        SPU_ENFORCE(!modulus.is_zero(), "invalid mdulus");
        SPU_ENFORCE((input >> 63) == 0, "input must be at most 63 bits");

        unsigned long long tmp[2];
        const uint64_t *const_ratio = modulus.const_ratio().data();
        seal::util::multiply_uint64(input, const_ratio[1], tmp);

        // Barrett subtraction
        return input - tmp[1] * modulus.value();
    }

    inline uint64_t barrett_reduce_128_lazy(const uint64_t *input,
                                            const seal::Modulus &modulus) {
        unsigned long long tmp1;
        unsigned long long tmp2[2];
        unsigned long long tmp3;
        unsigned long long carry;

        const uint64_t *const_ratio = modulus.const_ratio().data();

        // Multiply input and const_ratio
        // Round 1
        seal::util::multiply_uint64_hw64(input[0], const_ratio[0], &carry);

        seal::util::multiply_uint64(input[0], const_ratio[1], tmp2);
        tmp3 = tmp2[1] + seal::util::add_uint64(tmp2[0], carry, 0, &tmp1);

        // Round 2
        seal::util::multiply_uint64(input[1], const_ratio[0], tmp2);
        carry = tmp2[1] + seal::util::add_uint64(tmp1, tmp2[0], 0, &tmp1);

        // This is all we care about
        tmp1 = input[1] * const_ratio[1] + tmp3 + carry;

        // Barrett subtraction
        return input[0] - tmp1 * modulus.value();
    }

    struct FastMulMod {
        uint64_t cnst, p;
        uint64_t cnst_shoup;

        explicit FastMulMod(uint64_t cnst, uint64_t p) : cnst(cnst), p(p) {
            uint64_t cnst_128[2]{0, cnst};
            uint64_t shoup[2];
            seal::util::divide_uint128_inplace(cnst_128, p, shoup);
            cnst_shoup = shoup[0];  // cnst_shoup = cnst * 2^64 / p
        }

        uint64_t lazy(uint64_t x) const {
            unsigned long long hw64;
            seal::util::multiply_uint64_hw64(x, cnst_shoup, &hw64);
            std::uint64_t q = static_cast<std::uint64_t>(hw64) * p;
            return (x * cnst - q);
        }

        inline uint64_t operator()(uint64_t x) const {
            uint64_t t = lazy(x);
            return t - ((p & -static_cast<uint64_t>(t < p)) ^ p);
        }
    };

    class CKKSParams {
        seal::SecretKey secret_key_;
        seal::PublicKey public_key_;

        seal::GaloisKeys galois_keys_;
        seal::RelinKeys relin_keys_;

        seal::SEALContext ckks_context_;
        std::shared_ptr<seal::Encryptor> encryptor_;
        std::shared_ptr<seal::Evaluator> evaluator_;
        std::shared_ptr<seal::CKKSEncoder> encoder_;

        std::vector<uint64_t> coeff_modulus_;
        size_t slot_count_;

        void multiply_plain_inplace(seal::Ciphertext &encrypted, const seal::Plaintext &plaintext) const;

        void multiply_plain(const seal::Ciphertext &in, const seal::Plaintext &plaintext, seal::Ciphertext &out) const;

        void encode_non_ntt(const std::vector<int64_t> &vec, seal::parms_id_type parms_id, seal::Plaintext &plaintext,
                            bool is_ntt_form = true) const;

        void encode_non_ntt(const std::vector<double> &vec, seal::parms_id_type parms_id, double scale,
                            seal::Plaintext &plaintext, bool is_ntt_form = true) const;

        void decode_non_ntt(const seal::Plaintext &plaintext, std::vector<double> &vec);

        void encode_big_int(const std::vector<uint64_t> &arr, seal::parms_id_type parms_id, seal::Plaintext &plain);

        void decode_big_int(const seal::Plaintext &origin, std::vector<uint64_t> &dest);

        void mul_scalar(seal::Ciphertext &encrypted, double scalar, double scale) const;

        uint64_t rlwes_to_rlwe(const std::vector<seal::Ciphertext> &rlwes, uint32_t d, seal::Ciphertext &out) const;

        void lwes_lift_dim(const std::vector<std::vector<uint64_t>> &coeffs, const std::vector<uint64_t> &values,
                           uint64_t mod, const seal::KSwitchKeys &switch_keys, std::vector<seal::Ciphertext> &lwes);

    public:
        explicit CKKSParams(seal::SEALContext &context, bool enable_relin, bool enable_galois);

        explicit CKKSParams(seal::SEALContext &context, std::vector<uint8_t> &pk_bytes, std::vector<uint8_t> &rk_bytes,
                            std::vector<uint8_t> &gk_bytes);

        explicit CKKSParams(std::shared_ptr<CKKSParams> &ckks_ptr);

        ~CKKSParams() = default;

        uint64_t get_coeff_modulus(size_t i) const {
            SPU_ENFORCE(i < coeff_modulus_.size(), "invalid index out of range.");
            return coeff_modulus_.at(i);
        }

        bool empty_secret_key() const {
            return secret_key_.data().coeff_count() == 0;
        }

        size_t slot_count() const  {
            return slot_count_;
        }

        size_t coeff_modulus_size() const {
            return coeff_modulus_.size();
        }

        void sk_from_bytes(const std::vector<uint8_t> &bytes);

        void sk_to_bytes(std::vector<uint8_t> &bytes);

        void pk_from_bytes(const std::vector<uint8_t> &bytes);

        void rk_from_bytes(const std::vector<uint8_t> &bytes);

        void gk_from_bytes(const std::vector<uint8_t> &bytes);

        void pk_to_bytes(std::vector<uint8_t> &bytes);

        void rk_to_bytes(std::vector<uint8_t> &bytes);

        void gk_to_bytes(std::vector<uint8_t> &bytes);

        void get_secret_key(std::vector<uint64_t> &sk);

        void ciphertext_to_bytes(const seal::Ciphertext &encrypted, std::vector<uint8_t> &bytes);

        void bytes_to_ciphertext(const std::vector<uint8_t> &bytes, seal::Ciphertext &encrypted);

        void switchKeys_to_bytes(const seal::KSwitchKeys &switchKeys, std::vector<uint8_t> &bytes);

        void bytes_to_switchKeys(const std::vector<uint8_t> &bytes, seal::KSwitchKeys &switchKeys);

        void encrypt_ntt(const std::vector<double> &vec, seal::Ciphertext &encrypted, double scale = 0.0);

        void decrypt_ntt(const seal::Ciphertext &encrypted, std::vector<double> &vec, size_t vec_size = 0);

        void encrypt_non_ntt(const std::vector<double> &vec, seal::Ciphertext &encrypted, double scale = 0.0);

        void decrypt_non_ntt(const seal::Ciphertext &encrypted, std::vector<double> &vec, size_t vec_size = 0);

        void encrypt_big_int(const std::vector<uint64_t> &arr, seal::Ciphertext &ciphertext);

        void decrypt_big_int(const seal::Ciphertext &ciphertext, std::vector<uint64_t> &vec, size_t vec_size = 0);

        void add_inplace(seal::Ciphertext &in1, const seal::Ciphertext &in2) const;

        void add_int_inplace(seal::Ciphertext &ciphertext, const std::vector<uint64_t> &arr);

        void sub_int_inplace(seal::Ciphertext &ciphertext, const std::vector<uint64_t> &arr);

        void gen_switch_keys(const std::vector<uint64_t> &lwe_sk, seal::KSwitchKeys &switch_keys) const;

        void key_switch(seal::Ciphertext &ct, const seal::KSwitchKeys &switch_keys) const;

        // X^d -> -X^d
        void eval_auto(const seal::Ciphertext &in, uint32_t d, seal::Ciphertext &out) const;

        uint64_t sum(const seal::Ciphertext &ct, const std::vector<int> &indexs, std::vector<uint64_t> &coeffs);

        void sum(const std::vector<seal::Ciphertext> &cts, const std::vector<std::vector<int>> &index_vec,
                 std::vector<std::vector<uint64_t>> &coeffs, std::vector<uint64_t> &values);

        int get_aggregate_factor(int len);

        void lwes_to_rlwe(const std::vector<std::vector<uint64_t>> &coeffs, const std::vector<uint64_t> &values,
                          uint64_t mod, double scale, const seal::KSwitchKeys &switch_keys, seal::Ciphertext &out);

        void expand(const std::vector<uint64_t> &in, std::vector<uint64_t> &out);

        void decrypt_aggregate_lwes(const seal::Ciphertext &ct, std::vector<uint64_t> &vec, size_t len = 0);
    };
}


#endif //SPU_FAST_PACK_H
