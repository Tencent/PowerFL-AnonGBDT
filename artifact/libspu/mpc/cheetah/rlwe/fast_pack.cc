
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "seal/util/uintarith.h"
#include "seal/util/clipnormal.h"
#include "libspu/core/prelude.h"
#include "spdlog/spdlog.h"

#include "libspu/mpc/cheetah/rlwe/fast_pack.h"

namespace spu::mpc::cheetah {
    using namespace std;
    using namespace seal;

    void sample_poly_normal(
            std::shared_ptr<seal::UniformRandomGenerator> rng,
            const seal::EncryptionParameters &parms, const double stddev,
            uint64_t *destination)
    {
        auto coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();

        if (util::are_close(stddev, 0.0)) {
            util::set_zero_poly(coeff_count, coeff_modulus_size, destination);
            return;
        }

        seal::RandomToStandardAdapter engine(std::move(rng));
        util::ClippedNormalDistribution dist(
                0, stddev,
                util::global_variables::noise_distribution_width_multiplier * stddev);

        for (size_t i = 0; i < coeff_count; i++) {
            auto noise = static_cast<int64_t>(dist(engine));
            if (noise > 0) {
                for (size_t j = 0; j < coeff_modulus_size; j++) {
                    destination[i + j * coeff_count] = static_cast<uint64_t>(noise);
                }
            } else if (noise < 0) {
                noise = -noise;
                for (size_t j = 0; j < coeff_modulus_size; j++) {
                    destination[i + j * coeff_count] =
                            coeff_modulus[j].value() - static_cast<uint64_t>(noise);
                }
            } else {
                for (size_t j = 0; j < coeff_modulus_size; j++) {
                    destination[i + j * coeff_count] = 0;
                }
            }
        }
    }

    void encrypt_symmetric_zero(const SecretKey &secret_key, const SEALContext &context, parms_id_type parms_id,
                                bool is_ntt_form, double stddev, Ciphertext &destination)
    {
        // We use a fresh memory pool with `clear_on_destruction' enabled.
        seal::MemoryPoolHandle pool =
                seal::MemoryManager::GetPool(seal::mm_prof_opt::mm_force_new, true);

        const auto &context_data = *context.get_context_data(parms_id);
        const auto &parms = context_data.parms();
        const auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_modulus_size = coeff_modulus.size();
        size_t coeff_count = parms.poly_modulus_degree();
        const auto &ntt_tables = context_data.small_ntt_tables();
        size_t encrypted_size = 2;

        destination.resize(context, parms_id, encrypted_size);

        auto rng_error = parms.random_generator()->create();
        shared_ptr<seal::UniformRandomGenerator> rng_ciphertext;
        rng_ciphertext = seal::Blake2xbPRNGFactory().create();

        // Generate ciphertext: (c[0], c[1]) = ([-(as+e)]_q, a)
        uint64_t *c0 = destination.data(0);
        uint64_t *c1 = destination.data(1);

        // Sample a uniformly at random
        // sample the NTT form directly
        util::sample_poly_uniform(rng_ciphertext, parms, c1);

        // Sample e <-- chi
        auto noise(util::allocate_poly(coeff_count, coeff_modulus_size, pool));
        sample_poly_normal(rng_error, parms, stddev, noise.get());

        // calculate -(a*s + e) (mod q) and store in c[0]
        for (size_t i = 0; i < coeff_modulus_size; i++) {
            util::dyadic_product_coeffmod(secret_key.data().data() + i * coeff_count,
                                          c1 + i * coeff_count, coeff_count, coeff_modulus[i],
                                          c0 + i * coeff_count);

            if (is_ntt_form) {
                // Transform the noise e into NTT representation.
                ntt_negacyclic_harvey(noise.get() + i * coeff_count, ntt_tables[i]);
            }

            util::add_poly_coeffmod(noise.get() + i * coeff_count, c0 + i * coeff_count,
                                    coeff_count, coeff_modulus[i], c0 + i * coeff_count);

            util::negate_poly_coeffmod(c0 + i * coeff_count, coeff_count, coeff_modulus[i],
                                       c0 + i * coeff_count);
        }

        destination.parms_id() = parms_id;
        destination.is_ntt_form() = is_ntt_form;
        destination.scale() = 1.0;
    }

    void CKKSParams::multiply_plain_inplace(Ciphertext &encrypted, const Plaintext &plaintext) const {
        SPU_ENFORCE(is_valid_for(plaintext, ckks_context_), "invalid plaintext");
        if (plaintext.is_zero()) {
            double scale = encrypted.scale();
            encryptor_->encrypt_zero(encrypted.parms_id(), encrypted);
            encrypted.scale() = scale * plaintext.scale();
        } else {
            evaluator_->multiply_plain_inplace(encrypted, plaintext);
        }
    }

    void CKKSParams::multiply_plain(const Ciphertext &in, const Plaintext &plaintext, Ciphertext &out) const {
        out = in;
        multiply_plain_inplace(out, plaintext);
    }

    void CKKSParams::encode_non_ntt(const vector<int64_t> &vec, parms_id_type parms_id, Plaintext &plaintext,
                                    bool is_ntt_form) const
    {
        const auto &context_data = *ckks_context_.get_context_data(parms_id);
        const auto &parms = context_data.parms();
        const auto &coeff_modulus = parms.coeff_modulus();
        const auto &ntt_tables = context_data.small_ntt_tables();
        std::size_t coeff_modulus_size = coeff_modulus.size();
        std::size_t coeff_count = parms.poly_modulus_degree();

        SPU_ENFORCE(vec.size() <= coeff_count, "the size of input vector should be no more than {}", coeff_count);

        // Quick sanity check
        SPU_ENFORCE(util::product_fits_in(coeff_modulus_size, coeff_count), "invalid parameters");

        vector<int64_t> coeffds(coeff_count, 0);
        copy_n(vec.data(), vec.size(), coeffds.data());

        double max_coeff = 0;
        for (std::size_t i = 0; i < coeff_count; i++)
        {
            max_coeff = std::max(max_coeff, std::fabs(coeffds[i]));
        }

        // Verify that the values are not too large to fit in coeff_modulus
        // Note that we have an extra + 1 for the sign bit
        // Don't compute logarithmis of numbers less than 1
        int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max<>(max_coeff, 1.0)))) + 1;
        SPU_ENFORCE(max_coeff_bit_count < context_data.total_coeff_modulus_bit_count(), "encodedd values are too large");

        // Resize destination to appropriate size
        // Need to first set parms_id to zero, otherwise resize will throw an exception.
        plaintext.parms_id() = parms_id_zero;
        plaintext.resize(util::mul_safe(coeff_count, coeff_modulus_size));

        for (std::size_t i = 0; i < coeff_count; i++)
        {
            int64_t coeffd = coeffds[i];
            bool is_negative = std::signbit(coeffd);

            auto coeffu = static_cast<std::uint64_t>(std::fabs(coeffd));

            if (is_negative)
            {
                for (std::size_t j = 0; j < coeff_modulus_size; j++)
                {
                    plaintext[i + (j * coeff_count)] = util::negate_uint_mod(
                            util::barrett_reduce_64(coeffu, coeff_modulus[j]), coeff_modulus[j]);
                }
            }
            else
            {
                for (std::size_t j = 0; j < coeff_modulus_size; j++)
                {
                    plaintext[i + (j * coeff_count)] = util::barrett_reduce_64(coeffu, coeff_modulus[j]);
                }
            }
        }

        if (is_ntt_form) {
            // Transform to NTT domain
            for (std::size_t i = 0; i < coeff_modulus_size; i++)
            {
                util::ntt_negacyclic_harvey(plaintext.data(i * coeff_count), ntt_tables[i]);
            }
        }

        plaintext.parms_id() = context_data.parms_id();
        plaintext.scale() = 1.0;
    }

    void CKKSParams::encode_non_ntt(const vector<double> &vec, parms_id_type parms_id, double scale, Plaintext &plaintext,
                                    bool is_ntt_form) const

    {
        const auto &context_data = *ckks_context_.get_context_data(parms_id);
        const auto &parms = context_data.parms();
        const auto &coeff_modulus = parms.coeff_modulus();
        const auto &ntt_tables = context_data.small_ntt_tables();
        std::size_t coeff_modulus_size = coeff_modulus.size();
        std::size_t coeff_count = parms.poly_modulus_degree();

        SPU_ENFORCE(vec.size() <= coeff_count, "the size of input vector shoulde be no more than {}", coeff_count);

        // Quick sanity check
        SPU_ENFORCE(util::product_fits_in(coeff_modulus_size, coeff_count), "invalid parameters");

        // Check that scale is positive and not too large
        SPU_ENFORCE(scale > 0 && (static_cast<int>(log2(scale)) + 1 < context_data.total_coeff_modulus_bit_count()),
                    "scale out of bounds");

        vector<double> values(coeff_count, 0);
        for (size_t i = 0; i < vec.size(); i++) {
            values[i] = scale * vec[i];
        }

        double max_coeff = 0;
        for (std::size_t i = 0; i < coeff_count; i++)
        {
            max_coeff = std::max(max_coeff, std::fabs(values[i]));
        }

        // Verify that the values are not too large to fit in coeff_modulus
        // Note that we have an extra + 1 for the sign bit
        // Don't compute logarithmis of numbers less than 1
        int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max<>(max_coeff, 1.0)))) + 1;
        SPU_ENFORCE(max_coeff_bit_count < context_data.total_coeff_modulus_bit_count(), "encoded values are too large");

        double two_pow_64 = std::pow(2.0, 64);

        // Resize destination to appropriate size
        // Need to first set parms_id to zero, otherwise resize will throw an exception.
        plaintext.parms_id() = parms_id_zero;
        plaintext.resize(util::mul_safe(coeff_count, coeff_modulus_size));

        // Use faster decomposition methods when possible
        if (max_coeff_bit_count <= 64)
        {
            for (std::size_t i = 0; i < coeff_count; i++)
            {
                double coeffd = std::round(values[i]);
                bool is_negative = std::signbit(coeffd);

                auto coeffu = static_cast<std::uint64_t>(std::fabs(coeffd));

                if (is_negative)
                {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++)
                    {
                        plaintext[i + (j * coeff_count)] = util::negate_uint_mod(
                                util::barrett_reduce_64(coeffu, coeff_modulus[j]), coeff_modulus[j]);
                    }
                }
                else
                {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++)
                    {
                        plaintext[i + (j * coeff_count)] = util::barrett_reduce_64(coeffu, coeff_modulus[j]);
                    }
                }
            }
        }
        else if (max_coeff_bit_count <= 128)
        {
            for (std::size_t i = 0; i < coeff_count; i++)
            {
                double coeffd = std::round(values[i]);
                bool is_negative = std::signbit(coeffd);
                coeffd = std::fabs(coeffd);

                std::uint64_t coeffu[2]{ static_cast<std::uint64_t>(std::fmod(coeffd, two_pow_64)),
                                         static_cast<std::uint64_t>(coeffd / two_pow_64) };

                if (is_negative)
                {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++)
                    {
                        plaintext[i + (j * coeff_count)] = util::negate_uint_mod(
                                util::barrett_reduce_128(coeffu, coeff_modulus[j]), coeff_modulus[j]);
                    }
                }
                else
                {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++)
                    {
                        plaintext[i + (j * coeff_count)] = util::barrett_reduce_128(coeffu, coeff_modulus[j]);
                    }
                }
            }
        }
        else
        {
            // Slow case
            MemoryPoolHandle pool = MemoryManager::GetPool();

            vector<uint64_t> coeffu(coeff_modulus_size);
            for (std::size_t i = 0; i < coeff_count; i++)
            {
                double coeffd = std::round(values[i]);
                bool is_negative = std::signbit(coeffd);
                coeffd = std::fabs(coeffd);

                // We are at this point guaranteed to fit in the allocated space
                util::set_zero_uint(coeff_modulus_size, coeffu.data());
                auto coeffu_ptr = coeffu.data();
                while (coeffd >= 1)
                {
                    *coeffu_ptr++ = static_cast<std::uint64_t>(std::fmod(coeffd, two_pow_64));
                    coeffd /= two_pow_64;
                }

                // Next decompose this coefficient
                context_data.rns_tool()->base_q()->decompose(coeffu.data(), pool);

                // Finally replace the sign if necessary
                if (is_negative)
                {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++)
                    {
                        plaintext[i + (j * coeff_count)] = util::negate_uint_mod(coeffu[j], coeff_modulus[j]);
                    }
                }
                else
                {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++)
                    {
                        plaintext[i + (j * coeff_count)] = coeffu[j];
                    }
                }
            }
        }

        if (is_ntt_form) {
            // Transform to NTT domain
            for (std::size_t i = 0; i < coeff_modulus_size; i++)
            {
                util::ntt_negacyclic_harvey(plaintext.data(i * coeff_count), ntt_tables[i]);
            }
        }

        plaintext.parms_id() = ckks_context_.first_parms_id();
        plaintext.scale() = scale;
    }

    void CKKSParams::decode_non_ntt(const Plaintext &plaintext, vector<double> &vec)
    {
        SPU_ENFORCE(is_valid_for(plaintext, ckks_context_), "plain is not valid for encryption parameters");
        SPU_ENFORCE(plaintext.is_ntt_form(), "plain is not in NTT form");

        const auto &context_data = *ckks_context_.get_context_data(plaintext.parms_id());
        const auto &parms = context_data.parms();
        const auto &ntt_tables = context_data.small_ntt_tables();
        std::size_t coeff_modulus_size = parms.coeff_modulus().size();
        std::size_t coeff_count = parms.poly_modulus_degree();
        std::size_t rns_poly_uint64_count = util::mul_safe(coeff_count, coeff_modulus_size);

        // Check that scale is positive and not too large
        SPU_ENFORCE(plaintext.scale() > 0 && (static_cast<int>(std::log2(plaintext.scale())) <
                                              context_data.total_coeff_modulus_bit_count()), "scale out of bounds");

        const auto &decryption_modulus = context_data.total_coeff_modulus();
        const auto upper_half_threshold = context_data.upper_half_threshold();
        int logn = util::get_power_of_two(coeff_count);

        // Quick sanity check
        SPU_ENFORCE(logn >= 0 && coeff_count >= SEAL_POLY_MOD_DEGREE_MIN && coeff_count <= SEAL_POLY_MOD_DEGREE_MAX,
                    "invalid parameters");

        double inv_scale = 1.0 / plaintext.scale();

        vector<uint64_t> plain_copy(rns_poly_uint64_count);
        copy_n(plaintext.data(), rns_poly_uint64_count, plain_copy.data());

        // Transform each polynomial from NTT domain
        for (std::size_t i = 0; i < coeff_modulus_size; i++)
        {
            util::inverse_ntt_negacyclic_harvey(plain_copy.data() + (i * coeff_count), ntt_tables[i]);
        } // the row of plain_copy is coeff_modulus_size, the col of plain_copy is coeff_count

        // CRT-compose the polynomial
        MemoryPoolHandle pool = MemoryManager::GetPool();
        context_data.rns_tool()->base_q()->compose_array(plain_copy.data(), coeff_count, pool);
        // the row of plain_copy is coeff_count, the col of plain_copy is coeff_modulus_size

        // Create floating-point representations of the multi-precision integer coefficients
        double two_pow_64 = std::pow(2.0, 64);
        vector<double> res(coeff_count);
        for (std::size_t i = 0; i < coeff_count; i++)
        {
            res[i] = 0.0;
            if (util::is_greater_than_or_equal_uint(
                    plain_copy.data() + (i * coeff_modulus_size), upper_half_threshold, coeff_modulus_size))
            {
                double scaled_two_pow_64 = inv_scale;
                for (std::size_t j = 0; j < coeff_modulus_size; j++)
                {
                    if (plain_copy[i * coeff_modulus_size + j] > decryption_modulus[j])
                    {
                        auto diff = plain_copy[i * coeff_modulus_size + j] - decryption_modulus[j];
                        res[i] += (diff != 0) ? diff * scaled_two_pow_64 : 0.0;
                    }
                    else
                    {
                        auto diff = decryption_modulus[j] - plain_copy[i * coeff_modulus_size + j];
                        res[i] -= (diff != 0) ? diff * scaled_two_pow_64 : 0.0;
                    }
                    scaled_two_pow_64 *= two_pow_64;
                }
            }
            else
            {
                double scaled_two_pow_64 = inv_scale;
                for (std::size_t j = 0; j < coeff_modulus_size; j++)
                {
                    auto curr_coeff = plain_copy[i * coeff_modulus_size + j];
                    res[i] += (curr_coeff != 0) ? curr_coeff * scaled_two_pow_64 : 0.0;

                    scaled_two_pow_64 *= two_pow_64;
                }
            }

            // Scaling instead incorporated above; this can help in cases
            // where otherwise pow(two_pow_64, j) would overflow due to very
            // large coeff_modulus_size and very large scale
            // res[i] = res_accum * inv_scale;
        }

        vec.resize(coeff_count);
        copy_n(res.data(), coeff_count, vec.data());
    }

    void CKKSParams::encode_big_int(const vector<uint64_t> &arr, parms_id_type parms_id, Plaintext &plain) {
        const auto &context_data = ckks_context_.get_context_data(parms_id);
        const auto &coeff_modulus = context_data->parms().coeff_modulus();
        const auto &ntt_tables = context_data->small_ntt_tables();
        const auto &coeff_count = context_data->parms().poly_modulus_degree();

        size_t size = arr.size() / coeff_modulus.size();
        SPU_ENFORCE(arr.size() == coeff_modulus.size() * size && size <= coeff_count, "error in input length");

        plain.parms_id() = parms_id_zero;
        plain.resize(util::mul_safe(coeff_count, coeff_modulus.size()));

        size_t coeff_modulus_size = coeff_modulus.size();
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < coeff_modulus_size; j++) {
                SPU_ENFORCE(arr[i * coeff_modulus_size + j] < (1ULL << 63), "negative number is not allowed");
                plain[j * coeff_count + i] = arr[i * coeff_modulus_size + j] % coeff_modulus[j].value();
            }
        }

        for (size_t j = 0; j < coeff_modulus.size(); j++)
        {
            util::ntt_negacyclic_harvey(plain.data(j * coeff_count), ntt_tables[j]);
        }

        plain.parms_id() = parms_id;
        plain.scale() = 1.0;
    }

    void CKKSParams::decode_big_int(const Plaintext &origin, vector<uint64_t> &dest) {
        const auto &context_data = ckks_context_.get_context_data(origin.parms_id());
        const auto &coeff_modulus = context_data->parms().coeff_modulus();
        const auto &ntt_tables = context_data->small_ntt_tables();
        const auto &coeff_count = context_data->parms().poly_modulus_degree();

        Plaintext plain = origin;
        for (size_t j = 0; j < coeff_modulus.size(); j++)
        {
            util::inverse_ntt_negacyclic_harvey(plain.data(j * coeff_count), ntt_tables[j]);
        }

        size_t len = dest.size();
        dest.resize(len * coeff_modulus.size());

        for (size_t i = 0; i < len; i++) {
            for (size_t j = 0; j < coeff_modulus.size(); j++) {
                dest[i * coeff_modulus.size() + j] = plain[j * coeff_count + i];
            }
        }
    }

    void CKKSParams::mul_scalar(Ciphertext &encrypted, double scalar, double scale) const {
        double f_scalar = std::round(scalar * scale);
        int nbits = static_cast<int>(std::log2(std::abs(f_scalar)));
        if (nbits >= 63) {
            Plaintext plaintext;
            encoder_->encode(scalar, encrypted.parms_id(), scale, plaintext);
            evaluator_->multiply_plain_inplace(encrypted, plaintext);
        } else {
            const auto &context = ckks_context_.get_context_data(encrypted.parms_id());
            const uint64_t coeff_count = context->parms().poly_modulus_degree();
            const auto &coeff_modulus = context->parms().coeff_modulus();
            const uint64_t coeff_modulus_size = coeff_modulus.size();

            SPU_ENFORCE(nbits <= context->total_coeff_modulus_bit_count() + 2, "mul_scalar: scale out of bound.");

            const auto value = static_cast<uint64_t>(std::abs(f_scalar));
            bool sign = scalar < 0;

            for (size_t k = 0; k < encrypted.size(); k++) {
                uint64_t *poly = encrypted.data(k);
                for (size_t i = 0; i < coeff_modulus_size; i++) {
                    uint64_t multiplier = util::barrett_reduce_64(value, coeff_modulus[i]);

                    multiplier = sign ? coeff_modulus[i].value() - multiplier : multiplier;

                    util::multiply_poly_scalar_coeffmod(poly + i * coeff_count, coeff_count, multiplier, coeff_modulus[i],
                                                        poly + i * coeff_count);
                }
            }
            encrypted.scale() *= scale;
        }
    }

    CKKSParams::CKKSParams(SEALContext &context, bool enable_relin, bool enable_galois): ckks_context_(context) {
        KeyGenerator keygen(ckks_context_);
        secret_key_ = keygen.secret_key();
        keygen.create_public_key(public_key_);

        SecretKey sk = keygen.secret_key();

        const auto &parms = ckks_context_.first_context_data()->parms();
        for (const auto & coeff_modulus : parms.coeff_modulus()) {
            auto value = coeff_modulus.value();
            coeff_modulus_.push_back(value);
        }

        slot_count_ = parms.poly_modulus_degree() / 2;

        encryptor_ = std::make_shared<Encryptor>(ckks_context_, public_key_);
        evaluator_ = std::make_shared<Evaluator>(ckks_context_);
        encoder_   = std::make_shared<CKKSEncoder>(ckks_context_);

        if (enable_relin) {
            keygen.create_relin_keys(relin_keys_);
        }

        if (enable_galois) {
            auto galois_elts = ckks_context_.key_context_data()->galois_tool()->get_elts_all();

            sort(galois_elts.begin(), galois_elts.end());

            const uint32_t coeff_count = slot_count_<< 1U;
            for (size_t d = 1; d <= slot_count_; d = d << 1U) {
                unsigned int elt = (coeff_count + d) / d;
                if (!binary_search(galois_elts.begin(), galois_elts.end(), elt)) {
                    galois_elts.push_back(elt);
                }
            }

            keygen.create_galois_keys(galois_elts, galois_keys_);
        }
    }

    CKKSParams::CKKSParams(SEALContext &context, vector<uint8_t> &pk_bytes, vector<uint8_t> &rk_bytes,
                           vector<uint8_t> &gk_bytes):ckks_context_(context) {
        const auto &parms = ckks_context_.first_context_data()->parms();

        for (const auto & coeff_modulus : parms.coeff_modulus()) {
            auto value = static_cast<double>(coeff_modulus.value());
            coeff_modulus_.push_back(value);
        }

        slot_count_ = parms.poly_modulus_degree() / 2;

        pk_from_bytes(pk_bytes);
        rk_from_bytes(rk_bytes);
        gk_from_bytes(gk_bytes);

        encryptor_ = std::make_shared<Encryptor>(ckks_context_, public_key_);
        evaluator_ = std::make_shared<Evaluator>(ckks_context_);
        encoder_   = std::make_shared<CKKSEncoder>(ckks_context_);
    }

    CKKSParams::CKKSParams(std::shared_ptr<CKKSParams> &ckks_ptr):ckks_context_(ckks_ptr->ckks_context_) {
        const auto &parms = ckks_context_.first_context_data()->parms();

        for (const auto & coeff_modulus : parms.coeff_modulus()) {
            auto value = static_cast<double>(coeff_modulus.value());
            coeff_modulus_.push_back(value);
        }

        slot_count_ = parms.poly_modulus_degree() / 2;

        std::vector<uint8_t> pk_bytes;
        std::vector<uint8_t> rk_bytes;
        std::vector<uint8_t> gk_bytes;

        ckks_ptr->pk_to_bytes(pk_bytes);
        pk_from_bytes(pk_bytes);

        ckks_ptr->rk_to_bytes(rk_bytes);
        rk_from_bytes(rk_bytes);

        ckks_ptr->gk_to_bytes(gk_bytes);
        gk_from_bytes(gk_bytes);

        encryptor_ = std::make_shared<Encryptor>(ckks_context_, public_key_);
        evaluator_ = std::make_shared<Evaluator>(ckks_context_);
        encoder_   = std::make_shared<CKKSEncoder>(ckks_context_);

        if (!ckks_ptr->empty_secret_key()) {
            std::vector<uint8_t> sk_bytes;
            ckks_ptr->sk_to_bytes(sk_bytes);
            sk_from_bytes(sk_bytes);
        }
    }

    void CKKSParams::sk_from_bytes(const std::vector<uint8_t> &bytes) {
        SPU_ENFORCE(!bytes.empty(), "empty secret key");
        secret_key_.load(ckks_context_, reinterpret_cast<byte *>((uint8_t *)bytes.data()), bytes.size());
    }

    void CKKSParams::pk_from_bytes(const vector<uint8_t> &bytes) {
        SPU_ENFORCE(!bytes.empty(), "empty public key");
        public_key_.load(ckks_context_, reinterpret_cast<byte *>((uint8_t *)bytes.data()), bytes.size());
    }

    void CKKSParams::rk_from_bytes(const vector<uint8_t> &bytes) {
        if (!bytes.empty()) {
            relin_keys_.load(ckks_context_, reinterpret_cast<byte *>((uint8_t *)bytes.data()), bytes.size());
        }
    }

    void CKKSParams::gk_from_bytes(const vector<uint8_t> &bytes) {
        if (!bytes.empty()) {
            galois_keys_.load(ckks_context_, reinterpret_cast<byte *>((uint8_t *)bytes.data()), bytes.size());
        }
    }

    void CKKSParams::sk_to_bytes(std::vector<uint8_t> &bytes) {
        bytes.resize(secret_key_.save_size());
        secret_key_.save(reinterpret_cast<byte *>(bytes.data()), bytes.size());
    }

    void CKKSParams::pk_to_bytes(vector<uint8_t> &bytes) {
        bytes.resize(public_key_.save_size());
        public_key_.save(reinterpret_cast<byte *>(bytes.data()), bytes.size());
    }

    void CKKSParams::rk_to_bytes(vector<uint8_t> &bytes) {
        if (relin_keys_.size() == 0) {
            bytes.clear();
        } else {
            bytes.resize(relin_keys_.save_size());
            relin_keys_.save(reinterpret_cast<byte *>(bytes.data()), bytes.size());
        }
    }

    void CKKSParams::gk_to_bytes(vector<uint8_t> &bytes) {
        if (galois_keys_.size() == 0) {
            bytes.clear();
        } else {
            bytes.resize(galois_keys_.save_size());
            galois_keys_.save(reinterpret_cast<byte *>(bytes.data()), bytes.size());
        }
    }

    void CKKSParams::get_secret_key(vector<uint64_t> &sk) {
        SPU_ENFORCE(!empty_secret_key(), "empty secret key");

        const auto &context_data = ckks_context_.first_context_data();
        const auto &ntt_tables = context_data->small_ntt_tables();
        const auto coeff_count = context_data->parms().poly_modulus_degree();

        sk.resize(coeff_count);
        copy_n(secret_key_.data().data(), coeff_count, sk.data());

        util::inverse_ntt_negacyclic_harvey(sk.data(), ntt_tables[0]);
    }

    void CKKSParams::ciphertext_to_bytes(const seal::Ciphertext &encrypted, std::vector<uint8_t> &bytes) {
        SPU_ENFORCE(encrypted.size() != 0, "empty ciphertext");
        bytes.resize(encrypted.save_size());
        encrypted.save(reinterpret_cast<byte *>(bytes.data()), bytes.size());
    }

    void CKKSParams::bytes_to_ciphertext(const std::vector<uint8_t> &bytes, seal::Ciphertext &encrypted) {
        SPU_ENFORCE(!bytes.empty(), "empty byte stream");
        encrypted.load(ckks_context_, reinterpret_cast<byte *>((uint8_t *)bytes.data()), bytes.size());
    }

    void CKKSParams::switchKeys_to_bytes(const KSwitchKeys &switchKeys, vector<uint8_t> &bytes) {
        if (switchKeys.size() == 0) {
            bytes.clear();
        } else {
            bytes.resize(switchKeys.save_size());
            switchKeys.save(reinterpret_cast<byte *>(bytes.data()), bytes.size());
        }
    }

    void CKKSParams::bytes_to_switchKeys(const vector<uint8_t> &bytes, KSwitchKeys &switchKeys) {
        if (!bytes.empty()) {
            switchKeys.load(ckks_context_, reinterpret_cast<byte *>((uint8_t *)bytes.data()), bytes.size());
        }
    }

    void CKKSParams::encrypt_ntt(const vector<double> &vec, Ciphertext &encrypted, double scale) {
        SPU_ENFORCE(vec.size() <= slot_count_, "the size of input vector should be no more than {}", slot_count_);

        vector<double> pod(slot_count_, 0.0);

        copy_n(vec.data(), vec.size(), pod.data());

        scale = abs(scale) < 1 ? pow(2, 50) : scale;
        Plaintext plaintext;
        encoder_->encode(pod, scale, plaintext);
        encryptor_->encrypt(plaintext, encrypted);
    }

    void CKKSParams::decrypt_ntt(const Ciphertext &encrypted, vector<double> &vec, size_t vec_size) {
        if (empty_secret_key()) {
            SPDLOG_INFO("empty secret key");
            return;
        }

        Plaintext plaintext;
        Decryptor decryptor(ckks_context_, secret_key_);

        vector<double> pod;
        decryptor.decrypt(encrypted, plaintext);
        encoder_->decode(plaintext, pod);

        vec_size = vec_size == 0 ? vec.size() : vec_size;

        size_t len = min(pod.size(), vec_size);
        vec.resize(len);
        copy_n(pod.data(), len, vec.data());
    }

    void CKKSParams::encrypt_non_ntt(const vector<double> &vec, Ciphertext &encrypted, double scale) {
        scale = abs(scale) < 1 ? pow(2, 50) : scale;

        Plaintext plaintext;
        encode_non_ntt(vec, ckks_context_.first_parms_id(), scale, plaintext);
        encryptor_->encrypt(plaintext, encrypted);
    }

    void CKKSParams::decrypt_non_ntt(const Ciphertext &encrypted, vector<double> &vec, size_t vec_size) {
        if (empty_secret_key()) {
            SPDLOG_INFO("empty secret key");
            return;
        }

        Decryptor decryptor(ckks_context_, secret_key_);

        Ciphertext enc;
        if (!encrypted.is_ntt_form()) {
            evaluator_->transform_to_ntt(encrypted, enc);
        } else {
            enc = encrypted;
        }

        Plaintext plaintext;
        decryptor.decrypt(enc, plaintext);

        vector<double> pod;
        decode_non_ntt(plaintext, pod);

        vec_size = vec_size == 0 ? vec.size() : vec_size;

        size_t len = min(pod.size(), vec_size);
        vec.resize(len);
        copy_n(pod.data(), len, vec.data());
    }

    void CKKSParams::encrypt_big_int(const vector<uint64_t> &arr, Ciphertext &ciphertext) {
        Plaintext plaintext;
        encode_big_int(arr, ckks_context_.first_parms_id(), plaintext);
        encryptor_->encrypt(plaintext, ciphertext);
    }

    void CKKSParams::decrypt_big_int(const Ciphertext &ciphertext, vector<uint64_t> &vec, size_t vec_size) {
        if (empty_secret_key()) {
            SPDLOG_INFO("empty secret key");
            return;
        }

        Plaintext plaintext;
        Decryptor decryptor(ckks_context_, secret_key_);

        decryptor.decrypt(ciphertext, plaintext);

        vec_size = vec_size == 0 ? vec.size() : vec_size;

        size_t len = min(slot_count_ * 2, vec_size);
        vec.resize(len);

        decode_big_int(plaintext, vec);
    }

    void CKKSParams::add_inplace(Ciphertext &in1, const Ciphertext &in2) const {

        if (in1.coeff_modulus_size() == in2.coeff_modulus_size()) {
            if (util::are_close(in1.scale(), in2.scale())) {
                evaluator_->add_inplace(in1, in2);
            } else {
                        SPU_THROW("scale not equal.");
            }

            return;
        }

        Ciphertext enc_copy1;
        Ciphertext enc_copy2;
        if (in1.coeff_modulus_size() < in2.coeff_modulus_size()) {
            enc_copy1 = in1; enc_copy2 = in2;
        } else {
            enc_copy1 = in2; enc_copy2 = in1;
        }

        if (!util::are_close(enc_copy1.scale(), enc_copy2.scale())) {
            double scale = enc_copy1.scale() / enc_copy2.scale() * coeff_modulus_[enc_copy2.coeff_modulus_size()-1];
            mul_scalar(enc_copy2, 1.0, scale);
            evaluator_->rescale_to_next_inplace(enc_copy2);
        }

        evaluator_->mod_switch_to_inplace(enc_copy2, enc_copy1.parms_id());
        evaluator_->add(enc_copy1, enc_copy2, in1);
    }

    void CKKSParams::add_int_inplace(Ciphertext &ciphertext, const vector<uint64_t> &arr) {
        SPU_ENFORCE(arr.size() <= 2 * slot_count_, "data size is too large");

        Plaintext plaintext;
        encode_big_int(arr, ciphertext.parms_id(), plaintext);
        plaintext.scale() = ciphertext.scale();
        evaluator_->add_plain_inplace(ciphertext, plaintext);
    }

    void CKKSParams::sub_int_inplace(Ciphertext &ciphertext, const vector<uint64_t> &arr) {
        SPU_ENFORCE(arr.size() <= 2 * slot_count_, "data size is too large");

        Plaintext plaintext;
        encode_big_int(arr, ciphertext.parms_id(), plaintext);
        plaintext.scale() = ciphertext.scale();
        evaluator_->sub_plain_inplace(ciphertext, plaintext);
    }

    void CKKSParams::gen_switch_keys(const vector<uint64_t> &lwe_sk, KSwitchKeys &switch_keys) const {
        const auto &key_context_data = ckks_context_.key_context_data();
        const auto &key_parms = key_context_data->parms();
        size_t coeff_count = key_parms.poly_modulus_degree();
        const auto &key_modulus = key_parms.coeff_modulus();
        const auto &ntt_tables = key_context_data->small_ntt_tables();
        size_t decomp_mod_count = ckks_context_.first_context_data()->parms().coeff_modulus().size();

        SPU_ENFORCE(lwe_sk.size() <= coeff_count, "lwe_sk size too large");

        // Size check
        SPU_ENFORCE(util::product_fits_in(coeff_count, decomp_mod_count), "invalid parameters");

        // KSwitchKeys data allocated from pool given by MemoryManager::GetPool.
        vector<PublicKey> destination(decomp_mod_count);

        for (size_t j = 0; j < decomp_mod_count; j++) {
            const Modulus &qj = key_modulus[j];

            vector<uint64_t> temp(coeff_count, 0);
            temp[0] = lwe_sk[0] != 0 && lwe_sk[0] != 1 ? qj.value()-1 : lwe_sk[0];
            for (size_t i = 1; i < lwe_sk.size(); i++) {
                temp[coeff_count-i] = lwe_sk[i] != 0 && lwe_sk[i] != 1 ? 1 : util::negate_uint_mod(lwe_sk[i], qj);
            }

            encrypt_symmetric_zero(secret_key_, ckks_context_, key_parms.parms_id(), true,
                                   util::seal_he_std_parms_error_std_dev, destination[j].data());

            uint64_t factor = util::barrett_reduce_64(key_modulus.back().value(), qj);
            util::multiply_poly_scalar_coeffmod(temp.data(), coeff_count, factor, qj, temp.data());

            util::ntt_negacyclic_harvey(temp.data(), ntt_tables[j]);

            uint64_t *rns_ptr = destination[j].data().data(0) + j * coeff_count;
            util::add_poly_coeffmod(rns_ptr, temp.data(), coeff_count, qj, rns_ptr);

            destination[j].parms_id() = key_parms.parms_id();
        }

        switch_keys.data().resize(1);
        switch_keys.data()[0] = destination;
        switch_keys.parms_id() = key_parms.parms_id();
    }

    void CKKSParams::key_switch(Ciphertext &ct, const KSwitchKeys &switch_keys) const {
        SPU_ENFORCE(ct.size() >= 1, "key_switch: invalid ciphertext size");
        SPU_ENFORCE(is_metadata_valid_for(ct, ckks_context_), "key_switch: invalid runtime");
        SPU_ENFORCE(!ct.is_ntt_form(), "key_switch: require non_ntt ct");

        auto mem_pool = MemoryManager::GetPool();
        auto key_cntxt = ckks_context_.key_context_data();
        auto ct_cntxt = ckks_context_.get_context_data(ct.parms_id());

        const auto &inv_qk_mod_q = key_cntxt->rns_tool()->inv_q_last_mod_q();
        const auto &nttTables = key_cntxt->small_ntt_tables();
        const size_t coeff_count = ct_cntxt->parms().poly_modulus_degree();
        const size_t coeff_modulus_size = ct.coeff_modulus_size();
        const size_t max_coeff_modulus_size = ckks_context_.first_context_data()->parms().coeff_modulus().size();

        const auto &skeys = switch_keys.data(0);

        size_t index = ct.size() - 1;

        Ciphertext enc;
        enc.resize(ckks_context_, ct.parms_id(), 2);
        enc.is_ntt_form() = ct.is_ntt_form();
        enc.scale() = ct.scale();

        // [ct'[0]]_{qj} <- \sum_{i} [skeys[i, 0]]_{qj} * [ct[1]]_{qi} mod qj
        // [ct'[1]]_{qj} <- \sum_{i} [skeys[i, 1]]_{qj} * [ct[1]]_{qi} mod qj
        // qj includes special primes(s), qi only loop over cipher moduli.

        util::Pointer<uint64_t> lazy_mul_poly[2] = {
                util::allocate_poly(coeff_count * 2, 1, mem_pool),
                util::allocate_poly(coeff_count * 2, 1, mem_pool)};

        util::Pointer<uint64_t> spcl_rns_part[2] = {
                util::allocate_poly(coeff_count, 1, mem_pool),
                util::allocate_poly(coeff_count, 1, mem_pool)};

        util::Pointer<uint64_t> nrml_rns_part = util::allocate_poly(coeff_count, 1, mem_pool);

        for (ssize_t j = coeff_modulus_size; j >= 0; --j) {
            const bool is_special = (static_cast<size_t>(j) >= coeff_modulus_size);
            const size_t rns_idx = is_special ? max_coeff_modulus_size + (j - coeff_modulus_size) : j;

            std::fill_n(lazy_mul_poly[0].get(), coeff_count * 2, 0);
            std::fill_n(lazy_mul_poly[1].get(), coeff_count * 2, 0);

            std::vector<uint64_t> tmp_rns(coeff_count);

            for (size_t i = 0; i < coeff_modulus_size; ++i) {
                const uint64_t *ct_ptr = ct.data(index) + i * coeff_count;

                if (nttTables[i].modulus().value() > nttTables[rns_idx].modulus().value()) {
                    // qi > qk
                    auto mod_qj = nttTables[rns_idx].modulus();
                    std::transform(ct_ptr, ct_ptr + coeff_count, tmp_rns.data(),
                                   [&mod_qj](uint64_t u) {
                                       return seal::util::barrett_reduce_64(u, mod_qj);
                                   });
                } else {
                    // qi < qk
                    std::copy_n(ct_ptr, coeff_count, tmp_rns.data());
                }
                // ntt in [0, 4qi)
                util::ntt_negacyclic_harvey_lazy(tmp_rns.data(), nttTables[rns_idx]);
                const uint64_t *ct_qi_mod_qj = tmp_rns.data();  // [ct[l]_{qi}]_{qj}

                // [skeys[i, 0]]_{qj}
                const uint64_t *skeys0_qj = skeys[i].data().data(0) + rns_idx * coeff_count;
                // [skeys[i, 1]]_{qj}
                const uint64_t *skeys1_qj = skeys[i].data().data(1) + rns_idx * coeff_count;

                FMAU128(lazy_mul_poly[0].get(), skeys0_qj, ct_qi_mod_qj, coeff_count);
                FMAU128(lazy_mul_poly[1].get(), skeys1_qj, ct_qi_mod_qj, coeff_count);
            }

            // 2) Reduction and Rescale
            // 2-1) For special rns part, add (p-1)/2 and convert to the power-basis
            // 2-2) For normal rns part, compute qk^(1) (ct mod qi - ct mod qk) mod qi
            if (is_special) {
                for (int l : {0, 1}) {
                    auto acc_ptr = lazy_mul_poly[l].get();
                    uint64_t *dst_ptr = spcl_rns_part[l].get();
                    for (size_t d = 0; d < coeff_count; ++d, acc_ptr += 2) {
                        dst_ptr[d] = barrett_reduce_128_lazy(acc_ptr, nttTables[rns_idx].modulus());
                    }

                    util::inverse_ntt_negacyclic_harvey_lazy(dst_ptr, nttTables[rns_idx]);

                    const uint64_t half = nttTables[rns_idx].modulus().value() >> 1U;
                    SEAL_ITERATE(dst_ptr, coeff_count, [half, rns_idx, &nttTables](uint64_t &J) {
                        J = util::barrett_reduce_64(J + half, nttTables[rns_idx].modulus());
                    });
                }
            } else {
                const Modulus &mod_qj = nttTables[j].modulus();
                const uint64_t qk = nttTables[max_coeff_modulus_size].modulus().value();
                const uint64_t qj = mod_qj.value();
                const uint64_t neg_half_mod = qj - util::barrett_reduce_64(qk >> 1U, mod_qj);
                uint64_t qj_lazy = qj << 1U;  // some multiples of qi
                uint64_t inv_qk = inv_qk_mod_q[j].operand;

                std::vector<uint64_t> last_moduli(coeff_count);
                FastMulMod mulmod_s(inv_qk, qj);

                for (size_t l : {0U, 1U}) {  // two cipher components
                    const uint64_t *acc_ptr = lazy_mul_poly[l].get();
                    uint64_t *dst_ptr = nrml_rns_part.get();
                    // lazy reduce to [0, 2p)
                    for (size_t d = 0; d < coeff_count; ++d, acc_ptr += 2) {
                        dst_ptr[d] = barrett_reduce_128_lazy(acc_ptr, nttTables[rns_idx].modulus());
                    }

                    // [0, 2p)
                    util::inverse_ntt_negacyclic_harvey_lazy(dst_ptr, nttTables[j]);

                    const uint64_t *last_moduli_ptr = spcl_rns_part[l].get();
                    if (qk > qj) {
                        // Lazy add (p-1)/2, results in [0, 2p)
                        std::transform(
                                last_moduli_ptr, last_moduli_ptr + coeff_count, last_moduli.data(),
                                [neg_half_mod, &mod_qj](uint64_t u) {
                                    return barrett_reduce_64_lazy(u + neg_half_mod, mod_qj);
                                });
                    } else {
                        // Lazy add (p-1)/2, results in [0, 2p)
                        std::transform(
                                last_moduli_ptr, last_moduli_ptr + coeff_count, last_moduli.data(),
                                [neg_half_mod](uint64_t u) { return u + neg_half_mod; });
                    }

                    // qk^(-1) * ([ct]_qi - [ct]_qk) mod qi
                    std::transform(dst_ptr, dst_ptr + coeff_count, last_moduli.data(), dst_ptr,
                                   [&mulmod_s, qj_lazy](uint64_t c, uint64_t v) {
                                       return mulmod_s(c + qj_lazy - v);
                                   });

                    uint64_t *ans_ptr = enc.data(l) + j * coeff_count;
                    uint64_t *ct_ptr = ct.data(l) + j * coeff_count;

                    if (index == l) {
                        copy_n(dst_ptr, coeff_count, ans_ptr);
                    } else {
                        util::add_poly_coeffmod(dst_ptr, ct_ptr, coeff_count, mod_qj, ans_ptr);
                    }
                }
            }  // normal rns part
        }    // handle all the normal rns.

        ct = enc;
    }

    // X^d -> -X^d
    void CKKSParams::eval_auto(const Ciphertext &in, uint32_t d, Ciphertext &out) const {
        const auto coeff_count = slot_count_ << 1U;

        uint32_t galois_elt = (d + coeff_count) / d;
        evaluator_->apply_galois(in, galois_elt, galois_keys_, out);
    }

    uint64_t CKKSParams::rlwes_to_rlwe(const vector<Ciphertext> &rlwes, uint32_t d, Ciphertext &out) const {
        const auto coeff_count = slot_count_ << 1U;
        SPU_ENFORCE(rlwes.size() <= coeff_count, "the number of lwe ciphertext is too large");

        const auto parms_id = rlwes[0].parms_id();
        auto logL = static_cast<uint64_t>(std::log2(d));

        const auto &context_data = ckks_context_.get_context_data(parms_id);
        const auto &ntt_tables = context_data->small_ntt_tables();
        const auto &parms = context_data->parms();
        const auto &coeff_modulus = parms.coeff_modulus();
        const auto coeff_modulus_size = coeff_modulus.size();
        const auto prng = parms.random_generator()->create();

        vector<int64_t> valueX(coeff_count, 0);
        valueX[1] = 1;

        Plaintext plainX;
        encode_non_ntt(valueX, parms_id, plainX);

        vector<Plaintext> plainXs(logL + 1);
        plainXs[0] = plainX;

        for (size_t i = 1; i < plainXs.size(); i++) {
            for (size_t j = 0; j < coeff_modulus_size; j++) {
                util::dyadic_product_coeffmod(plainX.data() + j * coeff_count, plainX.data() + j * coeff_count,
                                              coeff_count, coeff_modulus[j], plainX.data() + j * coeff_count);
            }
            plainXs[i] = plainX;
        }

        Ciphertext enc;
        Ciphertext enc0;
        Ciphertext enc1;

        vector<Ciphertext> enc_buffer(rlwes.size());
        copy_n(rlwes.data(), rlwes.size(), enc_buffer.data());

        int l = logL;
        while (l >= 0) {
            size_t cnt = 0;
            for (size_t i = 0; i < enc_buffer.size(); i += 2) {
                if (i + 1 < enc_buffer.size()) {
                    evaluator_->multiply_plain(enc_buffer[i+1], plainXs[l], enc1);

                    evaluator_->add(enc_buffer[i], enc1, enc0);
                    evaluator_->sub(enc_buffer[i], enc1, enc);
                    eval_auto(enc, 1U << l, enc1);

                    evaluator_->add(enc0, enc1, enc_buffer[cnt]);
                } else if (enc_buffer.size() > 1) {
                    eval_auto(enc_buffer[i], 1U << l, enc1);
                    evaluator_->add(enc_buffer[i], enc1, enc_buffer[cnt]);
                } else {
                    enc_buffer[cnt] = enc_buffer[i];
                }

                cnt++;
            }

            enc_buffer.resize(cnt);
            l--;

            if (cnt == 1) {
                break;
            }
        }

        out = enc_buffer[0];

        vector<uint64_t> rand_poly(coeff_count, 0);

        uint64_t *rns_ptr0 = out.data(0);
        for (size_t j = 0; j < coeff_modulus_size; j++) {
            prng->generate(coeff_count * sizeof(uint64_t), reinterpret_cast<seal_byte *>(rand_poly.data()));
            util::modulo_poly_coeffs(rand_poly.data(), coeff_count, coeff_modulus[j], rand_poly.data());
            for (size_t i = 0; i < coeff_count; i += (1 << (l+1))) {
                rand_poly[i] = 0;
            }
            util::ntt_negacyclic_harvey(rand_poly.data(), ntt_tables[j]);
            util::add_poly_coeffmod(rns_ptr0 + j * coeff_count, rand_poly.data(), coeff_count, coeff_modulus[j],
                                    rns_ptr0 + j * coeff_count);
        }

        // trace
//            for (int i = 0; i <= l; i++) {
//                eval_auto(out, 1u << i, enc);
//                evaluator_->add_inplace(out, enc);
//            }

        return slot_count_ / (1U << (l+1));
    }

    void CKKSParams::lwes_lift_dim(const vector<vector<uint64_t>> &coeffs, const vector<uint64_t> &values,
                       uint64_t mod, const KSwitchKeys &switch_keys, vector<Ciphertext> &lwes) {
        SPU_ENFORCE(coeffs.size() == values.size(), "unequal size");

        size_t lwe_coeff_count = coeffs[0].size();
        size_t coeff_count = slot_count_ << 1U;

        SPU_ENFORCE(coeff_count >= (lwe_coeff_count << 1U), "only support the case lwe coeff count * 2 <= coeff count");

        lwes.clear();

        auto pool = MemoryManager::GetPool();

        const auto &context_data = ckks_context_.first_context_data();
        const auto &parms = context_data->parms();
        const auto &coeff_modulus = parms.coeff_modulus();
        const auto &rns_tool = context_data->rns_tool();

        SPU_ENFORCE(coeff_modulus.size() == 2, "only support the case coeff modulus size = 2");

        uint128_t total_coeff_modulus = static_cast<uint128_t>(coeff_modulus[0].value()) * coeff_modulus[1].value();
        uint128_t quotient = total_coeff_modulus / mod;
        uint128_t remainder= total_coeff_modulus - (quotient * mod);

        auto mod_switch = [&remainder, quotient, mod](uint64_t a) -> uint128_t {
            uint128_t ss = remainder * a;
            uint64_t qq = ss / mod;
            uint64_t rr = ss - (qq * mod);

            qq = (rr << 1U) >= mod ? qq + 1 : qq;
            return quotient * a + qq;
        };

        for (size_t i = 0; i < coeffs.size(); i += 2) {
            vector<uint128_t> b1(coeff_count, 0);
            vector<uint128_t> b0;

            // mod_switch
            for (size_t j = 0; j < lwe_coeff_count; j++) {
                b1[j] = mod_switch(coeffs[i][j]);
            }
            b0.push_back(mod_switch(values[i]));

            if (i + 1 < coeffs.size()) {
                for (size_t j = 0; j < lwe_coeff_count; j++) {
                    b1[j + slot_count_] = mod_switch(coeffs[i+1][j]);
                }
                b0.push_back(mod_switch(values[i+1]));
            }

            Ciphertext enc;
            enc.resize(ckks_context_, ckks_context_.first_parms_id(), 2);
            for (size_t j = 0; j < coeff_modulus.size(); j++) {
                uint64_t *rns_ptr1 = enc.data(1) + j * coeff_count;
                transform(b1.begin(), b1.end(), rns_ptr1, [&](uint128_t vv) -> uint64_t {
                    return static_cast<uint64_t>(vv % coeff_modulus[j].value());
                });

                uint64_t *rns_ptr0 = enc.data(0) + j * coeff_count;
                for (size_t l = 0; l < b0.size(); l++) {
                    rns_ptr0[l * slot_count_] = static_cast<uint64_t>(b0[l] % coeff_modulus[j].value());
                }
            }

            key_switch(enc, switch_keys);

            // rescale_next
            SEAL_ITERATE(util::iter(enc), enc.size(), [&](auto &I) {
                rns_tool->divide_and_round_q_last_inplace(I, pool);
            });

            Ciphertext dest;
            dest.resize(ckks_context_, context_data->next_context_data()->parms_id(), 2);
            SEAL_ITERATE(util::iter(enc, dest), 2, [&](auto I) {
                util::set_poly(get<0>(I), coeff_count, coeff_modulus.size() - 1, get<1>(I));
            });

            evaluator_->transform_to_ntt_inplace(dest);
            dest.scale() = 1.0;

            lwes.push_back(dest);
        }
    }

    uint64_t CKKSParams::sum(const Ciphertext &ct, const vector<int> &indexs, vector<uint64_t> &coeffs) {
        SPU_ENFORCE(ct.coeff_modulus_size() == 1, "only support coeff_modulus_size = 1");
        SPU_ENFORCE(ct.is_ntt_form(), "ciphertext must be non ntt form");
        SPU_ENFORCE(ct.size() == 2, "ciphertext size must be 2");

        const auto &context_data = ckks_context_.get_context_data(ct.parms_id());
        const auto &parms = context_data->parms();
        const auto &coeff_modulus = parms.coeff_modulus();
        const size_t coeff_count = slot_count_ << 1U;

        uint64_t vv = 0;
        for (auto index : indexs) {
            vv = util::add_uint_mod(vv, ct.data(0)[index], coeff_modulus[0]);
        }

        vector<uint64_t> tt(coeff_count);
        for (size_t i = 1; i < coeff_count; i++) {
            tt[coeff_count-i] = util::negate_uint_mod(ct.data(1)[i], coeff_modulus[0]);
        }
        tt[0] = ct.data(1)[0];

        size_t index = indexs[0];
        if (index != 0) {
            rotate(tt.begin(), tt.begin() + (coeff_count-index), tt.end());
            transform(tt.begin(), tt.begin() + index, tt.begin(), [&coeff_modulus](uint64_t vv) -> uint64_t {
                return util::negate_uint_mod(vv, coeff_modulus[0]);
            });
        }

        if (coeffs.size() < coeff_count) {
            coeffs.resize(coeff_count);
            copy_n(tt.data(), coeff_count, coeffs.data());
        } else {
            util::add_poly_coeffmod(tt.data(), coeffs.data(), coeff_count,
                                    coeff_modulus[0], coeffs.data());
        }

        vector<uint64_t> b1(coeff_count, 0);
        for (size_t i = 1; i < indexs.size(); i++) {
            SPU_ENFORCE(indexs[i-1] < indexs[i], "indexs should be ascending order");

            index = indexs[i] - indexs[i-1];

            if (index != 0) {
                rotate(tt.begin(), tt.begin() + (coeff_count-index), tt.end());
                transform(tt.begin(), tt.begin() + index, tt.begin(), [&coeff_modulus](uint64_t vv) -> uint64_t {
                    return util::negate_uint_mod(vv, coeff_modulus[0]);
                });
            }

            util::add_poly_coeffmod(tt.data(), coeffs.data(), coeff_count,
                                    coeff_modulus[0], coeffs.data());
        }

        return vv;
    }

    void CKKSParams::sum(const vector<Ciphertext> &cts, const vector<vector<int>> &index_vec,
             vector<vector<uint64_t>> &coeffs, vector<uint64_t> &values) {
        for (auto ct : cts) {
            SPU_ENFORCE(ct.is_ntt_form(), "input ciphertext should be in ntt form");
        }

        const auto &context_data = ckks_context_.get_context_data(cts[0].parms_id());
        const auto &parms = context_data->parms();
        const auto &coeff_modulus = parms.coeff_modulus()[0];
        const size_t coeff_count = slot_count_ << 1U;

        coeffs.resize(index_vec.size());
        values.resize(index_vec.size());

        vector<Ciphertext> encs(cts.size());
        Ciphertext enc;

        for (size_t l = 0; l < index_vec.size(); l++)
        {
            uint64_t offset = coeff_count;
            uint64_t pos = 0;
            for (size_t cnt = 0; cnt < cts.size(); cnt++)
            {
                vector<uint64_t> index;
                while (pos < index_vec[l].size() && index_vec[l][pos] < static_cast<int>(offset))
                {
                    uint64_t id = index_vec[l][pos] % coeff_count;
                    index.push_back(id);
                    pos = pos + 1;
                }
                offset += coeff_count;

                vector<int64_t> vec(coeff_count, 0);
                for (auto &id : index)
                {
                    if (id == 0) {
                        vec[0] = 1;
                    }
                    else {
                        vec[coeff_count - id] = -1;
                    }
                }

                Plaintext plaintext;
                encode_non_ntt(vec, cts[cnt].parms_id(), plaintext);
                multiply_plain(cts[cnt], plaintext, encs[cnt]);
            }

            evaluator_->add_many(encs, enc);
            evaluator_->transform_from_ntt_inplace(enc);

            coeffs[l].resize(coeff_count);
            for (size_t i = 1; i < coeff_count; i++) {
                coeffs[l][coeff_count-i] = util::negate_uint_mod(enc.data(1)[i], coeff_modulus);
            }
            coeffs[l][0] = enc.data(1)[0];

            values[l] = enc.data(0)[0] % coeff_modulus.value();
        }
    }

    int CKKSParams::get_aggregate_factor(int len) {
        const auto coeff_count = slot_count_ << 1U;
        const auto d = slot_count_ >> 1U;

        SPU_ENFORCE(len <= static_cast<int>(coeff_count), "invalid input length");

        auto logL = static_cast<uint64_t>(std::log2(d));

        int l = logL;
        while (l >= 0) {
            size_t cnt = (len + 1) >> 1U;
            len = cnt;
            l--;

            if (cnt == 1) {
                break;
            }
        }

        return slot_count_ / (1U << (l+1));
    }

    void CKKSParams::lwes_to_rlwe(const vector<vector<uint64_t>> &coeffs, const vector<uint64_t> &values,
                      uint64_t mod, double scale, const KSwitchKeys &switch_keys, Ciphertext &out) {
        vector<Ciphertext> rlwes;

//        auto time_start = std::chrono::high_resolution_clock::now();
        lwes_lift_dim(coeffs, values, mod, switch_keys, rlwes);
//        auto time_end = std::chrono::high_resolution_clock::now();
//        auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
//        std::cout << coeffs.size() << " lwes dim lift takes " << time_diff.count() << " us" << std::endl;

//        time_start = std::chrono::high_resolution_clock::now();
        uint64_t factor = rlwes_to_rlwe(rlwes, slot_count_ >> 1U, out);
//        time_end = std::chrono::high_resolution_clock::now();
//        time_diff = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
//        std::cout << rlwes.size() << " rlwes merge takes " << time_diff.count() << " us" << std::endl;

        out.scale() = scale * factor;
    }

    void CKKSParams::expand(const vector<uint64_t> &in, vector<uint64_t> &out) {
        uint64_t len = in.size();
        SPU_ENFORCE(len <= 2 * slot_count_, "error input length");

        vector<vector<uint64_t>> temp_vec;
        for (size_t i = 0; i < len; i += 2) {
            if (i + 1 < len) {
                temp_vec.push_back({0, slot_count_});
            } else {
                temp_vec.push_back({0});
            }
        }

        for (size_t l = slot_count_/2; l > 0; l = l >> 1u) {
            size_t cnt = 0;
            for (size_t i = 0; i < temp_vec.size(); i += 2) {
                std::copy_n(temp_vec[i].data(), temp_vec[i].size(), temp_vec[cnt].data());
                temp_vec[cnt].resize(temp_vec[i].size());
                if (i+1 < temp_vec.size()) {
                    for (auto j : temp_vec[i+1]) {
                        temp_vec[cnt].push_back(j + l);
                    }
                }
                cnt = cnt + 1;
            }
            temp_vec.resize(cnt);

            if (cnt == 1) {
                break;
            }
        }

        out.resize(slot_count_ << 1);
        for (size_t i = 0; i < len; i++) {
            out[temp_vec[0][i]] = in[i];
        }
    }

    void CKKSParams::decrypt_aggregate_lwes(const Ciphertext &ct, vector<uint64_t> &vec, size_t len) {
        vector<uint64_t> res(slot_count_ << 1U);
        decrypt_big_int(ct, res);

        len = len == 0 ? vec.size() : len;

        vector<vector<uint64_t>> temp_vec;
        for (size_t i = 0; i < len; i += 2) {
            if (i+1 < len) {
                temp_vec.push_back({0, slot_count_});
            } else {
                temp_vec.push_back({0});
            }
        }

        for (size_t l = slot_count_/2; l > 0; l = l >> 1U) {
            size_t cnt = 0;
            for (size_t i = 0; i < temp_vec.size(); i += 2) {
                copy_n(temp_vec[i].data(), temp_vec[i].size(), temp_vec[cnt].data());
                temp_vec[cnt].resize(temp_vec[i].size());
                if (i+1 < temp_vec.size()) {
                    for (auto j : temp_vec[i+1]) {
                        temp_vec[cnt].push_back(j + l);
                    }
                }
                cnt = cnt + 1;
            }
            temp_vec.resize(cnt);

            if (cnt == 1) {
                break;
            }
        }

        vec.resize(len);
        for (size_t i = 0; i < len; i++) {
            vec[i] = res[temp_vec[0][i]];
        }
    }
}