#include "libspu/mpc/cheetah/okvs/galois.h"

#include <utility>

#include "absl/strings/escaping.h"
#include "yacl/base/exception.h"
#include "yacl/utils/platform_utils.h"

#ifdef __x86_64__
#include "cpu_features/cpuinfo_x86.h"
#endif

namespace spu::mpc::cheetah {

// namespace {

#ifdef __x86_64__
    static const auto kCpuFeatures = cpu_features::GetX86Info().features;
    static const bool kHasPCLML = kCpuFeatures.pclmulqdq;
#else
    static const bool kHasPCLML = false;
#endif

    bool hasPCLML() { return kHasPCLML; }

#ifdef __x86_64__
    void mm_gf128Mul(const yacl::block& x, const yacl::block& y, yacl::block& xy1,
                     yacl::block& xy2) {
        yacl::block t1 = _mm_clmulepi64_si128(x, y, (int)0x00);
        yacl::block t2 = _mm_clmulepi64_si128(x, y, 0x10);
        yacl::block t3 = _mm_clmulepi64_si128(x, y, 0x01);
        yacl::block t4 = _mm_clmulepi64_si128(x, y, 0x11);
        t2 = (t2 ^ t3);
        t3 = _mm_slli_si128(t2, 8);
        t2 = _mm_srli_si128(t2, 8);
        t1 = (t1 ^ t3);
        t4 = (t4 ^ t2);

        xy1 = t1;
        xy2 = t4;
    }

    yacl::block mm_gf128Reduce(const yacl::block& x, const yacl::block& x1) {
        auto mul256_low = x;
        auto mul256_high = x1;
        static const constexpr std::uint64_t mod = 0b10000111;

        /* reduce w.r.t. high half of mul256_high */
        const __m128i modulus = _mm_loadl_epi64((const __m128i*)&(mod));
        __m128i tmp = _mm_clmulepi64_si128(mul256_high, modulus, 0x01);
        mul256_low = _mm_xor_si128(mul256_low, _mm_slli_si128(tmp, 8));
        mul256_high = _mm_xor_si128(mul256_high, _mm_srli_si128(tmp, 8));

        /* reduce w.r.t. low half of mul256_high */
        tmp = _mm_clmulepi64_si128(mul256_high, modulus, 0x00);
        mul256_low = _mm_xor_si128(mul256_low, tmp);

        // std::cout << "redu " << bits(x, 128) << std::endl;
        // std::cout << "     " << bits(mul256_low, 128) << std::endl;

        return mul256_low;
    }
#endif

// Multiplication in GF2^128, Reference
// The Galois/Counter Mode of Operation (GCM)
// https://csrc.nist.rip/groups/ST/toolkit/BCM/documents/proposedmodes/gcm/gcm-revised-spec.pdf
// P9 Algorithm 1 Multiplication in GF2^128
    uint128_t cc_gf128Mul(const uint128_t a, const uint128_t b) {
        uint128_t z = yacl::MakeUint128(0, 0);
        uint128_t v = a;

        uint128_t mask1 = yacl::MakeUint128(0, 1);
        uint128_t mask127 = yacl::MakeUint128(0x8000000000000000, 0);
        uint128_t r = yacl::MakeUint128(0, 0x0000000000000087);

        for (size_t i = 0; i < 128; ++i) {
            if ((b >> i) & mask1) {
                z = z ^ v;
            }

            if (v & mask127) {
                v = v << 1;
                v = v ^ r;
            } else {
                v = v << 1;
            }
        }

        return z;
    }

//}  // namespace

    Galois128::Galois128(uint64_t a, uint64_t b) {
#ifdef __x86_64__
        if (yacl::hasAVX2()) {
            value_ = yacl::block(a, b);
        } else {
#endif
            value_ = yacl::MakeUint128(a, b);
#ifdef __x86_64__
        }
#endif
    }

    Galois128::Galois128(const uint128_t b) {
#ifdef __x86_64__
        if (yacl::hasAVX2()) {
            std::pair<uint64_t, uint64_t> b64 = yacl::DecomposeUInt128(b);
            value_ = yacl::block(b64.first, b64.second);
        } else {
#endif
            value_ = b;
#ifdef __x86_64__
        }
#endif
    }

    Galois128 Galois128::Mul(const Galois128& rhs) const {
#ifdef __x86_64__
        if (yacl::hasAVX2()) {
            yacl::block xy1, xy2;

            mm_gf128Mul(std::get<yacl::block>(value_),
                        std::get<yacl::block>(rhs.value_), xy1, xy2);

            return Galois128(mm_gf128Reduce(xy1, xy2));
        } else {
#endif
            uint128_t z = cc_gf128Mul(std::get<uint128_t>(value_),
                                      std::get<uint128_t>(rhs.value_));

            return Galois128(z);
#ifdef __x86_64__
        }
#endif
    }

    Galois128 Galois128::Pow(std::uint64_t i) const {
        Galois128 pow2(*this);
        Galois128 zeroblock(0, 0);

        if (std::memcmp(pow2.data(), zeroblock.data(), 16) == 0)
            return Galois128(0, 0);

        Galois128 s(0, 1);
        while (i) {
            if (i & 1) {
                // s = 1 * i_0 * x^{2^{1}} * ... * i_j x^{2^{j+1}}
                s = s.Mul(pow2);
            }

            // pow2 = x^{2^{j+1}}
            pow2 = pow2.Mul(pow2);
            i >>= 1;
        }

        return s;
    }

    Galois128 Galois128::Inv() const {
        /* calculate el^{-1} as el^{2^{128}-2}. the addition chain below
           requires 142 mul/sqr operations total. */
        Galois128 a = *this;

        Galois128 result(0, 0);
        for (int64_t i = 0; i <= 6; ++i) {
            /* entering the loop a = el^{2^{2^i}-1} */
            Galois128 b(a);
            for (int64_t j = 0; j < (1 << i); ++j) {
                b = b * b;
            }
            /* after the loop b = a^{2^i} = el^{2^{2^i}*(2^{2^i}-1)} */
            a = a * b;
            /* now a = el^{2^{2^{i+1}}-1} */

            if (i == 0) {
                result = b;
            } else {
                result = result * b;
            }
        }

        YACL_ENFORCE(Mul(result).get<uint128_t>(0) == yacl::MakeUint128(0, 1));

        /* now result = el^{2^128-2} */
        return result;
    }


#ifdef __x86_64__
    void mm_gf64Mul(uint64_t x, uint64_t y, yacl::block& xy) {
        xy = _mm_clmulepi64_si128(yacl::block(x), yacl::block(y), (int)0x00);
    }

    uint64_t mm_gf64Reduce(const yacl::block& x) {
        auto xx = x.as<uint64_t>();
        static const constexpr std::uint64_t mod = 0b00011011;

        auto tmp = _mm_clmulepi64_si128(yacl::block(xx[1]), yacl::block(mod), 0x00);
        auto tt = yacl::block(tmp).as<uint64_t>();

        return (tt[1] << 4) ^ (tt[1] << 3) ^ (tt[1] << 1) ^ tt[1] ^ xx[0] ^ tt[0];
    }
#endif

// Multiplication in GF2^64, Reference
// The Galois/Counter Mode of Operation (GCM)
// https://csrc.nist.rip/groups/ST/toolkit/BCM/documents/proposedmodes/gcm/gcm-revised-spec.pdf
// P9 Algorithm 1 Multiplication in GF2^128
    uint64_t cc_gf64Mul(const uint64_t a, const uint64_t b) {
        uint64_t z = 0;
        uint64_t v = a;

        uint64_t mask1 = 1;
        uint64_t mask63 = 0x8000000000000000;
        uint64_t r = 0x000000000000001B;

        for (size_t i = 0; i < 64; ++i) {
            if ((b >> i) & mask1) {
                z = z ^ v;
            }

            if (v & mask63) {
                v = v << 1;
                v = v ^ r;
            } else {
                v = v << 1;
            }
        }

        return z;
    }

//}  // namespace

    Galois64 Galois64::Mul(const Galois64& rhs) const {
#ifdef __x86_64__
        if (yacl::hasAVX2()) {
            yacl::block xy;

            mm_gf64Mul(value_, rhs.value_, xy);

            return Galois64(mm_gf64Reduce(xy));
        } else {
#endif
            uint64_t z = cc_gf64Mul(value_, rhs.value_);

            return Galois64(z);
#ifdef __x86_64__
        }
#endif
    }

    Galois64 &Galois64::operator=(const Galois64& other) {
        this->value_ = other.value_;
        return *this;
    }

    Galois64 Galois64::Pow(std::uint64_t i) const {
        Galois64 pow2(*this);
        Galois64 zeroblock(0);

        if (std::memcmp(pow2.data(), zeroblock.data(), 8) == 0)
            return Galois64(0);

        Galois64 s(1);
        while (i) {
            if (i & 1) {
                // s = 1 * i_0 * x^{2^{1}} * ... * i_j x^{2^{j+1}}
                s = s.Mul(pow2);
            }

            // pow2 = x^{2^{j+1}}
            pow2 = pow2.Mul(pow2);
            i >>= 1;
        }

        return s;
    }

    Galois64 Galois64::Inv() const {
        /* calculate el^{-1} as el^{2^{128}-2}. the addition chain below
           requires 142 mul/sqr operations total. */
        Galois64 a = *this;

        Galois64 result(0);
        for (int64_t i = 0; i <= 6; ++i) {
            /* entering the loop a = el^{2^{2^i}-1} */
            Galois64 b(a);
            for (int64_t j = 0; j < (1 << i); ++j) {
                b = b * b;
            }
            /* after the loop b = a^{2^i} = el^{2^{2^i}*(2^{2^i}-1)} */
            a = a * b;
            /* now a = el^{2^{2^{i+1}}-1} */

            if (i == 0) {
                result = b;
            } else {
                result = result * b;
            }
        }

        YACL_ENFORCE(Mul(result).get<uint64_t>(0) == 1);

        /* now result = el^{2^128-2} */
        return result;
    }

}  // namespace psi::okvs

namespace std {

    std::ostream& operator<<(std::ostream& os, spu::mpc::cheetah::Galois128 x) {
        return os << absl::BytesToHexString(
                absl::string_view((const char*)x.data(), 16));
    }

    std::ostream& operator<<(std::ostream& os, spu::mpc::cheetah::Galois64 x) {
        return os << absl::BytesToHexString(
                absl::string_view((const char*)x.data(), 8));
    }

}  // namespace std