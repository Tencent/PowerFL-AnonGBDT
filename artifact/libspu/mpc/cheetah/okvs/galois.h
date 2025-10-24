

#ifndef SPU_GALOIS_H
#define SPU_GALOIS_H

#pragma once

#include <ostream>
#include <variant>

#include "absl/strings/escaping.h"
#include "spdlog/spdlog.h"
#include "yacl/base/block.h"
#include "yacl/base/exception.h"
#include "yacl/base/int128.h"

namespace spu::mpc::cheetah {

    // Galois field 2^128
    // polynoimal : x^128+x^7+x^2+x+1

    using Galois128Type = std::variant<yacl::block, uint128_t>;

    class Galois128 {
    public:
        Galois128(uint64_t a, uint64_t b);
        explicit Galois128(uint64_t v) : Galois128(0, v) {}

        Galois128(const Galois128& v) { value_ = v.value_; }

#ifdef __x86_64__
        explicit Galois128(const yacl::block& b) { value_ = b; }
#endif

        explicit Galois128(const uint128_t b);

        Galois128& operator=(const Galois128&) = default;

        Galois128 Mul(const Galois128& rhs) const;
        Galois128 Pow(std::uint64_t i) const;
        Galois128 Inv() const;

        inline Galois128 operator*(const Galois128& rhs) const {
            return Galois128(Mul(rhs));
        }

        inline Galois128 operator*(const uint128_t& rhs) const {
            return Galois128(Mul(Galois128(rhs)));
        }

        inline Galois128 operator*(const uint64_t& rhs) const {
            return Galois128(Mul(Galois128(rhs)));
        }

        const uint8_t* data() { return (const uint8_t*)&value_; }

        template <typename T>
        typename std::enable_if<std::is_standard_layout<T>::value &&
                                std::is_trivial<T>::value && (sizeof(T) <= 16) &&
                                (16 % sizeof(T) == 0),
                std::array<T, 16 / sizeof(T)> >::type
        get() {
            std::array<T, 16 / sizeof(T)> output;
            std::memcpy(output.data(), data(), 16);
            return output;
        }

        template <typename T>
        typename std::enable_if<std::is_standard_layout<T>::value &&
                                std::is_trivial<T>::value && (sizeof(T) <= 16) &&
                                (16 % sizeof(T) == 0),
                T>::type
        get(size_t index) {
            YACL_ENFORCE(index < 16 / sizeof(T));

            T output;
            std::memcpy(&output, data() + sizeof(T) * index, sizeof(T));
            return output;
        }

    private:
        Galois128Type value_;
    };



    // Galois field 2^64
    // polynoimal : x^128+x^4+x^3+x+1

    using Galois64Type = uint64_t;

    class Galois64 {
    public:
        explicit Galois64(uint64_t a) : value_(a) {}

        Galois64(const Galois64& v) { value_ = v.value_; }

        Galois64& operator=(const Galois64& other);

        Galois64 Mul(const Galois64& rhs) const;
        Galois64 Pow(std::uint64_t i) const;
        Galois64 Inv() const;

        inline Galois64 operator*(const Galois64& rhs) const {
            return Galois64(Mul(rhs));
        }

        inline Galois64 operator*(const uint128_t& rhs) const {
            return Galois64(Mul(Galois64(rhs)));
        }

        inline Galois64 operator*(const uint64_t& rhs) const {
            return Galois64(Mul(Galois64(rhs)));
        }

        const uint8_t* data() { return (const uint8_t*)&value_; }

        template <typename T>
        typename std::enable_if<std::is_standard_layout<T>::value &&
                                std::is_trivial<T>::value && (sizeof(T) <= 16) &&
                                (16 % sizeof(T) == 0),
                std::array<T, 16 / sizeof(T)> >::type
        get() {
            std::array<T, 16 / sizeof(T)> output;
            std::memcpy(output.data(), data(), 16);
            return output;
        }

        template <typename T>
        typename std::enable_if<std::is_standard_layout<T>::value &&
                                std::is_trivial<T>::value && (sizeof(T) <= 16) &&
                                (16 % sizeof(T) == 0),
                T>::type
        get(size_t index) {
            YACL_ENFORCE(index < 16 / sizeof(T));

            T output;
            std::memcpy(&output, data() + sizeof(T) * index, sizeof(T));
            return output;
        }

    private:
        Galois64Type value_;
    };




    uint128_t cc_gf128Mul(const uint128_t a, const uint128_t b);

    uint64_t cc_gf64Mul(const uint64_t a, const uint64_t b);

}  // namespace psi::okvs

namespace std {

    std::ostream& operator<<(std::ostream& os, spu::mpc::cheetah::Galois128 x);

    std::ostream& operator<<(std::ostream& os, spu::mpc::cheetah::Galois64 x);
}

#endif //SPU_GALOIS_H
