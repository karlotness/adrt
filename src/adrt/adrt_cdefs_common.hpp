/*
 * Copyright (c) 2022 Karl Otness, Donsub Rim
 * All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ADRT_CDEFS_COMMON_H
#define ADRT_CDEFS_COMMON_H

#include <cstddef>
#include <array>
#include <type_traits>
#include <limits>
#include <cassert>

#ifdef _OPENMP
#define ADRT_OPENMP(def) _Pragma(def)
#else
#define ADRT_OPENMP(def)
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define ADRT_RESTRICT __restrict
#else
#define ADRT_RESTRICT
#endif

namespace adrt {

    using std::size_t;

    int num_iters(size_t shape);

    inline namespace _literals {
        constexpr size_t operator"" _uz(unsigned long long val) {
            return static_cast<size_t>(val);
        }
    }

    namespace _const {

        // Constants as function templates standing in for C++14 variable templates
        template<typename scalar>
        constexpr scalar pi() {
            static_assert(std::is_floating_point<scalar>::value, "Float constants only available for floating point types");
            return static_cast<scalar>(3.141592653589793238462643383279502884L);
        }

        template<typename scalar>
        constexpr scalar pi_2() {
            static_assert(std::is_floating_point<scalar>::value, "Float constants only available for floating point types");
            return static_cast<scalar>(1.570796326794896619231321691639751442L);
        }

        template<typename scalar>
        constexpr scalar pi_4() {
            static_assert(std::is_floating_point<scalar>::value, "Float constants only available for floating point types");
            return static_cast<scalar>(0.785398163397448309615660845819875721L);
        }

        template<typename scalar>
        constexpr scalar pi_8() {
            static_assert(std::is_floating_point<scalar>::value, "Float constants only available for floating point types");
            return static_cast<scalar>(0.392699081698724154807830422909937861L);
        }

        template<typename scalar>
        constexpr scalar sqrt2() {
            static_assert(std::is_floating_point<scalar>::value, "Float constants only available for floating point types");
            return static_cast<scalar>(1.414213562373095048801688724209698079L);
        }

        template<typename scalar>
        constexpr scalar sqrt2_2() {
            static_assert(std::is_floating_point<scalar>::value, "Float constants only available for floating point types");
            return static_cast<scalar>(0.707106781186547524400844362104849039L);
        }

        template<typename scalar>
        constexpr typename std::enable_if<std::numeric_limits<scalar>::digits < std::numeric_limits<size_t>::digits, size_t>::type largest_consecutive_float_size_t() {
            static_assert(std::is_floating_point<scalar>::value, "Must specify a float type for largest size_t computation");
            static_assert(std::numeric_limits<scalar>::is_iec559 && std::numeric_limits<scalar>::radix == 2, "Our computation for largest consecutive size_t requires standard float");
            return 1_uz << std::numeric_limits<scalar>::digits;
        }

        template<typename scalar>
        constexpr typename std::enable_if<std::numeric_limits<scalar>::digits >= std::numeric_limits<size_t>::digits, size_t>::type largest_consecutive_float_size_t() {
            static_assert(std::is_floating_point<scalar>::value, "Must specify a float type for largest size_t computation");
            static_assert(std::numeric_limits<scalar>::is_iec559 && std::numeric_limits<scalar>::radix == 2, "Our computation for largest consecutive size_t requires standard float");
            return std::numeric_limits<size_t>::max();
        }

        constexpr bool openmp_enabled() {
            #ifdef _OPENMP
            return true;
            #else
            return false;
            #endif
        }

    } // end namespace adrt::_const

    namespace _common {

        // Template computing a logical and of its parameters (like C++17's std::conjunction)
        template<typename... terms>
        struct conjunction : std::true_type {};

        template<typename term, typename... terms>
        struct conjunction<term, terms...> : std::conditional<bool(term::value), adrt::_common::conjunction<terms...>, std::false_type>::type {};

        // Simplified version of C++14's std::index_sequence
        template <size_t... Idx>
        struct index_sequence {};

        template<size_t N, size_t... Build_Idx>
        struct _impl_make_index_sequence : adrt::_common::_impl_make_index_sequence<N - 1, N - 1, Build_Idx...> {};

        template<size_t... Build_Idx>
        struct _impl_make_index_sequence<0, Build_Idx...> {
            using type = adrt::_common::index_sequence<Build_Idx...>;
        };

        // Simplified version of C++14's std::make_index_sequence
        template<size_t I>
        using make_index_sequence = typename adrt::_common::_impl_make_index_sequence<I>::type;

        // Similar to (but simpler than) C++17's std::optional
        template <typename V>
        class Optional {
            static_assert(std::is_trivially_destructible<V>::value, "Optional<V> may only be used with trivially-destructible types.");

            struct Empty {};

            bool ok;
            union {
                V val;
                Empty none;
            };

        public:
            Optional(): ok(false), none() {}

            Optional(V value): ok(true), val(value) {}

            bool has_value() const {
                return ok;
            }

            V &operator*() {
                assert(has_value());
                return val;
            }

            const V &operator*() const {
                assert(has_value());
                return val;
            }

            explicit operator bool() const {
                return has_value();
            }
        };

        adrt::_common::Optional<size_t> mul_check(size_t a, size_t b);

        adrt::_common::Optional<size_t> shape_product(const size_t *shape, size_t n);

        template<size_t N>
        adrt::_common::Optional<size_t> shape_product(const std::array<size_t, N> &shape) {
            return adrt::_common::shape_product(shape.data(), shape.size());
        }

        inline size_t floor_div(size_t val, size_t d) {
            assert(d != 0u);
            // Only for non-negative values
            return val / d;
        }

        inline size_t floor_div2(size_t val) {
            return adrt::_common::floor_div(val, 2_uz);
        }

        inline size_t ceil_div(size_t val, size_t d) {
            assert(d != 0u);
            // Only for non-negative values
            return (val / d) + (val % d == 0_uz ? 0_uz : 1_uz);
        }

        inline size_t ceil_div2(size_t val) {
            return adrt::_common::ceil_div(val, 2_uz);
        }

        template<typename scalar>
        scalar clamp(scalar v, scalar lo, scalar hi) {
            assert(lo <= hi);
            if(v < lo) {
                return lo;
            }
            else if (v > hi) {
                return hi;
            }
            return v;
        }

        template<typename scalar>
        scalar lerp_pos_to_neg(scalar a, scalar t) {
            static_assert(std::is_floating_point<scalar>::value, "Interpolation requires floating point");
            return a * (static_cast<scalar>(2) * (static_cast<scalar>(0.5L) - t));
        }

        template<typename scalar>
        scalar lerp_same_signs(scalar a, scalar b, scalar t) {
            static_assert(std::is_floating_point<scalar>::value, "Interpolation requires floating point");
            assert((a <= 0 && b <= 0) || (a >= 0 && b >= 0));
            // a and b have same sign, subtraction won't magnify
            return a + t * (b - a);
        }

        // Implementation struct: unrolls compute_strides loop
        template<size_t N, size_t I>
        struct _impl_compute_strides {
            static inline size_t compute_strides(const std::array<size_t, N> &shape_in, std::array<size_t, N> &strides_out) {
                static_assert(I < N && I > 0u, "Index out of range. Do not use this template manually!");
                const size_t step_size = adrt::_common::_impl_compute_strides<N, I + 1>::compute_strides(shape_in, strides_out);
                std::get<I>(strides_out) = step_size;
                return step_size * std::get<I>(shape_in);
            }
        };

        template<size_t N>
        struct _impl_compute_strides<N, 0> {
            static inline std::array<size_t, N> compute_strides(const std::array<size_t, N> &shape_in) {
                static_assert(N > 0u, "Strides to compute must have at least one dimension");
                std::array<size_t, N> strides_out;
                const size_t step_size = adrt::_common::_impl_compute_strides<N, 1>::compute_strides(shape_in, strides_out);
                std::get<0>(strides_out) = step_size;
                return strides_out;
            }
        };

        template<size_t N>
        struct _impl_compute_strides<N, N> {
            static inline size_t compute_strides(const std::array<size_t, N>&, std::array<size_t, N>&) {
                static_assert(N > 0u, "Strides to compute must have at least one dimension");
                // Terminate recursion with initial stride of 1
                return 1_uz;
            }
        };

        // Implementation struct: unrolls array_stride_access loop
        template<size_t N, size_t I>
        struct _impl_array_stride_access {
            template<typename T, typename... Idx>
            static inline size_t compute_offset(const std::array<size_t, N> &strides, T idx, Idx... idxs) {
                static_assert(I < N, "Index out of range. Do not use this template manually!");
                static_assert(sizeof...(idxs) == N - I - 1_uz, "Parameters unpacked incorrectly. Do not use this template manually!");
                static_assert(std::is_same<size_t, T>::value, "All indexing arguments should be size_t");
                return (std::get<I>(strides) * idx) + adrt::_common::_impl_array_stride_access<N, I + 1>::compute_offset(strides, idxs...);
            }
        };

        template<size_t N>
        struct _impl_array_stride_access<N, N> {
            static inline size_t compute_offset(const std::array<size_t, N>&) {
                static_assert(N > 0u, "Array must have at least one dimension");
                // Terminate recursion, initialize accumulator to zero
                return 0_uz;
            }
        };

        template<size_t N>
        inline std::array<size_t, N> compute_strides(const std::array<size_t, N> &shape_in) {
            static_assert(N > 0u, "Strides to compute must have at least one dimension");
            #ifndef NDEBUG
            {
                // If asserts enabled, check that shapes are nonzero
                for(size_t i = 0; i < N; ++i) {
                    assert(shape_in[i] > 0u);
                }
            }
            #endif
            return adrt::_common::_impl_compute_strides<N, 0>::compute_strides(shape_in);
        }

        template <typename scalar, size_t N, typename... Idx>
        inline scalar& array_stride_access(scalar *const buf, const std::array<size_t, N> &strides, Idx... idxs) {
            static_assert(N > 0u, "Array must have at least one dimension");
            static_assert(sizeof...(idxs) == N, "Must provide N array indices");
            static_assert(adrt::_common::conjunction<std::is_same<size_t, Idx>...>::value, "All indexing arguments should be size_t");
            assert(buf);
            const size_t offset = adrt::_common::_impl_array_stride_access<N, 0>::compute_offset(strides, idxs...);
            return buf[offset];
        }

        template <typename scalar, size_t N, typename... Idx>
        inline scalar& array_access(scalar *const buf, const std::array<size_t, N> &shape, Idx... idxs) {
            #ifndef NDEBUG
            {
                // If asserts enabled, check array bounds
                const std::array<size_t, N> idx {idxs...};
                for(size_t i = 0; i < N; ++i) {
                    assert(idx[i] < shape[i]);
                }
            }
            #endif
            return adrt::_common::array_stride_access(buf, adrt::_common::compute_strides(shape), idxs...);
        }

    } // end namespace adrt::_common

    namespace _assert {
        template <size_t NA, size_t NB>
        bool same_total_size(const std::array<size_t, NA> &a, const std::array<size_t, NB> &b) {
            static_assert(NA > 0u, "Must have at least one entry in array a");
            static_assert(NB > 0u, "Must have at least one entry in array b");
            const adrt::_common::Optional<size_t> size_a = adrt::_common::shape_product(a);
            const adrt::_common::Optional<size_t> size_b = adrt::_common::shape_product(b);
            return size_a.has_value() && size_b.has_value() && (*size_a == *size_b);
        }
    } // end namespace adrt::_assert

} // end namespace adrt

#endif //ADRT_CDEFS_COMMON_H
