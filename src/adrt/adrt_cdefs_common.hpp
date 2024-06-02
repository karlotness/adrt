/*
 * Copyright Karl Otness, Donsub Rim
 *
 * SPDX-License-Identifier: BSD-3-Clause
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
#include <algorithm>
#include <optional>
#include <utility>
#include <span>

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
        constexpr size_t operator""_uz(unsigned long long val) {
            return static_cast<size_t>(val);
        }
    }

    namespace _const {

        template<typename result, typename value>
        constexpr result as_floating_point(value val) {
            static_assert(std::is_floating_point_v<result>, "Float constants only available for floating point types");
            static_assert(std::is_same_v<value, long double>, "Float value should be specified as long double");
            return static_cast<result>(val);
        }

        template<typename scalar>
        inline constexpr scalar pi_2 = adrt::_const::as_floating_point<scalar>(1.570796326794896619231321691639751442L);

        template<typename scalar>
        inline constexpr scalar pi_4 = adrt::_const::as_floating_point<scalar>(0.785398163397448309615660845819875721L);

        template<typename scalar>
        inline constexpr scalar pi_8 = adrt::_const::as_floating_point<scalar>(0.392699081698724154807830422909937861L);

        template<typename scalar>
        inline constexpr scalar sqrt2_2 = adrt::_const::as_floating_point<scalar>(0.707106781186547524400844362104849039L);

        template<typename scalar>
        constexpr std::enable_if_t<(std::numeric_limits<scalar>::digits < std::numeric_limits<size_t>::digits), size_t> _largest_consecutive_float_size_t() {
            static_assert(std::is_floating_point_v<scalar>, "Must specify a float type for largest size_t computation");
            static_assert(std::numeric_limits<scalar>::is_iec559 && std::numeric_limits<scalar>::radix == 2, "Our computation for largest consecutive size_t requires standard float");
            static_assert(std::numeric_limits<scalar>::max_exponent >= std::numeric_limits<scalar>::digits, "Max exponent is too small to cover digits");
            return 1_uz << std::numeric_limits<scalar>::digits;
        }

        template<typename scalar>
        constexpr std::enable_if_t<(std::numeric_limits<scalar>::digits >= std::numeric_limits<size_t>::digits), size_t> _largest_consecutive_float_size_t() {
            static_assert(std::is_floating_point_v<scalar>, "Must specify a float type for largest size_t computation");
            static_assert(std::numeric_limits<scalar>::is_iec559 && std::numeric_limits<scalar>::radix == 2, "Our computation for largest consecutive size_t requires standard float");
            static_assert(std::numeric_limits<scalar>::max_exponent >= std::numeric_limits<size_t>::digits - 1, "Max exponent is too small to cover digits");
            return std::numeric_limits<size_t>::max();
        }

        template<typename scalar>
        inline constexpr size_t largest_consecutive_float_size_t = adrt::_const::_largest_consecutive_float_size_t<scalar>();

        // DOC ANCHOR: adrt.core.threading_enabled
        #ifdef _OPENMP
        inline constexpr bool openmp_enabled = true;
        #else
        inline constexpr bool openmp_enabled = false;
        #endif

    } // end namespace adrt::_const

    namespace _common {

        std::optional<size_t> mul_check(size_t a, size_t b);

        std::optional<size_t> shape_product(std::span<const size_t> shape);

        template<size_t I, typename T, size_t Extent>
        constexpr decltype(auto) get(std::span<T, Extent> span) {
            static_assert(Extent != std::dynamic_extent, "Span must have statically-known extent");
            static_assert(I < Extent, "Span access is out of bounds");
            return span[I];
        }

        template<size_t I, typename T>
        constexpr decltype(auto) get(T &&obj) {
            return std::get<I>(std::forward<T>(obj));
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

        // Similar to C++20's std::lerp
        template<typename scalar>
        scalar lerp(scalar a, scalar b, scalar t) {
            static_assert(std::is_floating_point_v<scalar>, "Interpolation requires floating point");
            if((a <= static_cast<scalar>(0) && b <= static_cast<scalar>(0)) || (a >= static_cast<scalar>(0) && b >= static_cast<scalar>(0))) {
                // a and b have same sign, subtraction won't magnify
                return a + t * (b - a);
            }
            else {
                // a and b have opposite signs, subtraction would magnify
                return t * b + (static_cast<scalar>(1) - t) * a;
            }
        }

        template<size_t N, size_t... Ints>
        inline std::array<size_t, N> compute_strides(std::span<const size_t, N> shape_in, std::index_sequence<Ints...>) {
            static_assert(N > 0u, "Strides to compute must have at least one dimension");
            static_assert(std::is_same_v<std::index_sequence<Ints...>, std::make_index_sequence<N - 1_uz>>, "Invalid indexing pack. Do not call this overload directly!");
            assert(std::all_of(std::cbegin(shape_in), std::cend(shape_in), [](size_t v){return v > 0u;}));
            std::array<size_t, N> strides_out;
            adrt::_common::get<N - 1_uz>(strides_out) = 1_uz;
            (static_cast<void>(adrt::_common::get<N - Ints - 2_uz>(strides_out) = adrt::_common::get<N - Ints - 1_uz>(shape_in) * adrt::_common::get<N - Ints - 1_uz>(strides_out)), ...);
            return strides_out;
        }

        template<size_t N>
        inline std::array<size_t, N> compute_strides(std::span<const size_t, N> shape_in) {
            static_assert(N > 0u, "Strides to compute must have at least one dimension");
            return adrt::_common::compute_strides(shape_in, std::make_index_sequence<N - 1_uz>{});
        }

        template <size_t... Ints, typename scalar, size_t N, typename... Idx>
        inline scalar& array_stride_access(std::index_sequence<Ints...>, scalar *const buf, std::span<const size_t, N> strides, Idx... idxs) {
            static_assert(N > 0u, "Array must have at least one dimension");
            static_assert(sizeof...(idxs) == N, "Must provide N array indices");
            static_assert(std::is_same_v<std::index_sequence<Ints...>, std::make_index_sequence<N>>, "Invalid indexing pack. Do not call this overload directly!");
            static_assert(std::conjunction_v<std::is_same<size_t, Idx>...>, "All indexing arguments should be size_t");
            assert(buf);
            return buf[(... + (adrt::_common::get<Ints>(strides) * idxs))];
        }

        template <typename scalar, typename... Idx>
        inline scalar& array_stride_access(scalar *const buf, std::span<const size_t, sizeof...(Idx)> strides, Idx... idxs) {
            return adrt::_common::array_stride_access(std::make_index_sequence<sizeof...(idxs)>{}, buf, strides, idxs...);
        }

        template <typename scalar, typename... Idx>
        inline scalar& array_access(scalar *const buf, std::span<const size_t, sizeof...(Idx)> shape, Idx... idxs) {
            assert(std::equal(std::cbegin(shape), std::cend(shape), std::cbegin(std::array<size_t, sizeof...(idxs)>{idxs...}), [](size_t shape_v, size_t idx_v){return idx_v < shape_v;}));
            return adrt::_common::array_stride_access(buf, adrt::_common::compute_strides(shape), idxs...);
        }

    } // end namespace adrt::_common

    namespace _assert {
        template <typename A, typename B>
        bool same_total_size(A &&a, B &&b) {
            const std::optional<size_t> size_a = adrt::_common::shape_product(std::forward<A>(a));
            const std::optional<size_t> size_b = adrt::_common::shape_product(std::forward<B>(b));
            return size_a.has_value() && size_b.has_value() && (*size_a == *size_b);
        }
    } // end namespace adrt::_assert

} // end namespace adrt

#endif //ADRT_CDEFS_COMMON_H
