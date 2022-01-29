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

#pragma once
#ifndef ADRT_CDEFS_COMMON_H
#define ADRT_CDEFS_COMMON_H

#include <cstddef>
#include <array>
#include <type_traits>

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

#ifndef NDEBUG
#include <cassert>
#define ADRT_ASSERT(cond) assert(cond);
#else
#define ADRT_ASSERT(cond)
#endif

namespace adrt {

    using std::size_t;

    int num_iters(size_t shape);

    namespace _const {

        // Constants as function templates standing in for C++14 variable templates
        template<typename scalar>
        constexpr scalar pi_4() {
            static_assert(std::is_floating_point<scalar>::value, "Float constants only available for floating point types");
            return static_cast<scalar>(0.785398163397448309615660845819875721L);
        }

        template<typename scalar>
        constexpr scalar sqrt2() {
            static_assert(std::is_floating_point<scalar>::value, "Float constants only available for floating point types");
            return static_cast<scalar>(1.414213562373095048801688724209698079L);
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

        // Simple optional type that always default-initializes its value
        // Similar to (but simpler than) C++17's std::optional
        template <typename V>
        class Optional {
            bool ok;
            V val;

        public:
            Optional(): ok{false} {}

            Optional(V value): ok{true}, val{value} {}

            bool has_value() const {
                return ok;
            }

            void set_ok(bool flag) {
                ok = flag;
            }

            V &operator*() {
                // No assertion here, could be trying to set the value
                return val;
            }

            const V &operator*() const {
                ADRT_ASSERT(has_value())
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

        inline size_t floor_div2(size_t val) {
            // Only for non-negative values
            return val / size_t{2};
        }

        inline size_t ceil_div(size_t val, size_t d) {
            ADRT_ASSERT(d != 0)
            // Only for non-negative values
            return (val / d) + (val % d == size_t{0} ? size_t{0} : size_t{1});
        }

        inline size_t ceil_div2(size_t val) {
            return adrt::_common::ceil_div(val, size_t{2});
        }

        // Implementation struct: unrolls compute_strides loop
        template<size_t N, size_t I>
        struct _impl_compute_strides {
            static inline size_t compute_strides(const std::array<size_t, N> &shape_in, std::array<size_t, N> &strides_out) {
                static_assert(I < N, "Index out of range. Do not use this template manually!");
                const size_t step_size = _impl_compute_strides<N, I + 1>::compute_strides(shape_in, strides_out);
                std::get<I>(strides_out) = step_size;
                return step_size * std::get<I>(shape_in);
            }
        };

        template<size_t N>
        struct _impl_compute_strides<N, N> {
            static inline size_t compute_strides(const std::array<size_t, N>&, std::array<size_t, N>&) {
                // Terminate recursion with initial stride of 1
                return size_t{1};
            }
        };

        // Implementation struct: unrolls array_stride_access loop
        template<size_t N, size_t I>
        struct _impl_array_stride_access {
            template<typename T, typename... Idx>
            static inline size_t compute_offset(const std::array<size_t, N> &strides, T idx, const Idx... idxs) {
                static_assert(I < N, "Index out of range. Do not use this template manually!");
                static_assert(sizeof...(idxs) == N - I - 1, "Parameters unpacked incorrectly. Do not use this template manually!");
                static_assert(std::is_same<size_t, T>::value, "All indexing arguments should be size_t");
                return (std::get<I>(strides) * idx) + _impl_array_stride_access<N, I + 1>::compute_offset(strides, idxs...);
            }
        };

        template<size_t N>
        struct _impl_array_stride_access<N, N> {
            static inline size_t compute_offset(const std::array<size_t, N>&) {
                // Terminate recursion, initialize accumulator to zero
                return size_t{0};
            }
        };

        template<size_t N>
        inline std::array<size_t, N> compute_strides(const std::array<size_t, N> &shape_in) {
            #ifndef NDEBUG
            {
                // If asserts enabled, check that shapes are nonzero
                for(size_t i = 0; i < N; ++i) {
                    ADRT_ASSERT(shape_in[i] > 0)
                }
            }
            #endif
            std::array<size_t, N> strides_out;
            _impl_compute_strides<N, 0>::compute_strides(shape_in, strides_out);
            return strides_out;
        }

        template <typename scalar, size_t N, typename... Idx>
        inline scalar& array_stride_access(scalar *const buf, const std::array<size_t, N> &strides, const Idx... idxs) {
            static_assert(sizeof...(idxs) == N, "Must provide N array indices");
            static_assert(adrt::_common::conjunction<std::is_same<size_t, Idx>...>::value, "All indexing arguments should be size_t");
            ADRT_ASSERT(buf)
            const size_t offset = _impl_array_stride_access<N, 0>::compute_offset(strides, idxs...);
            return buf[offset];
        }

        template <typename scalar, size_t N, typename... Idx>
        inline scalar& array_access(scalar *const buf, const std::array<size_t, N> &shape, const Idx... idxs) {
            #ifndef NDEBUG
            {
                // If asserts enabled, check array bounds
                const std::array<size_t, N> idx {idxs...};
                for(size_t i = 0; i < N; ++i) {
                    ADRT_ASSERT(idx[i] < shape[i])
                }
            }
            #endif
            return array_stride_access(buf, compute_strides(shape), idxs...);
        }

    } // end namespace adrt::_common

    #ifndef NDEBUG
    namespace _assert {
        template <size_t NA, size_t NB>
        bool same_total_size(const std::array<size_t, NA> &a, const std::array<size_t, NB> &b) {
            static_assert(NA > 0, "Must have at least one entry in array a");
            static_assert(NB > 0, "Must have at least one entry in array b");
            const adrt::_common::Optional<size_t> size_a = adrt::_common::shape_product(a);
            const adrt::_common::Optional<size_t> size_b = adrt::_common::shape_product(b);
            return size_a.has_value() && size_b.has_value() && (*size_a == *size_b);
        }
    } // end namespace adrt::_assert
    #endif // NDEBUG

} // end namespace adrt

#endif //ADRT_CDEFS_COMMON_H
