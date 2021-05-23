/*
 * Copyright (c) 2020, 2021 Karl Otness, Donsub Rim
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
#include <tuple>

#ifdef _OPENMP
#define ADRT_OPENMP(def) _Pragma(def)
#else
#define ADRT_OPENMP(def)
#endif

namespace adrt {

    using std::size_t;

    int num_iters(size_t shape);

    namespace _const {
        const long double pi_4 = 0.785398163397448309615660845819875721L;
        const long double sqrt2 = 1.414213562373095048801688724209698079L;
    } // end namespace adrt::_const

    namespace _common {

        std::tuple<bool, size_t> mul_check(size_t a, size_t b);

        inline bool is_pow2(size_t val) {
            if(val == 0) {
                return false;
            }
            return !(val & (val - 1));
        }

        inline size_t floor_div2(size_t val) {
            // Only for non-negative values
            return val / 2;
        }

        inline size_t ceil_div2(size_t val) {
            // Only for non-negative values
            return (val / 2) + (val % 2);
        }

        template<size_t N>
        std::array<size_t, N> compute_strides(const std::array<size_t, N> &shape_in) {
            std::array<size_t, N> strides_out;
            size_t step_size = 1;
            for(size_t i = 0; i < N; ++i) {
                size_t idx_i = N - i - 1;
                strides_out[idx_i] = step_size;
                step_size *= shape_in[idx_i];
            }
            return strides_out;
        }

        template <typename scalar, size_t N, typename... Idx>
        scalar& array_stride_access(scalar *const buf, const std::array<size_t, N> &strides, const Idx... idxs) {
            static_assert(sizeof...(idxs) == N, "Must provide N array indices");
            const std::array<size_t, N> idx {idxs...};
            size_t acc = 0;
            for(size_t i = 0; i < N; ++i) {
                acc += strides[i] * idx[i];
            }
            return buf[acc];
        }

        template <typename scalar, size_t N, typename... Idx>
        scalar& array_access(scalar *const buf, const std::array<size_t, N> &shape, const Idx... idxs) {
            const std::array<size_t, N> strides = compute_strides(shape);
            return array_stride_access(buf, strides, idxs...);
        }

    } // end namespace adrt::_common
} // end namespace adrt

inline int adrt_num_iters(size_t shape) {
    return adrt::num_iters(shape);
}

template <typename adrt_shape, size_t N>
std::array<adrt_shape, N> adrt_compute_strides(const std::array<adrt_shape, N> &shape_in) {
    std::array<adrt_shape, N> strides_out;
    adrt_shape step_size = 1;
    for(size_t i = 0; i < N; ++i) {
        size_t idx_i = N - i - 1;
        strides_out[idx_i] = step_size;
        step_size *= shape_in[idx_i];
    }
    return strides_out;
}

template <typename adrt_scalar, typename adrt_shape, size_t N, typename... Idx>
adrt_scalar& adrt_array_stride_access(adrt_scalar *const buf, const std::array<adrt_shape, N> &strides,
                                             const Idx... idxs) {
    static_assert(sizeof...(idxs) == N, "Must provide N array indices");
    const std::array<adrt_shape, N> idx {idxs...};
    size_t acc = 0;
    for(size_t i = 0; i < N; ++i) {
        acc += strides[i] * idx[i];
    }
    return buf[acc];
}

template <typename adrt_scalar, typename adrt_shape, size_t N, typename... Idx>
adrt_scalar& adrt_array_access(adrt_scalar *const buf, const std::array<adrt_shape, N> &shape,
                                      const Idx... idxs) {
    const std::array<adrt_shape, N> strides = adrt_compute_strides(shape);
    return adrt_array_stride_access(buf, strides, idxs...);
}

inline bool adrt_is_pow2(size_t val) {
    return adrt::_common::is_pow2(val);
}

inline size_t adrt_floor_div2(size_t val) {
    // Only for non-negative values
    return adrt::_common::floor_div2(val);
}

inline size_t adrt_ceil_div2(size_t val) {
    // Only for non-negative values
    return adrt::_common::ceil_div2(val);
}

#endif //ADRT_CDEFS_COMMON_H
