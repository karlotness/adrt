/*
 * Copyright (C) 2020 Karl Otness, Donsub Rim
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
#ifndef ADRTC_CDEFS_COMMON_H
#define ADRTC_CDEFS_COMMON_H

#define PY_SSIZE_T_CLEAN
#ifndef Py_LIMITED_API
#error "Must use limited API"
#endif
#include "Python.h"

#ifndef NPY_NO_DEPRECATED_API
#error "Must use NumPy API versioning"
#endif
#include "numpy/arrayobject.h"

#include <array>
#include <limits>

template <typename adrt_shape, size_t N>
static std::array<adrt_shape, N> adrt_compute_strides(const std::array<adrt_shape, N> &shape_in) {
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
static adrt_scalar& adrt_array_stride_access(adrt_scalar *const buf, const std::array<adrt_shape, N> &strides,
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
static adrt_scalar& adrt_array_access(adrt_scalar *const buf, const std::array<adrt_shape, N> &shape,
                                            const Idx... idxs) {
    const std::array<adrt_shape, N> strides = adrt_compute_strides(shape);
    return adrt_array_stride_access(buf, strides, idxs...);
}

template <typename adrt_shape>
inline adrt_shape adrt_floor_div2(const adrt_shape val) {
    return val / 2;
}

template <typename adrt_shape>
inline adrt_shape adrt_ceil_div2(const adrt_shape val) {
    adrt_shape div = val / 2;
    adrt_shape rem = val % 2;
    if(rem > 0) {
        return div + 1;
    }
    return div;
}

template <typename adrt_shape>
static adrt_shape adrt_num_iters(const adrt_shape shape) {
    if(shape <= 1) {
        return 0;
    }
    const adrt_shape shape_max_half = std::numeric_limits<adrt_shape>::max() / 2;
    adrt_shape num_iters = 1;
    adrt_shape segment_length = 2;
    while(segment_length < shape && segment_length <= shape_max_half) {
        ++num_iters;
        segment_length *= 2;
    }
    // If shape is larger than the max power of two fitting in adrt_shape
    // we need one more doubling
    if(segment_length < shape) {
        ++num_iters;
    }
    return num_iters;
}

#endif //ADRTC_CDEFS_COMMON_H