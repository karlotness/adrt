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
#define Py_LIMITED_API 0x03040000
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include "numpy/arrayobject.h"

template <typename adrt_scalar, typename adrt_shape>
inline adrt_scalar& adrt_array_3d_access(adrt_scalar *const buf, const adrt_shape shape[3],
                                         const adrt_shape plane, const adrt_shape row, const adrt_shape col) {
    return buf[(shape[1] * shape[2]) * plane + shape[2] * row + col];
}

template <typename adrt_scalar, typename adrt_shape>
inline adrt_scalar& adrt_array_5d_mod_access(adrt_scalar *const buf, const adrt_shape shape[5],
                                             const adrt_shape quadrant, const adrt_shape plane,
                                             const adrt_shape row, const adrt_shape col, const adrt_shape a) {
    adrt_shape mod_access[5] = {
        quadrant % shape[0],
        plane    % shape[1],
        row      % shape[2],
        col      % shape[3],
        a        % shape[4]};
    // Fix any negative values
    for (int i = 0; i < 5; ++i) {
        if (mod_access[i] < 0) {
            mod_access[i] = shape[i] + mod_access[i];
        }
    }
    adrt_scalar &res = buf[(shape[1] * shape[2] * shape[3] * shape[4]) * mod_access[0] + \
               (shape[2] * shape[3] * shape[4]) * mod_access[1] + \
               (shape[3] * shape[4]) * mod_access[2] + \
               shape[4] * mod_access[3] + \
               mod_access[4]];
    return res;
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
adrt_shape adrt_num_iters(const adrt_shape shape[3]) {
    if(shape[1] <= 1 && shape[2] <= 1) {
        return 0;
    }
    adrt_shape num_iters = 1;
    adrt_shape segment_length = 2;
    while(segment_length < shape[1] && segment_length < shape[2] && (segment_length * 2) > segment_length) {
        ++num_iters;
        segment_length *= 2;
    }
    return num_iters;
}

#endif //ADRTC_CDEFS_COMMON_H
