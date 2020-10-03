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
#ifndef ADRTC_CDEFS_IADRT_H
#define ADRTC_CDEFS_IADRT_H

#include "adrt_cdefs_common.hpp"
#include <array>
#include <algorithm>

template <typename adrt_scalar, typename adrt_shape>
static bool iadrt_impl(const adrt_scalar *const data, const unsigned char ndims, const adrt_shape *const shape, adrt_scalar *const out,
                       const adrt_shape *const base_output_shape) {

    const std::array<adrt_shape, 4> corrected_shape =
        {(ndims > 3 ? shape[0] : 1),
         (ndims > 3 ? shape[1] : shape[0]),
         (ndims > 3 ? shape[2] : shape[1]),
         (ndims > 3 ? shape[3] : shape[2])};

    const std::array<adrt_shape, 3> output_shape =
        {(ndims > 3 ? base_output_shape[0] : 1),
         (ndims > 3 ? base_output_shape[1] : base_output_shape[0]),
         (ndims > 3 ? base_output_shape[2] : base_output_shape[1])};

    // Require that the matrix be square (power of two checked elsewhere)
    if(corrected_shape[2] != 2*corrected_shape[3]-1) {
        PyErr_SetString(PyExc_ValueError, "Provided array must be of dimensions (2*N - 1) x N");
        return false;
    }

    // Check that shape is sensible
    for(int i = 0; i < 4; ++i) {
        if(corrected_shape[i] <= 0) {
            PyErr_SetString(PyExc_ValueError, "Provided array must have no axes of zero size");
            return false;
        }
    }

    const size_t buf_size = corrected_shape[0] * corrected_shape[2] * corrected_shape[3];

    // Allocate two of these buffers
    adrt_scalar *const aux = PyMem_New(adrt_scalar, 2 * buf_size);
    if(!aux) {
        PyErr_NoMemory();
        return false;
    }

    // NO PYTHON API BELOW THIS POINT
    Py_BEGIN_ALLOW_THREADS;

    const adrt_shape num_iters = adrt_num_iters(corrected_shape[3]);

    adrt_scalar *curr = aux;
    adrt_scalar *prev = aux + buf_size;

    // Order: plane, row, col, l
    std::array<adrt_shape, 4> prev_shape =
        {corrected_shape[0],
         corrected_shape[2],
         corrected_shape[3],
         1};
    std::array<adrt_shape, 4> curr_shape = prev_shape;

    // Direct copy row by row
    for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
        for(adrt_shape col = 0; col < corrected_shape[3]; ++col) {
            for(adrt_shape row = 0; row < corrected_shape[2]; ++row) {
                adrt_array_access(prev, prev_shape, plane, row, col, 0)
                    = adrt_array_access(data, corrected_shape, plane, 2, row, col);
            }
        }
    }

    // Outer loop over iterations (this loop must be serial)
    for(adrt_shape i = 1; i <= num_iters; ++i) {
        curr_shape[2] = adrt_floor_div2(prev_shape[2]);
        curr_shape[3] = 2 * prev_shape[3];

        for(adrt_shape rev_row = 0; rev_row < curr_shape[1]; ++rev_row) {
            const adrt_shape row = curr_shape[1] - rev_row - 1;
            // Inner loops (these loops can be parallel)
            #pragma omp parallel for collapse(3) shared(curr, prev, curr_shape, prev_shape, i)
            for(adrt_shape plane = 0; plane < curr_shape[0]; ++plane) {
                for(adrt_shape col = 0; col < curr_shape[2]; ++col) {
                    for(adrt_shape l = 0; l < curr_shape[3]; ++l) {
                        adrt_scalar val = 0;
                        const adrt_shape prev_l = adrt_floor_div2(l);
                        if(l % 2 == 0) {
                            // l + 1 odd
                            val += adrt_array_access(prev, prev_shape, plane, row, 2 * col, prev_l);
                            if(row + 1 < prev_shape[1] && 2 * col + 1 < prev_shape[2]) {
                                val -= adrt_array_access(prev, prev_shape, plane, row + 1, 2 * col + 1, prev_l);
                            }
                        }
                        else {
                            // l + 1 even
                            if(row + 1 + col < prev_shape[1]) {
                                if(2 * col + 1 < prev_shape[2]){
                                    val += adrt_array_access(prev, prev_shape, plane, row + 1 + col, 2 * col + 1, prev_l);
                                }
                                val -= adrt_array_access(prev, prev_shape, plane, row + 1 + col, 2 * col, prev_l);
                            }
                        }
                        if(row + 1 < curr_shape[1]) {
                            val += adrt_array_access(curr, curr_shape, plane, row + 1, col, l);
                        }
                        adrt_array_access(curr, curr_shape, plane, row, col, l) = val;
                    }
                }
            }
        }

        // Swap the "curr" and "prev" buffers and shapes
        std::swap(curr, prev);
        prev_shape = curr_shape;
    }

    // Copy results to destination buffer
    for(adrt_shape plane = 0; plane < output_shape[0]; ++plane) {
        for(adrt_shape row = 0; row < output_shape[1]; ++row) {
            for(adrt_shape col = 0; col < output_shape[2]; ++col) {
                adrt_array_access(out, output_shape, plane, row, col) = adrt_array_access(prev, prev_shape, plane, row, 0, col);
            }
        }
    }

    // PYTHON API ALLOWED BELOW THIS POINT
    Py_END_ALLOW_THREADS;

    PyMem_Free(aux);
    return true;
}
#endif // ADRTC_CDEFS_IADRT_H
