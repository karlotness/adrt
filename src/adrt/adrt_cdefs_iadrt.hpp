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
#ifndef ADRTC_CDEFS_IADRT_H
#define ADRTC_CDEFS_IADRT_H

#include "adrt_cdefs_common.hpp"
#include <array>
#include <utility>

template <typename adrt_scalar, typename adrt_shape>
static bool iadrt_impl(const adrt_scalar *const data, const unsigned char
ndim, const adrt_shape *const shape, const int base_iter_start, const int base_iter_end, adrt_scalar *const out, const adrt_shape *const base_output_shape) {

    // Shape (plane, quadrant, row, col)
    const std::array<adrt_shape, 4> corrected_shape =
        {(ndim > 3 ? shape[0] : 1),
         (ndim > 3 ? shape[1] : shape[0]),
         (ndim > 3 ? shape[2] : shape[1]),
         (ndim > 3 ? shape[3] : shape[2])};

    // Output shape (plane, row, col)
    const std::array<adrt_shape, 4> output_shape =
        {(ndim > 3 ? base_output_shape[0] : 1),
         (ndim > 3 ? base_output_shape[1] : base_output_shape[0]),
         (ndim > 3 ? base_output_shape[2] : base_output_shape[1]),
         (ndim > 3 ? base_output_shape[3] : base_output_shape[2])};

    // set orientation for input output:
    // note input orientation is ignored when iter_start == 0

    // Require that the matrix have right dimensions
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


    const size_t buf_size = corrected_shape[0] * corrected_shape[1] * corrected_shape[2] * corrected_shape[3];

    // Allocate two of these buffers
    adrt_scalar *const aux = PyMem_New(adrt_scalar, 2 * buf_size);
    if(!aux) {
        PyErr_NoMemory();
        return false;
    }


    // NO PYTHON API BELOW THIS POINT
    Py_BEGIN_ALLOW_THREADS;

    const adrt_shape num_iters = adrt_num_iters(corrected_shape[3]);

    int iter_end   = (base_iter_end < 0? base_iter_end + num_iters + 1: base_iter_end);
    int iter_start = (base_iter_start < 0? base_iter_start + num_iters + 1: base_iter_start);

    adrt_scalar *curr = aux;
    adrt_scalar *prev = aux + buf_size;

    // Order: quadrant, plane, row, l, col
    std::array<adrt_shape, 5> prev_shape =
        {corrected_shape[1],   // 4
         corrected_shape[0],   // planes
         1,
         corrected_shape[3],   // N
         corrected_shape[2]    // 2*N - 1
         };
    std::array<adrt_shape, 5> curr_shape = prev_shape;

    // Copy results to destination buffer
    for(adrt_shape quadrant = 0; quadrant < 4; ++quadrant) {
        for(adrt_shape plane = 0; plane < prev_shape[1]; ++plane) {
            for(adrt_shape d = 0; d < prev_shape[3]; ++d) {
                for(adrt_shape a = 0; a < prev_shape[4]; ++a) {
                    adrt_shape acc_d = d;
                    adrt_shape acc_a = a;

                    const adrt_scalar val = adrt_array_access(data,
                                corrected_shape, plane, quadrant, a, d);
                    adrt_array_access(prev, prev_shape, quadrant, plane, 0, acc_d, acc_a) = val;
                    adrt_array_access(curr, prev_shape, quadrant, plane, 0, acc_d, acc_a) = 0.0;
                }
            }
        }
    }

    for(adrt_shape i = 1; i <= iter_start; ++i) {
        curr_shape[2] = 2 * prev_shape[2];
        curr_shape[3] = adrt_floor_div2(prev_shape[3]);

        // Swap the "curr" and "prev" buffers and shapes
        prev_shape = curr_shape;
    }


    // Outer loop over iterations (this loop must be serial)
    for(adrt_shape i = iter_start+1; i <= iter_end; ++i) {
        curr_shape[2] = 2 * prev_shape[2];
        curr_shape[3] = adrt_floor_div2(prev_shape[3]);
        for(adrt_shape rev_row = 0; rev_row < curr_shape[4]; ++rev_row) {
            const adrt_shape row = curr_shape[4] - rev_row - 1;
            // Inner loops (these loops can be parallel)
            #pragma omp parallel for collapse(3) shared(curr, prev, curr_shape, prev_shape)
            for(adrt_shape quadrant = 0; quadrant < curr_shape[0]; ++quadrant) {
            for(adrt_shape plane = 0; plane < curr_shape[1]; ++plane) {
                for(adrt_shape col = 0; col < curr_shape[3]; ++col) {
                    for(adrt_shape l = 0; l < curr_shape[2]; ++l) {
                        adrt_scalar val = 0;
                        adrt_scalar two = 2.0;
                        const adrt_shape prev_l = adrt_floor_div2(l);
                        if(l % 2 == 0) {
                            // l + 1 odd
                            val += adrt_array_access(prev, prev_shape, quadrant, plane, prev_l, 2 * col, row);
                            if(row + 1 < prev_shape[4] && 2 * col + 1 < prev_shape[3]) {
                                val -= adrt_array_access(prev, prev_shape, quadrant, plane, prev_l, 2 * col + 1, row + 1);
                            }
                        }
                        else {
                            // l + 1 even
                            if(row + 1 + col < prev_shape[4]) {
                                if(2 * col + 1 < prev_shape[3]){
                                    val += adrt_array_access(prev, prev_shape, quadrant, plane, prev_l, 2 * col + 1, row + 1 + col);
                                }
                                val -= adrt_array_access(prev, prev_shape, quadrant, plane, prev_l, 2 * col, row + 1 + col);
                            }
                        }
                        val *= two;
                        if(row + 1 < curr_shape[4]) {
                            val += adrt_array_access(curr, curr_shape, quadrant, plane, l, col, row + 1);
                        }
                        adrt_array_access(curr, curr_shape, quadrant, plane, l,
col, row) = val;
                    }
                }
            }
        }

        }
        // Swap the "curr" and "prev" buffers and shapes
        std::swap(curr, prev);
        prev_shape = curr_shape;
    }

    // reset shape
    prev_shape[0] = corrected_shape[1];
    prev_shape[1] = corrected_shape[0];
    prev_shape[2] = 1;
    prev_shape[3] = corrected_shape[3];
    prev_shape[4] = corrected_shape[2];

    // Copy results to destination buffer
    for(adrt_shape quadrant = 0; quadrant < 4; ++quadrant) {
        for(adrt_shape plane = 0; plane < prev_shape[1]; ++plane) {
            for(adrt_shape d = 0; d < prev_shape[3]; ++d) {
                for(adrt_shape a = 0; a < prev_shape[4]; ++a) {
                    adrt_shape acc_d = d;
                    adrt_shape acc_a = a;

                    const adrt_scalar val = adrt_array_access(prev, prev_shape, quadrant, plane, 0, acc_d, acc_a);
                    adrt_array_access(out, output_shape, plane, quadrant, a, d) = val;
                }
            }
        }
    }


    // PYTHON API ALLOWED BELOW THIS POINT
    Py_END_ALLOW_THREADS;

    PyMem_Free(aux);
    return true;
}
#endif // ADRTC_CDEFS_IADRT_H
