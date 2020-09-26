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

    // Check that shape is sensible
    for(int i = 0; i < 4; ++i) {
        if(corrected_shape[i] <= 0) {
            PyErr_SetString(PyExc_ValueError, "Provided array must have no axes of zero size");
            return false;
        }
    }

    // add extra row of buffers of size: planes*(2*N)*N
    const size_t buf_size = corrected_shape[0] * (2 * corrected_shape[3]) * corrected_shape[3];

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

    std::array<adrt_shape, 4> prev_shape =
        {corrected_shape[0],
         2 * corrected_shape[3],
         corrected_shape[3],
         1};
    std::array<adrt_shape, 4> curr_shape = {0};

    // Direct copy row by row
    for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
        for(adrt_shape col = 0; col < corrected_shape[3]; ++col) {
            // pad extra row with zeros for inverse computation
            adrt_array_access(prev, prev_shape, plane, 0, col, 0) = 0;
            for(adrt_shape row = 0; row < corrected_shape[2]; ++row) {
                adrt_array_access(prev, prev_shape, plane, row + 1, col, 0)
                    = adrt_array_access(data, corrected_shape, plane, 0, row, col);
            }
        }
    }

    // Outer loop over iterations (this loop must be serial)
    for(adrt_shape i = 1; i <= num_iters; ++i) {
        // Compute the curr_shape for the current buffer (based on prev shape)
        curr_shape[3] = adrt_floor_div2(prev_shape[3]); // stride for section

        // Inner loops (these loops can be parallel)
        #pragma omp parallel for collapse(4) default(none) shared(curr, prev, curr_shape, prev_shape, i)
        for(adrt_shape plane = 0; plane < curr_shape[0]; ++plane) {
            for(adrt_shape j = 0; j < curr_shape[4]; ++j) {
                for(adrt_shape a = 0; a < curr_shape[3]; ++a) {
                    for(adrt_shape x = 0; x < corrected_shape[1]+1; ++x) {

                        // right image
                        adrt_scalar raval =
                        adrt_array_access(prev,prev_shape,
                                                    plane, x, j, 2*a);

                        adrt_scalar rbval = 0;
                        rbval = adrt_array_access(prev, prev_shape,
                                                    plane, x, j, 2*a + 1);

                        // check the index access for x
                        const adrt_shape xb = x - a - 1;
                        if(xb >= 0 && xb < corrected_shape[1]) {
                        adrt_array_access(curr, curr_shape,
                                        plane, xb, 2*j + 1, a) = rbval - raval;
                        }

                        // left image
                        adrt_scalar lbval = adrt_array_access(
                                            prev,prev_shape, plane, x, j, 2*a);
                        // Need to check the index access for x
                        const adrt_shape xb1 = x+1;
                        adrt_scalar laval = 0;
                        if(xb1 >= 0 && xb1 < corrected_shape[1]+1) {
                            laval = adrt_array_access(
                                    prev, prev_shape,
                                    plane, xb1, j, 2*a + 1);
                        }
                        adrt_array_access(curr, curr_shape,
                                    plane, x, 2*j, a) =  lbval - laval;

                    }

                    // set entries out of scope to zero
                    for(adrt_shape y = corrected_shape[1]-a;
                                    y < corrected_shape[1]+1; y++){
                    adrt_array_access(curr, curr_shape,
                                    plane, y, 2*j+1, a) = 0;
                    }
                    adrt_array_access(curr, curr_shape,
                                  plane, 0, (2*j), a) = 0;
                }
            }

        // summation sweep from below
        for(adrt_shape an = 0; an < corrected_shape[2]; ++an) {
            adrt_scalar sumval = 0.0;
            for(adrt_shape xn = corrected_shape[1]-1; xn >=0 ; --xn) {
                sumval += adrt_array_access(curr,curr_shape,
                                                      plane, xn, 0, an);
                adrt_array_access(curr, curr_shape,
                                                plane, xn, 0, an) = sumval;
            }
        }
        }

        curr_shape[3] = 2*prev_shape[3]; // limits for section index

        // Swap the "curr" and "prev" buffers and shapes
        std::swap(curr, prev);
        prev_shape = curr_shape;
    }

    // Copy results to destination buffer
    for(adrt_shape plane = 0; plane < output_shape[0]; ++plane) {
        for(adrt_shape d = 0; d < output_shape[1]; ++d) {
            for(adrt_shape a = 0; a < output_shape[2]; ++a) {
                // re-order to match input, avoiding buffer row
                adrt_array_access(out, output_shape,
                                  plane, a, output_shape[1]-1-d)
                    = adrt_array_access(prev, prev_shape,
                                        plane, d+1, a, 0);
            }
        }
    }

    // PYTHON API ALLOWED BELOW THIS POINT
    Py_END_ALLOW_THREADS;

    PyMem_Free(aux);
    return true;
}
#endif // ADRTC_CDEFS_IADRT_H
