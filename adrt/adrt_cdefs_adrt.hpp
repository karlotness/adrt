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
#ifndef ADRTC_CDEFS_ADRT_H
#define ADRTC_CDEFS_ADRT_H

#include "adrt_cdefs_common.hpp"
#include <array>

template <typename adrt_scalar, typename adrt_shape>
static bool adrt_impl(const adrt_scalar *const data, const unsigned char ndims, const adrt_shape *const shape, adrt_scalar *const out,
                      const adrt_shape *const base_output_shape) {
    // Shape (plane, row, col)
    const std::array<adrt_shape, 3> corrected_shape =
        {(ndims > 2 ? shape[0] : 1),
         (ndims > 2 ? shape[1] : shape[0]),
         (ndims > 2 ? shape[2] : shape[1])};

    const std::array<adrt_shape, 4> output_shape =
        {(ndims > 2 ? base_output_shape[0] : 1),
         (ndims > 2 ? base_output_shape[1] : base_output_shape[0]),
         (ndims > 2 ? base_output_shape[2] : base_output_shape[1]),
         (ndims > 2 ? base_output_shape[3] : base_output_shape[2])};

    // Require that the matrix be square (power of two checked elsewhere)
    if(corrected_shape[1] != corrected_shape[2]) {
        PyErr_SetString(PyExc_ValueError, "Provided array must be square");
        return false;
    }

    // Check that shape is sensible
    for(int i = 0; i < 3; ++i) {
        if(corrected_shape[i] <= 0) {
            PyErr_SetString(PyExc_ValueError, "Provided array must have no axes of zero size");
            return false;
        }
    }

    // Allocate auxiliary memory
    const size_t img_size = corrected_shape[0] * corrected_shape[1] * (2 * corrected_shape[2] - 1);
    const size_t buf_size = 4 * img_size; // One buffer per quadrant of size planes * N * N
    // Allocate two of these buffers
    adrt_scalar *const aux = PyMem_New(adrt_scalar, 2 * buf_size);
    if(!aux) {
        PyErr_NoMemory();
        return false;
    }

    // NO PYTHON API BELOW THIS POINT
    Py_BEGIN_ALLOW_THREADS;

    const adrt_shape num_iters = adrt_num_iters(corrected_shape[1]);

    adrt_scalar *curr = aux;
    adrt_scalar *prev = aux + buf_size;
    // Each quadrant has a different shape (the padding goes in a different place)
    std::array<adrt_shape, 5> curr_shape = {0};
    std::array<adrt_shape, 5> prev_shape = {4, corrected_shape[0], corrected_shape[1], 2 * corrected_shape[2] - 1, 1};

    // First, memcpy in the base image into each buffer
    for(adrt_shape quadrant = 0; quadrant < 4; ++quadrant) {
        const adrt_shape zero = 0;
        if(quadrant == 0) {
            // Direct copy row by row
            for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
                for(adrt_shape row = 0; row < corrected_shape[1]; ++row) {
                    for(adrt_shape col = 0; col < corrected_shape[2]; ++col) {
                        adrt_array_access(prev, prev_shape, quadrant, plane, row, corrected_shape[2] - col - 1, zero) = \
                            adrt_array_access(data, corrected_shape, plane, row, col);
                    }
                }
            }
        }
        else if (quadrant == 1) {
            // Transpose the squares
            for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
                for(adrt_shape row = 0; row < corrected_shape[1]; ++row) {
                    for(adrt_shape col = 0; col < corrected_shape[2]; ++col) {
                        adrt_array_access(prev, prev_shape, quadrant, plane, corrected_shape[1] - row - 1, corrected_shape[2] - col - 1, zero) = \
                            adrt_array_access(data, corrected_shape, plane, col, corrected_shape[1] - row - 1);
                    }
                }
            }
        }
        else if (quadrant == 2) {
            // Transpose the squares and flip along x
            for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
                for(adrt_shape row = 0; row < corrected_shape[1]; ++row) {
                    for(adrt_shape col = 0; col < corrected_shape[2]; ++col) {
                        adrt_array_access(prev, prev_shape, quadrant, plane, corrected_shape[1] - row - 1, corrected_shape[2] - col - 1, zero) = \
                            adrt_array_access(data, corrected_shape, plane, corrected_shape[2] - col - 1, corrected_shape[1] - row - 1);
                    }
                }
            }
        }
        else {
            // Flip along y
            for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
                for(adrt_shape row = 0; row < corrected_shape[1]; ++row) {
                    for(adrt_shape col = 0; col < corrected_shape[2]; ++col) {
                        adrt_array_access(prev, prev_shape, quadrant, plane, row, corrected_shape[2] - col - 1, zero) = \
                            adrt_array_access(data, corrected_shape, plane, corrected_shape[1] - row - 1, col);
                    }
                }
            }
        }
        // Fill the rest with zeros
        for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
            for(adrt_shape row = 0; row < corrected_shape[1]; ++row) {
                for(adrt_shape col = corrected_shape[2]; col < 2 * corrected_shape[2] - 1; ++col) {
                    adrt_array_access(prev, prev_shape, quadrant, plane, row, col, zero) = 0;
                }
            }
        }
    }

    // Outer loop over iterations (this loop must be serial)
    for(adrt_shape i = 1; i <= num_iters; ++i) {
        // Compute the curr_shape for the current buffer (based on prev shape)
        curr_shape[0] = 4; // We always have four quadrants
        curr_shape[1] = corrected_shape[0]; // We always have the same number of planes
        curr_shape[2] = adrt_ceil_div2(prev_shape[2]); // We halve the number of rows
        curr_shape[3] = 2 * corrected_shape[2] - 1; // Keep the same number of columns
        curr_shape[4] = prev_shape[4] * 2; // The number of angles doubles

        // Inner loops (these loops can be parallel)
        #pragma omp parallel for collapse(5) default(none) shared(curr, prev, curr_shape, prev_shape, i)
        for(adrt_shape quadrant = 0; quadrant < curr_shape[0]; ++quadrant) {
            for(adrt_shape plane = 0; plane < curr_shape[1]; ++plane) {
                for(adrt_shape j = 0; j < curr_shape[2]; ++j) {
                    for(adrt_shape x = 0; x < curr_shape[3]; ++x) {
                        for(adrt_shape a = 0; a < curr_shape[4]; ++a) {
                            // TODO: Adjust loop bounds to avoid operations on all zeros. This will make x depend on the angle.
                            // Will likely have to fuse the iterations by hand
                            adrt_scalar aval = adrt_array_access(prev, prev_shape, quadrant, plane, 2 * j, x, adrt_floor_div2(a));
                            // Need to check the index access for x
                            const adrt_shape xb_idx = x - adrt_ceil_div2(a);
                            adrt_scalar bval = 0;
                            if(xb_idx >= 0 && xb_idx < prev_shape[3]) {
                                bval = adrt_array_access(prev, prev_shape, quadrant, plane, (2 * j) + 1, xb_idx, adrt_floor_div2(a));
                            }
                            adrt_array_access(curr, curr_shape, quadrant, plane, j, x, a) = aval + bval;
                        }
                    }
                }
            }
        }

        // Swap the "curr" and "prev" buffers and shapes
        adrt_scalar *const tmp = curr;
        curr = prev;
        prev = tmp;
        prev_shape = curr_shape;
    }

    // Copy results to destination buffer
    for(adrt_shape quadrant = 0; quadrant < 4; ++quadrant) {
        const adrt_shape zero = 0;
        for(adrt_shape plane = 0; plane < output_shape[0]; ++plane) {
            for(adrt_shape d = 0; d < prev_shape[3]; ++d) {
                for(adrt_shape a = 0; a < prev_shape[4]; ++a) {
                    adrt_shape acc_d = 0;
                    adrt_shape acc_a = 0;
                    if(quadrant == 0) {
                        acc_d = d;
                        acc_a = a;
                    }
                    else if(quadrant == 1) {
                        acc_d = prev_shape[3] - d - 1;
                        acc_a = prev_shape[4] - a - 1;
                    }
                    else if(quadrant == 2) {
                        acc_d = d;
                        acc_a = a;
                    }
                    else {
                        acc_d = prev_shape[3] - d - 1;
                        acc_a = prev_shape[4] - a - 1;
                    }
                    const adrt_scalar val = adrt_array_access(prev, prev_shape, quadrant, plane, zero, acc_d, acc_a);
                    adrt_array_access(out, output_shape, plane, quadrant, d, a) = val;
                }
            }
        }
    }

    // PYTHON API ALLOWED BELOW THIS POINT
    Py_END_ALLOW_THREADS;

    PyMem_Free(aux);
    return true;
}

#endif // ADRTC_CDEFS_ADRT_H
