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

#include "adrtc_cdefs_common.hpp"
#include <cstring>

template <typename adrt_scalar, typename adrt_shape>
static bool _adrt(const adrt_scalar *const data, const unsigned char ndims, const adrt_shape *const shape, adrt_scalar *const out) {
    // Shape (plane, row, col)
    const adrt_shape corrected_shape[3] =
        {(ndims > 2 ? shape[0] : 1),
         (ndims > 2 ? shape[1] : shape[0]),
         (ndims > 2 ? shape[2] : shape[1])};

    const adrt_shape output_shape[3] =
        {corrected_shape[0],
         corrected_shape[1],
         4 * (corrected_shape[2] - 1)};

    // Require that the matrix be square (power of two checked elsewhere)
    if(corrected_shape[1] != corrected_shape[2]) {
        return false;
    }

    // Check that shape is sensible
    for(int i = 0; i < 3; ++i) {
        if(corrected_shape[i] <= 0) {
            return false;
        }
    }

    // Allocate auxiliary memory
    const size_t img_size = corrected_shape[0] * corrected_shape[1] * corrected_shape[2];
    const size_t buf_size = 4 * 2 * img_size; // One buffer per quadrant of size planes * N * 2N
    // Allocate two of these buffers
    adrt_scalar *const aux = PyMem_New(adrt_scalar, 2 * buf_size);
    if(!aux) {
        PyErr_NoMemory();
        return false;
    }

    // NO PYTHON API BELOW THIS POINT
    Py_BEGIN_ALLOW_THREADS;

    const adrt_shape num_iters = adrt_num_iters(corrected_shape);

    adrt_scalar *curr = aux;
    adrt_scalar *prev = aux + buf_size;
    // Each quadrant has a different shape (the padding goes in a different place)
    adrt_shape curr_shape[5] = {0};
    adrt_shape prev_shape[5] = {4, corrected_shape[0], corrected_shape[1], 2 * corrected_shape[2], 1};

    // First, memcpy in the base image into each buffer
    for(adrt_shape quadrant = 0; quadrant < 4; ++quadrant) {
        const adrt_shape zero = 0;
        if(quadrant == 0) {
            // Direct copy row by row
            for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
                for(adrt_shape row = 0; row < corrected_shape[1]; ++row) {
                    for(adrt_shape col = 0; col < corrected_shape[2]; ++col) {
                        adrt_array_5d_mod_access(prev, prev_shape, quadrant, plane, row, col, zero) = \
                            adrt_array_3d_access(data, corrected_shape, plane, row, col);
                    }
                }
            }
        }
        else if (quadrant == 1) {
            // Transpose the squares
            for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
                for(adrt_shape row = 0; row < corrected_shape[1]; ++row) {
                    for(adrt_shape col = 0; col < corrected_shape[2]; ++col) {
                        adrt_array_5d_mod_access(prev, prev_shape, quadrant, plane, row, col, zero) = \
                            adrt_array_3d_access(data, corrected_shape, plane, col, row);
                    }
                }
            }
        }
        else if (quadrant == 2) {
            // Transpose the squares and flip along x
            for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
                for(adrt_shape row = 0; row < corrected_shape[1]; ++row) {
                    for(adrt_shape col = 0; col < corrected_shape[2]; ++col) {
                        adrt_array_5d_mod_access(prev, prev_shape, quadrant, plane, row, col, zero) = \
                            adrt_array_3d_access(data, corrected_shape, plane, col, corrected_shape[1] - row - 1);
                    }
                }
            }
        }
        else {
            // Flip along y
            for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
                for(adrt_shape row = 0; row < corrected_shape[1]; ++row) {
                    for(adrt_shape col = 0; col < corrected_shape[2]; ++col) {
                        adrt_array_5d_mod_access(prev, prev_shape, quadrant, plane, row, col, zero) = \
                            adrt_array_3d_access(data, corrected_shape, plane, corrected_shape[1] - row - 1, col);
                    }
                }
            }
        }
        // End if.
        // In all cases pad with zeros on the "right"
        for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
            for(adrt_shape row = 0; row < corrected_shape[1]; ++row) {
                for(adrt_shape col = corrected_shape[2]; col < 2 * corrected_shape[2]; ++col) {
                    adrt_array_5d_mod_access(prev, prev_shape, quadrant, plane, row, col, zero) = 0;
                }
            }
        }
    }

    // Outer loop over iterations (this loop must be serial)
    for(adrt_shape i = 1; i <= num_iters; ++i) {
        // Compute the curr_shape for the current buffer (based on prev shape)
        curr_shape[0] = 4; // We always have four quadrants
        curr_shape[1] = corrected_shape[0]; // We always have the same number of planes
        curr_shape[2] = corrected_shape[1]; // In these quadrants the X dimension doesn't change
        curr_shape[3] = adrt_ceil_div2(prev_shape[3]); // But the Y dimension is halved
        curr_shape[4] = prev_shape[4] * 2; // The number of angles doubles

        const adrt_shape two_to_i = 1<<i;
        const adrt_shape j_lim = corrected_shape[2] / two_to_i;

        // Inner loops (these loops can be parallel)
#pragma omp parallel for collapse(5) default(none) shared(curr, prev, curr_shape, prev_shape, i, j_lim, corrected_shape)
        for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
            for(adrt_shape quadrant = 0; quadrant < 4; ++quadrant) {
                for(adrt_shape a = 0; a <= (1<<i) - 1; ++a) {
                    for(adrt_shape j = 0; j < j_lim; ++j) {
                        for(adrt_shape x = 0; x <= 2 * corrected_shape[1] - 1; ++x) {
                            const adrt_scalar aval = adrt_array_5d_mod_access(prev, prev_shape, quadrant, plane, x, j, adrt_floor_div2(a));
                            const adrt_scalar bval = adrt_array_5d_mod_access(prev, prev_shape, quadrant, plane, x - adrt_ceil_div2(a), j + 1, adrt_floor_div2(a));
                            adrt_array_5d_mod_access(curr, curr_shape, quadrant, plane, x, j, a) = aval + bval;
                        }
                    }
                }
            }
        }

        // Swap the "curr" and "prev" buffers and shapes
        adrt_scalar *const tmp = curr;
        curr = prev;
        prev = tmp;
        std::memcpy(prev_shape, curr_shape, sizeof(adrt_shape) * 5);
    }

    // Copy results to destination buffer
    // TODO: Fix overlaps. There should be (4N - 3) unique angles
    for(adrt_shape quadrant = 0; quadrant < 4; ++quadrant) {
        const adrt_shape zero = 0;
        for(adrt_shape plane = 0; plane < output_shape[0]; ++plane) {
            for(adrt_shape d = 0; d < output_shape[1]; ++d) {
                for(adrt_shape a = 0; a < output_shape[1]; ++a) {
                    const adrt_scalar val = adrt_array_5d_mod_access(prev, prev_shape, quadrant, plane, d, zero, a);
                    adrt_array_3d_access(out, output_shape, plane, d, (output_shape[1] * quadrant) + a) = val;
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
