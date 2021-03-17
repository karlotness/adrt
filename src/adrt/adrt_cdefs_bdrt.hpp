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
#ifndef ADRTC_CDEFS_BDRT_H
#define ADRTC_CDEFS_BDRT_H

#include "adrt_cdefs_py.hpp"
#include "adrt_cdefs_common.hpp"
#include <array>
#include <utility>
#include <type_traits>

template <typename adrt_scalar, typename adrt_shape>
static bool bdrt_impl(const adrt_scalar *const data, const unsigned char ndims,
const adrt_shape *const shape, const int base_iter_start, const int
base_iter_end, adrt_scalar *const out,
                      const adrt_shape *const base_output_shape) {
    // The current implementation multiplies values by float constants and will not work correctly with integers
    static_assert(std::is_floating_point<adrt_scalar>::value, "Backprojection requires floating point");

    // Shape (plane, quadrant, row, col)
    const std::array<adrt_shape, 4> corrected_shape =
        {(ndims > 3 ? shape[0] : 1),
         (ndims > 3 ? shape[1] : shape[0]),
         (ndims > 3 ? shape[2] : shape[1]),
         (ndims > 3 ? shape[3] : shape[2])};

    // Output shape (plane, row, col)
    const std::array<adrt_shape, 4> output_shape =
        {(ndims > 3 ? base_output_shape[0] : 1),
         (ndims > 3 ? base_output_shape[1] : base_output_shape[0]),
         (ndims > 3 ? base_output_shape[2] : base_output_shape[1]),
         (ndims > 3 ? base_output_shape[3] : base_output_shape[2])};

    // Require that the matrix be square (power of two checked elsewhere)
    if(corrected_shape[2] != 2*corrected_shape[3]-1) {
        PyErr_SetString(PyExc_ValueError, "Provided array must be of dimension 2*N-1 x N");
        return false;
    }

    // Require that second dimension corresp to quadrants
    if(corrected_shape[1] != 4) {
        PyErr_SetString(PyExc_ValueError, "Data for four quadrants must be provided");
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
    const size_t img_size = corrected_shape[0] * corrected_shape[1]
                          * corrected_shape[2] * corrected_shape[3];
    const size_t buf_size = img_size; // One buffer per quadrant of size planes * N * N
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

    // Each quadrant has a different shape (the padding goes in a different place)
    std::array<adrt_shape, 4> prev_shape =
                              {corrected_shape[0], corrected_shape[1],
                               corrected_shape[2], corrected_shape[3]};

    // First, memcpy in the base image into each buffer
    for(adrt_shape quadrant = 0; quadrant < corrected_shape[1]; ++quadrant) {
        for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
            for(adrt_shape row = 0; row < corrected_shape[2]; ++row) {
                for(adrt_shape col = 0; col < corrected_shape[3]; ++col) {
                    adrt_array_access(prev, prev_shape, plane, quadrant,
                                      row, col)
                    = adrt_array_access(data, corrected_shape,
                                           plane, quadrant, row, col);
                }
            }
        }
    }

    std::array<adrt_shape, 5> prev_stride = {
                    corrected_shape[1]*corrected_shape[2]*corrected_shape[3],
                    corrected_shape[2]*corrected_shape[3],
                    corrected_shape[3],corrected_shape[3],1};
    std::array<adrt_shape, 5> curr_stride = {
                    // fixed strides (does not change in loop)
                    prev_stride[0], // plane stride
                    prev_stride[1], // quadrant stride
                    prev_stride[2], // row stride
                    // variable strides
                    prev_stride[3], // section stride
                    prev_stride[4]};// no. of sections

    adrt_shape nsections = 1;

    for(adrt_shape i = 1; i <= iter_start; ++i) {
        // Compute the curr_stride for the current buffer (based on prev shape)
        curr_stride[3] = adrt_floor_div2(prev_stride[3]); // stride for section
        nsections *= 2;

        // Swap the "curr" and "prev" buffers and shapes
        prev_stride = curr_stride;
    }

    // Outer loop over iterations (this loop must be serial)
    for(adrt_shape i = iter_start + 1; i <= iter_end; ++i) {
        // Compute the curr_stride for the current buffer (based on prev shape)
        curr_stride[3] = adrt_floor_div2(prev_stride[3]); // stride for section

        // Inner loops (these loops can be parallel)
        #pragma omp parallel for collapse(5) shared(curr, prev, curr_stride, prev_stride, nsections)
        for(adrt_shape plane = 0; plane < corrected_shape[0]; ++plane) {
            for(adrt_shape quadrant = 0; quadrant < corrected_shape[1]; ++quadrant) {
                for(adrt_shape j = 0; j < nsections; ++j) {
                    for(adrt_shape a = 0; a < curr_stride[3]; ++a) {
                        for(adrt_shape x = 0; x < corrected_shape[2]; ++x) {

                            // right image
                            // check the index access for x
                            adrt_scalar raval = 0;
                            const adrt_shape lxa = x + a;
                            if(lxa >= 0 && lxa < corrected_shape[2]) {
                                raval = adrt_array_stride_access(
                                            prev, prev_stride,
                                            plane, quadrant, lxa, j, 2*a);
                            }
                            // check the index access for x
                            adrt_scalar rbval = 0;
                            const adrt_shape lxb = x + a + 1;
                            if(lxb >= 0 && lxb < corrected_shape[2]) {
                                rbval = adrt_array_stride_access(
                                            prev,prev_stride,
                                            plane, quadrant, lxb, j, 2*a + 1);
                            }

                            adrt_array_stride_access(curr, curr_stride,
                                          plane, quadrant, x, 2*j + 1, a)
                                = raval + rbval;

                            // left image
                            adrt_scalar lbval = adrt_array_stride_access(
                                        prev, prev_stride,
                                        plane, quadrant, x, j, 2*a);

                            adrt_scalar laval = adrt_array_stride_access(
                                        prev, prev_stride,
                                        plane, quadrant, x, j, 2*a + 1);

                            adrt_array_stride_access(curr, curr_stride,
                                        plane, quadrant, x, 2*j, a)
                                = laval + lbval;

                        }
                    }
                }
            }
        }
        nsections *= 2;

        // Swap the "curr" and "prev" buffers and shapes
        std::swap(curr,prev);
        prev_stride = curr_stride;
    }

    // Copy results to destination buffer
    for(adrt_shape plane = 0; plane < output_shape[0]; ++plane) {
        for(adrt_shape d = 0; d < output_shape[2]; ++d) {
            for(adrt_shape a = 0; a < output_shape[3]; ++a) {
                for(adrt_shape quadrant = 0; quadrant < output_shape[1]; ++quadrant) {

                    adrt_array_access(out, prev_shape, plane, quadrant, d, a)
                    = adrt_array_access(prev, prev_shape,plane,quadrant, d, a);
                }
            }
        }
    }


    // PYTHON API ALLOWED BELOW THIS POINT
    Py_END_ALLOW_THREADS;

    PyMem_Free(aux);
    return true;
}

#endif // ADRTC_CDEFS_BDRT_H
