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
#ifndef ADRT_CDEFS_INTERP_ADRTCART_H
#define ADRT_CDEFS_INTERP_ADRTCART_H

#include "adrt_cdefs_py.hpp"
#include "adrt_cdefs_common.hpp"
#include <cmath>
#include <array>
#include <utility>
#include <type_traits>

template <typename adrt_scalar, typename adrt_shape>
static bool interp_adrtcart_impl(const adrt_scalar *const data, const unsigned char ndims, const adrt_shape *const shape, adrt_scalar *const out, const adrt_shape *const base_output_shape) {
    // The current implementation uses floating point constants and will not work correctly with integers
    static_assert(std::is_floating_point<adrt_scalar>::value, "Cartesian interpolation requires floating point");

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
    const size_t col_size = corrected_shape[2];  // 2N-1
    const size_t buf_size = col_size;            // One buffer per a column

    // Allocate two of these buffers
    adrt_scalar *const buf_col = PyMem_New(adrt_scalar, buf_size);
    if(!buf_col) {
        PyErr_NoMemory();
        return false;
    }

    // NO PYTHON API BELOW THIS POINT
    Py_BEGIN_ALLOW_THREADS;

    const adrt_shape N = corrected_shape[3];
    const adrt_scalar Nf = static_cast<adrt_scalar>(N);

    // uniform angle grid
    const adrt_scalar dtheta =
        static_cast<adrt_scalar>(adrt::_const::pi_4) / (Nf - 1);

    // uniform offset grid
    const adrt_scalar dt = static_cast<adrt_scalar>(adrt::_const::sqrt2) / Nf;
    const adrt_scalar t_lb = (-1 * static_cast<adrt_scalar>(adrt::_const::sqrt2) + dt) / 2;

    for(adrt_shape plane = 0; plane < output_shape[0]; plane++){
    for(adrt_shape quadrant = 0; quadrant < output_shape[1]; quadrant++){

    for(adrt_shape i = 0; i < N; i++){
        adrt_scalar theta = i * dtheta;

        adrt_scalar s = std::ceil((Nf - 1) * std::tan(theta));
        adrt_scalar theta_lower = std::atan((s - 1) / (Nf - 1));
        adrt_scalar theta_upper = std::atan((s      ) / (Nf - 1));

        adrt_scalar wgt = (theta - theta_lower) / (theta_upper - theta_lower);

        wgt = (wgt > 1 ? 1 : wgt);
        wgt = (wgt < 0 ? 0 : wgt);

        // compute, store interpolated column in buffer
        for(adrt_shape j = 0; j < 2 * N - 1; j++){

            const adrt_shape d = j;
            const adrt_shape a_left  = s - 1;
            const adrt_shape a_right = s;

            const adrt_scalar val_left = adrt_array_access(data,
                corrected_shape, plane, quadrant, d, a_left);

            const adrt_scalar val_right = adrt_array_access(data,
                corrected_shape, plane, quadrant, d, a_right);

            buf_col[j] = (1 - wgt) * val_left + wgt * val_right;
        }

        adrt_scalar h_star = (1 - std::tan(theta)) / 2;
        adrt_scalar cos_factor = std::cos(theta);

        adrt_scalar h = 1 - Nf;

        // do column (buffer) interpolation
        for (adrt_shape j = 0; j < N; j++){

            adrt_scalar t = t_lb + j * dt;

            h = std::round(Nf * (t / cos_factor + h_star));
            adrt_scalar t_lower = cos_factor * ((h - adrt_scalar{0.5}) / Nf - h_star);
            adrt_scalar t_upper = cos_factor * ((h + adrt_scalar{0.5}) / Nf - h_star);

            wgt = (t - t_lower) / (t_upper - t_lower);

            const adrt_shape d = j;
            const adrt_shape a = i;
            int hi = Nf - h;

            if (hi > 2*N - 1 || hi < 1)
            adrt_array_access(out, output_shape, plane, quadrant, d, a) = 0.0;
            else {
            // put inteprolated value to destination buffer
            adrt_array_access(out, output_shape, plane, quadrant, d, a)
            = ((1 - wgt) * buf_col[hi - 1] + wgt * buf_col[hi]) / cos_factor;
            }
        }
    }
    }
    }

    // PYTHON API ALLOWED BELOW THIS POINT
    Py_END_ALLOW_THREADS;

    PyMem_Free(buf_col);
    return true;
}

#endif // ADRT_CDEFS_INTERP_ADRTCART_H
