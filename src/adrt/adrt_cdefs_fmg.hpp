/*
 * Copyright (c) 2022 Karl Otness, Donsub Rim
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
#ifndef ADRT_CDEFS_FMG_H
#define ADRT_CDEFS_FMG_H

#include <array>
#include <cassert>
#include <type_traits>
#include "adrt_cdefs_common.hpp"

namespace adrt {

    // Defined in: adrt_cdefs_common.cpp
    bool fmg_restriction_is_valid_shape(const std::array<size_t, 4> &shape);
    bool fmg_prolongation_is_valid_shape(const std::array<size_t, 3> &shape);
    bool fmg_highpass_is_valid_shape(const std::array<size_t, 3> &shape);
    std::array<size_t, 4> fmg_restriction_result_shape(const std::array<size_t, 4> &shape);
    std::array<size_t, 3> fmg_prolongation_result_shape(const std::array<size_t, 3> &shape);
    std::array<size_t, 3> fmg_highpass_result_shape(const std::array<size_t, 3> &shape);

    template <typename adrt_scalar>
    void fmg_restriction(const adrt_scalar *const ADRT_RESTRICT data, const std::array<size_t, 4> &shape, adrt_scalar *const ADRT_RESTRICT out) {
        static_assert(std::is_floating_point<adrt_scalar>::value, "FMG restriction requires floating point");
        assert(data);
        assert(out);
        assert(adrt::fmg_restriction_is_valid_shape(shape));

        const std::array<size_t, 4> output_shape = adrt::fmg_restriction_result_shape(shape);

        ADRT_OPENMP("omp parallel for collapse(4) default(none) shared(data, shape, out, output_shape)")
        for(size_t batch = 0; batch < std::get<0>(output_shape); ++batch) {
            for(size_t quadrant = 0; quadrant < 4u; ++quadrant) {
                for(size_t row = 0; row < std::get<2>(output_shape); ++row) {
                    for(size_t col = 0; col < std::get<3>(output_shape); ++col) {
                        const adrt_scalar val_a = adrt::_common::array_access(data, shape, batch, quadrant, 2_uz * row, 2_uz * col);
                        const adrt_scalar val_b = adrt::_common::array_access(data, shape, batch, quadrant, 2_uz * row + 1_uz, 2_uz * col);
                        adrt::_common::array_access(out, output_shape, batch, quadrant, row, col) = (val_a + val_b) / static_cast<adrt_scalar>(4);
                    }
                }
            }
        }
    }

    template <typename adrt_scalar>
    void fmg_prolongation(const adrt_scalar *const ADRT_RESTRICT data, const std::array<size_t, 3> &shape, adrt_scalar *const ADRT_RESTRICT out) {
        assert(data);
        assert(out);
        assert(adrt::fmg_prolongation_is_valid_shape(shape));

        const std::array<size_t, 3> output_shape = adrt::fmg_prolongation_result_shape(shape);

        ADRT_OPENMP("omp parallel for collapse(3) default(none) shared(data, shape, out, output_shape)")
        for(size_t batch = 0; batch < std::get<0>(shape); ++batch) {
            for(size_t row = 0; row < std::get<1>(shape); ++row) {
                for(size_t col = 0; col < std::get<2>(shape); ++col) {
                    const adrt_scalar val = adrt::_common::array_access(data, shape, batch, row, col);
                    adrt::_common::array_access(out, output_shape, batch, 2_uz * row, 2_uz * col) = val;
                    adrt::_common::array_access(out, output_shape, batch, 2_uz * row, 2_uz * col + 1_uz) = val;
                    adrt::_common::array_access(out, output_shape, batch, 2_uz * row + 1_uz, 2_uz * col) = val;
                    adrt::_common::array_access(out, output_shape, batch, 2_uz * row + 1_uz, 2_uz * col + 1_uz) = val;
                }
            }
        }
    }

    template <typename adrt_scalar>
    void fmg_highpass(const adrt_scalar *const ADRT_RESTRICT data, const std::array<size_t, 3> &shape, adrt_scalar *const ADRT_RESTRICT out) {
        static_assert(std::is_floating_point<adrt_scalar>::value, "FMG high-pass filter requires floating point");
        assert(data);
        assert(out);
        assert(adrt::fmg_highpass_is_valid_shape(shape));
        assert(shape == adrt::fmg_highpass_result_shape(shape));

        // Convolution constants for kernel [[a, b, a], [[b, c, b]], [a, b, a]]
        const adrt_scalar conv_a = static_cast<adrt_scalar>(-1) / static_cast<adrt_scalar>(16);
        const adrt_scalar conv_b = static_cast<adrt_scalar>(-1) / static_cast<adrt_scalar>(8);
        const adrt_scalar conv_c = static_cast<adrt_scalar>(3) / static_cast<adrt_scalar>(4);

        ADRT_OPENMP("omp parallel for collapse(3) default(none) shared(data, shape, out, conv_a, conv_b, conv_c)")
        for(size_t batch = 0; batch < std::get<0>(shape); ++batch) {
            for(size_t row = 0; row < std::get<1>(shape); ++row) {
                for(size_t col = 0; col < std::get<2>(shape); ++col) {
                    const size_t prev_row = (row > 0u ? row - 1_uz : 1_uz);
                    const size_t next_row = (row < std::get<1>(shape) - 1u ? row + 1_uz : std::get<1>(shape) - 2_uz);
                    const size_t prev_col = (col > 0u ? col - 1_uz : 1_uz);
                    const size_t next_col = (col < std::get<2>(shape) - 1u ? col + 1_uz : std::get<2>(shape) - 2_uz);
                    adrt_scalar val = 0;
                    // Conv row 1
                    val += conv_a * adrt::_common::array_access(data, shape, batch, prev_row, prev_col);
                    val += conv_b * adrt::_common::array_access(data, shape, batch, prev_row, col);
                    val += conv_a * adrt::_common::array_access(data, shape, batch, prev_row, next_col);
                    // Conv row 2
                    val += conv_b * adrt::_common::array_access(data, shape, batch, row, prev_col);
                    val += conv_c * adrt::_common::array_access(data, shape, batch, row, col);
                    val += conv_b * adrt::_common::array_access(data, shape, batch, row, next_col);
                    // Conv row 3
                    val += conv_a * adrt::_common::array_access(data, shape, batch, next_row, prev_col);
                    val += conv_b * adrt::_common::array_access(data, shape, batch, next_row, col);
                    val += conv_a * adrt::_common::array_access(data, shape, batch, next_row, next_col);
                    // Store result
                    adrt::_common::array_access(out, shape, batch, row, col) = val;
                }
            }
        }
    }

}

#endif // ADRT_CDEFS_FMG_H
