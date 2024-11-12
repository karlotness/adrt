/*
 * Copyright 2023 Karl Otness, Donsub Rim
 *
 * SPDX-License-Identifier: BSD-3-Clause
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

#ifndef ADRT_CDEFS_FMG_H
#define ADRT_CDEFS_FMG_H

#include <array>
#include <span>
#include <cassert>
#include <concepts>
#include <algorithm>
#include "adrt_cdefs_common.hpp"

namespace adrt {

    // Defined in: adrt_cdefs_common.cpp
    bool fmg_restriction_is_valid_shape(std::span<const size_t, 4> shape);
    bool fmg_prolongation_is_valid_shape(std::span<const size_t, 3> shape);
    bool fmg_highpass_is_valid_shape(std::span<const size_t, 3> shape);
    std::array<size_t, 4> fmg_restriction_result_shape(std::span<const size_t, 4> shape);
    std::array<size_t, 3> fmg_prolongation_result_shape(std::span<const size_t, 3> shape);
    std::array<size_t, 3> fmg_highpass_result_shape(std::span<const size_t, 3> shape);

    template <std::floating_point adrt_scalar>
    void fmg_restriction(const adrt_scalar *const ADRT_RESTRICT data, std::span<const size_t, 4> shape, adrt_scalar *const ADRT_RESTRICT out) {
        assert(data);
        assert(out);
        assert(adrt::fmg_restriction_is_valid_shape(shape));

        const std::array<size_t, 4> output_shape = adrt::fmg_restriction_result_shape(shape);

        ADRT_OPENMP("omp parallel for collapse(4) default(none) shared(data, shape, out, output_shape)")
        for(size_t batch = 0; batch < adrt::_common::get<0>(output_shape); ++batch) {
            for(size_t quadrant = 0; quadrant < 4u; ++quadrant) {
                for(size_t row = 0; row < adrt::_common::get<2>(output_shape); ++row) {
                    for(size_t col = 0; col < adrt::_common::get<3>(output_shape); ++col) {
                        const adrt_scalar val_a = adrt::_common::array_access(data, shape, batch, quadrant, 2_uz * row, 2_uz * col);
                        const adrt_scalar val_b = adrt::_common::array_access(data, shape, batch, quadrant, 2_uz * row + 1_uz, 2_uz * col);
                        adrt::_common::array_access(out, output_shape, batch, quadrant, row, col) = (val_a + val_b) / static_cast<adrt_scalar>(4);
                    }
                }
            }
        }
    }

    template <typename adrt_scalar>
    void fmg_prolongation(const adrt_scalar *const ADRT_RESTRICT data, std::span<const size_t, 3> shape, adrt_scalar *const ADRT_RESTRICT out) {
        assert(data);
        assert(out);
        assert(adrt::fmg_prolongation_is_valid_shape(shape));

        const std::array<size_t, 3> output_shape = adrt::fmg_prolongation_result_shape(shape);

        ADRT_OPENMP("omp parallel for collapse(3) default(none) shared(data, shape, out, output_shape)")
        for(size_t batch = 0; batch < adrt::_common::get<0>(shape); ++batch) {
            for(size_t row = 0; row < adrt::_common::get<1>(shape); ++row) {
                for(size_t col = 0; col < adrt::_common::get<2>(shape); ++col) {
                    const adrt_scalar val = adrt::_common::array_access(data, shape, batch, row, col);
                    adrt::_common::array_access(out, output_shape, batch, 2_uz * row, 2_uz * col) = val;
                    adrt::_common::array_access(out, output_shape, batch, 2_uz * row, 2_uz * col + 1_uz) = val;
                    adrt::_common::array_access(out, output_shape, batch, 2_uz * row + 1_uz, 2_uz * col) = val;
                    adrt::_common::array_access(out, output_shape, batch, 2_uz * row + 1_uz, 2_uz * col + 1_uz) = val;
                }
            }
        }
    }

    template <std::floating_point adrt_scalar>
    void fmg_highpass(const adrt_scalar *const ADRT_RESTRICT data, std::span<const size_t, 3> shape, adrt_scalar *const ADRT_RESTRICT out) {
        assert(data);
        assert(out);
        assert(adrt::fmg_highpass_is_valid_shape(shape));
        assert(std::ranges::equal(shape, adrt::fmg_highpass_result_shape(shape)));

        // Convolution constants for kernel [[a, b, a], [b, c, b], [a, b, a]]
        const adrt_scalar conv_a = static_cast<adrt_scalar>(-0.0625L); // -1/16
        const adrt_scalar conv_b = static_cast<adrt_scalar>(-0.125L); // -1/8
        const adrt_scalar conv_c = static_cast<adrt_scalar>(0.75L); // 3/4

        ADRT_OPENMP("omp parallel for collapse(2) default(none) shared(data, shape, out, conv_a, conv_b, conv_c)")
        for(size_t batch = 0; batch < adrt::_common::get<0>(shape); ++batch) {
            for(size_t row = 0; row < adrt::_common::get<1>(shape); ++row) {
                const size_t prev_row = (row == 0u ? 1_uz : row - 1_uz);
                const size_t next_row = (row == adrt::_common::get<1>(shape) - 1_uz ? row - 1_uz : row + 1_uz);

                // First col
                {
                    const size_t col = 0;
                    const size_t prev_col = 1;
                    // Conv row 1
                    const adrt_scalar v11 = conv_a * adrt::_common::array_access(data, shape, batch, prev_row, prev_col);
                    const adrt_scalar v12 = conv_b * adrt::_common::array_access(data, shape, batch, prev_row, col);
                    // Conv row 2
                    const adrt_scalar v21 = conv_b * adrt::_common::array_access(data, shape, batch, row, prev_col);
                    const adrt_scalar v22 = conv_c * adrt::_common::array_access(data, shape, batch, row, col);
                    // Conv row 3
                    const adrt_scalar v31 = conv_a * adrt::_common::array_access(data, shape, batch, next_row, prev_col);
                    const adrt_scalar v32 = conv_b * adrt::_common::array_access(data, shape, batch, next_row, col);
                    // Store result
                    adrt::_common::array_access(out, shape, batch, row, col) = (v11 + v21 + v31) + (v12 + v22 + v32) + (v11 + v21 + v31);
                }

                // Middle columns
                ADRT_OPENMP("omp simd")
                for(size_t col = 1; col < adrt::_common::get<2>(shape) - 1_uz; ++col) {
                    const size_t prev_col = col - 1_uz;
                    const size_t next_col = col + 1_uz;
                    // Conv row 1
                    const adrt_scalar v11 = conv_a * adrt::_common::array_access(data, shape, batch, prev_row, prev_col);
                    const adrt_scalar v12 = conv_b * adrt::_common::array_access(data, shape, batch, prev_row, col);
                    const adrt_scalar v13 = conv_a * adrt::_common::array_access(data, shape, batch, prev_row, next_col);
                    // Conv row 2
                    const adrt_scalar v21 = conv_b * adrt::_common::array_access(data, shape, batch, row, prev_col);
                    const adrt_scalar v22 = conv_c * adrt::_common::array_access(data, shape, batch, row, col);
                    const adrt_scalar v23 = conv_b * adrt::_common::array_access(data, shape, batch, row, next_col);
                    // Conv row 3
                    const adrt_scalar v31 = conv_a * adrt::_common::array_access(data, shape, batch, next_row, prev_col);
                    const adrt_scalar v32 = conv_b * adrt::_common::array_access(data, shape, batch, next_row, col);
                    const adrt_scalar v33 = conv_a * adrt::_common::array_access(data, shape, batch, next_row, next_col);
                    // Store result
                    adrt::_common::array_access(out, shape, batch, row, col) = (v11 + v21 + v31) + (v12 + v22 + v32) + (v13 + v23 + v33);
                }

                // Last col
                {
                    const size_t col = adrt::_common::get<2>(shape) - 1_uz;
                    const size_t prev_col = col - 1_uz;
                    // Conv row 1
                    const adrt_scalar v11 = conv_a * adrt::_common::array_access(data, shape, batch, prev_row, prev_col);
                    const adrt_scalar v12 = conv_b * adrt::_common::array_access(data, shape, batch, prev_row, col);
                    // Conv row 2
                    const adrt_scalar v21 = conv_b * adrt::_common::array_access(data, shape, batch, row, prev_col);
                    const adrt_scalar v22 = conv_c * adrt::_common::array_access(data, shape, batch, row, col);
                    // Conv row 3
                    const adrt_scalar v31 = conv_a * adrt::_common::array_access(data, shape, batch, next_row, prev_col);
                    const adrt_scalar v32 = conv_b * adrt::_common::array_access(data, shape, batch, next_row, col);
                    // Store result
                    adrt::_common::array_access(out, shape, batch, row, col) = (v11 + v21 + v31) + (v12 + v22 + v32) + (v11 + v21 + v31);
                }
            }
        }
    }

}

#endif // ADRT_CDEFS_FMG_H
