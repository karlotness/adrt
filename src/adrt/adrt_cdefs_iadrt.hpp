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

#ifndef ADRT_CDEFS_IADRT_H
#define ADRT_CDEFS_IADRT_H

#include <array>
#include <span>
#include <utility>
#include <algorithm>
#include <cassert>
#include "adrt_cdefs_common.hpp"

namespace adrt {

    // Defined in: adrt_cdefs_common.cpp
    bool iadrt_is_valid_shape(std::span<const size_t, 4> shape);
    std::array<size_t, 5> iadrt_buffer_shape(std::span<const size_t, 4> shape);
    std::array<size_t, 4> iadrt_result_shape(std::span<const size_t, 4> shape);

    namespace _impl {

    template <typename adrt_scalar>
    std::array<size_t, 5> iadrt_core(const adrt_scalar *const ADRT_RESTRICT data, std::span<const size_t, 5> in_shape, adrt_scalar *const ADRT_RESTRICT out) {
        assert(data);
        assert(out);

        const std::array<size_t, 5> curr_shape = {
            adrt::_common::get<0>(in_shape), // Keep batch dimension
            4, // Always 4 quadrants
            adrt::_common::get<2>(in_shape) * 2_uz, // Double the number of processed "columns"
            adrt::_common::floor_div2(adrt::_common::get<3>(in_shape)), // We halve the number of "columns"
            adrt::_common::get<4>(in_shape), // Keep the same number of "rows"
        };

        assert(adrt::_assert::same_total_size(in_shape, curr_shape));

        ADRT_OPENMP("omp for collapse(4)")
        for(size_t batch = 0; batch < adrt::_common::get<0>(curr_shape); ++batch) {
            for(size_t quadrant = 0; quadrant < 4u; ++quadrant) {
                for(size_t l = 0; l < adrt::_common::get<2>(curr_shape); ++l) {
                    for(size_t col = 0; col < adrt::_common::get<3>(curr_shape); ++col) {
                        const size_t prev_l = adrt::_common::floor_div2(l);
                        // The loop below must be serial
                        for(size_t rev_row = 0; rev_row < adrt::_common::get<4>(curr_shape); ++rev_row) {
                            const size_t row = adrt::_common::get<4>(curr_shape) - rev_row - 1_uz;
                            adrt_scalar val = 0;
                            if(l % 2_uz == 0u) {
                                // l + 1 odd
                                val += adrt::_common::array_access(data, in_shape, batch, quadrant, prev_l, 2_uz * col, row);
                                if(row + 1_uz < adrt::_common::get<4>(in_shape) && 2_uz * col + 1_uz < adrt::_common::get<3>(in_shape)) {
                                    val -= adrt::_common::array_access(data, in_shape, batch, quadrant, prev_l, 2_uz * col + 1_uz, row + 1_uz);
                                }
                            }
                            else {
                                // l + 1 even
                                if(row + 1_uz + col < adrt::_common::get<4>(in_shape)) {
                                    if(2_uz * col + 1_uz < adrt::_common::get<3>(in_shape)){
                                        val += adrt::_common::array_access(data, in_shape, batch, quadrant, prev_l, 2_uz * col + 1_uz, row + 1_uz + col);
                                    }
                                    val -= adrt::_common::array_access(data, in_shape, batch, quadrant, prev_l, 2_uz * col, row + 1_uz + col);
                                }
                            }
                            if(row + 1_uz < adrt::_common::get<4>(curr_shape)) {
                                // Must ensure previous values are written before reading below (requires row loop to be serial)
                                val += adrt::_common::array_access(out, curr_shape, batch, quadrant, l, col, row + 1_uz);
                            }
                            adrt::_common::array_access(out, curr_shape, batch, quadrant, l, col, row) = val;
                        }
                    }
                }
            }
        }

        return curr_shape;
    }

    } // end namespace: adrt::_impl

    // DOC ANCHOR: adrt.iadrt +2
    template <typename adrt_scalar>
    void iadrt_basic(const adrt_scalar *const ADRT_RESTRICT data, std::span<const size_t, 4> shape, adrt_scalar *const ADRT_RESTRICT tmp, adrt_scalar *const ADRT_RESTRICT out) {
        assert(data);
        assert(tmp);
        assert(out);
        assert(adrt::iadrt_is_valid_shape(shape));
        assert(adrt::_assert::same_total_size(adrt::iadrt_result_shape(shape), adrt::iadrt_buffer_shape(shape)));

        const int num_iters = adrt::num_iters(adrt::_common::get<3>(shape));
        const std::array<size_t, 4> output_shape = adrt::iadrt_result_shape(shape);

        ADRT_OPENMP("omp parallel default(none) shared(data, shape, tmp, out, num_iters, output_shape)")
        {
            // Choose the ordering of the two buffers so that we always end with result in tmp (ready to copy out)
            adrt_scalar *buf_a = tmp;
            adrt_scalar *buf_b = out;
            if(num_iters % 2 != 0) {
                std::swap(buf_a, buf_b);
            }
            std::array<size_t, 5> buf_shape = adrt::iadrt_buffer_shape(shape);
            const size_t block_stride = 16;

            // Copy data to tmp buffer (always load into buf_a)
            ADRT_OPENMP("omp for collapse(4)")
            for(size_t batch = 0; batch < adrt::_common::get<0>(shape); ++batch) {
                for(size_t quadrant = 0; quadrant < 4u; ++quadrant) {
                    // Note: no overflow here (or in other blocked loop) because very large shapes (> size_t_max - 16) are impossible
                    // The input array must be (2 * N - 1)-by-N. With that dimension the buffer is too large to exist
                    for(size_t c_start = 0; c_start < adrt::_common::get<3>(shape); c_start += block_stride) {
                        for(size_t r_start = 0; r_start < adrt::_common::get<2>(shape); r_start += block_stride) {
                            // Transpose inside each block (serial)
                            for(size_t c = c_start; c < std::min(c_start + block_stride, adrt::_common::get<3>(shape)); ++c) {
                                for(size_t r = r_start; r < std::min(r_start + block_stride, adrt::_common::get<2>(shape)); ++r) {
                                    adrt::_common::array_access(buf_a, buf_shape, batch, quadrant, 0_uz, c, r) =
                                        adrt::_common::array_access(data, shape, batch, quadrant, r, c);
                                }
                            }
                        }
                    }
                }
            }

            // Perform computations
            for(int i = 0; i < num_iters; ++i) {
                buf_shape = adrt::_impl::iadrt_core(buf_a, buf_shape, buf_b);
                std::swap(buf_a, buf_b);
            }

            // Copy result to out buffer (always tmp -> out)
            ADRT_OPENMP("omp for collapse(4) nowait")
            for(size_t batch = 0; batch < adrt::_common::get<0>(output_shape); ++batch) {
                for(size_t quadrant = 0; quadrant < 4u; ++quadrant) {
                    for(size_t r_start = 0; r_start < adrt::_common::get<2>(output_shape); r_start += block_stride) {
                        for(size_t c_start = 0; c_start < adrt::_common::get<3>(output_shape); c_start += block_stride) {
                            // Transpose inside each block (serial)
                            for(size_t r = r_start; r < std::min(r_start + block_stride, adrt::_common::get<2>(output_shape)); ++r) {
                                for(size_t c = c_start; c < std::min(c_start + block_stride, adrt::_common::get<3>(output_shape)); ++c) {
                                    adrt::_common::array_access(out, output_shape, batch, quadrant, r, c) =
                                        adrt::_common::array_access(tmp, buf_shape, batch, quadrant, c, 0_uz, r);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}

#endif // ADRT_CDEFS_IADRT_H
