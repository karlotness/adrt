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

#ifndef ADRT_CDEFS_ADRT_H
#define ADRT_CDEFS_ADRT_H

#include <array>
#include <span>
#include <utility>
#include <algorithm>
#include <cassert>
#include "adrt_cdefs_common.hpp"

namespace adrt {

    // Defined in: adrt_cdefs_common.cpp
    bool adrt_is_valid_shape(std::span<const size_t, 3> shape);
    bool adrt_step_is_valid_shape(std::span<const size_t, 4> shape);
    bool adrt_step_is_valid_iter(std::span<const size_t, 4> shape, int iter);
    std::array<size_t, 5> adrt_buffer_shape(std::span<const size_t, 3> shape);
    std::array<size_t, 4> adrt_result_shape(std::span<const size_t, 3> shape);
    std::array<size_t, 4> adrt_step_result_shape(std::span<const size_t, 4> shape);

    namespace _impl {

    template <typename adrt_scalar>
    std::array<size_t, 5> adrt_core(const adrt_scalar *const ADRT_RESTRICT data, std::span<const size_t, 5> in_shape, adrt_scalar *const ADRT_RESTRICT out) {
        assert(data);
        assert(out);

        const std::array<size_t, 5> curr_shape = {
            adrt::_common::get<0>(in_shape), // Keep batch dimension
            4, // Always 4 quadrants
            adrt::_common::floor_div2(adrt::_common::get<2>(in_shape)), // We halve the number of rows
            adrt::_common::get<3>(in_shape) * 2_uz, // The number of angles doubles
            adrt::_common::get<4>(in_shape), // Keep the same number of columns
        };

        assert(adrt::_assert::same_total_size(in_shape, curr_shape));

        ADRT_OPENMP("omp for collapse(4)")
        for(size_t batch = 0; batch < adrt::_common::get<0>(curr_shape); ++batch) {
            for(size_t quadrant = 0; quadrant < 4u; ++quadrant) {
                for(size_t row = 0; row < adrt::_common::get<2>(curr_shape); ++row) {
                    for(size_t angle = 0; angle < adrt::_common::get<3>(curr_shape); ++angle) {
                        // Pair of loops below split at ceil(angle/2) to avoid extra bounds check in loop body
                        const size_t ceil_div2_angle = adrt::_common::ceil_div2(angle);
                        ADRT_OPENMP("omp simd")
                        for(size_t col = 0; col < ceil_div2_angle; ++col) {
                            const adrt_scalar aval = adrt::_common::array_access(data, in_shape, batch, quadrant, 2_uz * row, adrt::_common::floor_div2(angle), col);
                            adrt::_common::array_access(out, curr_shape, batch, quadrant, row, angle, col) = aval;
                        }
                        // This second loop requires col >= ceil(angle/2) to avoid bounds check
                        ADRT_OPENMP("omp simd")
                        for(size_t col = ceil_div2_angle; col < adrt::_common::get<4>(curr_shape); ++col) {
                            const adrt_scalar aval = adrt::_common::array_access(data, in_shape, batch, quadrant, 2_uz * row, adrt::_common::floor_div2(angle), col);
                            const size_t b_col_idx = col - ceil_div2_angle;
                            const adrt_scalar bval = adrt::_common::array_access(data, in_shape, batch, quadrant, (2_uz * row) + 1_uz, adrt::_common::floor_div2(angle), b_col_idx);
                            adrt::_common::array_access(out, curr_shape, batch, quadrant, row, angle, col) = aval + bval;
                        }
                    }
                }
            }
        }

        return curr_shape;
    }

    } // end namespace: adrt::_impl

    // DOC ANCHOR: adrt.adrt +2
    template <typename adrt_scalar>
    void adrt_basic(const adrt_scalar *const ADRT_RESTRICT data, std::span<const size_t, 3> shape, adrt_scalar *const ADRT_RESTRICT tmp, adrt_scalar *const ADRT_RESTRICT out) {
        assert(data);
        assert(tmp);
        assert(out);
        assert(adrt::adrt_is_valid_shape(shape));
        assert(adrt::_assert::same_total_size(adrt::adrt_result_shape(shape), adrt::adrt_buffer_shape(shape)));

        const int num_iters = adrt::num_iters(adrt::_common::get<2>(shape));
        const std::array<size_t, 4> output_shape = adrt::adrt_result_shape(shape);

        ADRT_OPENMP("omp parallel default(none) shared(data, shape, tmp, out, num_iters, output_shape)")
        {
            // Choose the ordering of the two buffers so that we always end with result in tmp (ready to copy out)
            adrt_scalar *buf_a = tmp;
            adrt_scalar *buf_b = out;
            if(num_iters % 2 != 0) {
                std::swap(buf_a, buf_b);
            }
            std::array<size_t, 5> buf_shape = adrt::adrt_buffer_shape(shape);
            const size_t block_stride = 16;

            // Copy data to tmp buffer (always load into buf_a)
            // QUADRANT 0 (Direct copy row by row)
            ADRT_OPENMP("omp for collapse(3) nowait")
            for(size_t batch = 0; batch < adrt::_common::get<0>(shape); ++batch) {
                for(size_t row = 0; row < adrt::_common::get<1>(shape); ++row) {
                    for(size_t col = 0; col < adrt::_common::get<2>(shape); ++col) {
                        adrt::_common::array_access(buf_a, buf_shape, batch, 0_uz, row, 0_uz, adrt::_common::get<2>(shape) - col - 1_uz) =
                            adrt::_common::array_access(data, shape, batch, row, col);
                    }
                }
            }
            // QUADRANT 1 (Transpose the squares)
            ADRT_OPENMP("omp for collapse(3) nowait")
            for(size_t batch = 0; batch < adrt::_common::get<0>(shape); ++batch) {
                // Note: no overflow here (or in other blocked loops) because very large shapes (> size_t_max - 16) are impossible
                // The input array must be square and with that dimension, the input would be too large to exist
                for(size_t row_start = 0; row_start < adrt::_common::get<1>(shape); row_start += block_stride) {
                    for(size_t col_start = 0; col_start < adrt::_common::get<2>(shape); col_start += block_stride) {
                        // Transpose inside each block
                        for(size_t row = row_start; row < std::min(row_start + block_stride, adrt::_common::get<1>(shape)); ++row) {
                            for(size_t col = col_start; col < std::min(col_start + block_stride, adrt::_common::get<2>(shape)); ++col) {
                                adrt::_common::array_access(buf_a, buf_shape, batch, 1_uz, adrt::_common::get<1>(shape) - row - 1_uz, 0_uz, adrt::_common::get<2>(shape) - col - 1_uz) =
                                    adrt::_common::array_access(data, shape, batch, col, adrt::_common::get<1>(shape) - row - 1_uz);
                            }
                        }
                    }
                }
            }
            // QUADRANT 2 (Transpose the squares and flip along x)
            ADRT_OPENMP("omp for collapse(3) nowait")
            for(size_t batch = 0; batch < adrt::_common::get<0>(shape); ++batch) {
                for(size_t row_start = 0; row_start < adrt::_common::get<1>(shape); row_start += block_stride) {
                    for(size_t col_start = 0; col_start < adrt::_common::get<2>(shape); col_start += block_stride) {
                        // Transpose inside each block
                        for(size_t row = row_start; row < std::min(row_start + block_stride, adrt::_common::get<1>(shape)); ++row) {
                            for(size_t col = col_start; col < std::min(col_start + block_stride, adrt::_common::get<2>(shape)); ++col) {
                                adrt::_common::array_access(buf_a, buf_shape, batch, 2_uz, adrt::_common::get<1>(shape) - row - 1_uz, 0_uz, adrt::_common::get<2>(shape) - col - 1_uz) =
                                    adrt::_common::array_access(data, shape, batch, adrt::_common::get<2>(shape) - col - 1_uz, adrt::_common::get<1>(shape) - row - 1_uz);
                            }
                        }
                    }
                }
            }
            // QUADRANT 3 (Flip along y)
            ADRT_OPENMP("omp for collapse(3) nowait")
            for(size_t batch = 0; batch < adrt::_common::get<0>(shape); ++batch) {
                for(size_t row = 0; row < adrt::_common::get<1>(shape); ++row) {
                    for(size_t col = 0; col < adrt::_common::get<2>(shape); ++col) {
                        adrt::_common::array_access(buf_a, buf_shape, batch, 3_uz, row, 0_uz, adrt::_common::get<2>(shape) - col - 1_uz) =
                            adrt::_common::array_access(data, shape, batch, adrt::_common::get<1>(shape) - row - 1_uz, col);
                    }
                }
            }
            // Fill rest with zeros
            ADRT_OPENMP("omp for collapse(4)")
            for(size_t batch = 0; batch < adrt::_common::get<0>(shape); ++batch) {
                for(size_t quadrant = 0; quadrant < 4u; ++quadrant) {
                    for(size_t row = 0; row < adrt::_common::get<1>(shape); ++row) {
                        for(size_t col = adrt::_common::get<2>(shape); col < 2_uz * adrt::_common::get<2>(shape) - 1_uz; ++col) {
                            adrt::_common::array_access(buf_a, buf_shape, batch, quadrant, row, 0_uz, col) = 0;
                        }
                    }
                }
            }

            // Perform computations
            for(int i = 0; i < num_iters; ++i) {
                buf_shape = adrt::_impl::adrt_core(buf_a, buf_shape, buf_b);
                std::swap(buf_a, buf_b);
            }

            // Copy result to out buffer (always tmp -> out)
            ADRT_OPENMP("omp for collapse(4) nowait")
            for(size_t batch = 0; batch < adrt::_common::get<0>(output_shape); ++batch) {
                for(size_t quadrant = 0; quadrant < 4u; ++quadrant) {
                    for(size_t d_start = 0; d_start < adrt::_common::get<2>(output_shape); d_start += block_stride) {
                        for(size_t a_start = 0; a_start < adrt::_common::get<3>(output_shape); a_start += block_stride) {
                            // Inner blocks serial
                            for(size_t d = d_start; d < std::min(d_start + block_stride, adrt::_common::get<2>(output_shape)); ++d) {
                                for(size_t a = a_start; a < std::min(a_start + block_stride, adrt::_common::get<3>(output_shape)); ++a) {
                                    const adrt_scalar val = adrt::_common::array_access(tmp, buf_shape, batch, quadrant, 0_uz, a, d);
                                    adrt::_common::array_access(out, output_shape, batch, quadrant, d, a) = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // DOC ANCHOR: adrt.core.adrt_step +2
    template <typename adrt_scalar>
    void adrt_step(const adrt_scalar *const ADRT_RESTRICT data, std::span<const size_t, 4> shape, adrt_scalar *const ADRT_RESTRICT out, int iter) {
        // Requires 0 <= iter < num_iters(n), must be checked elsewhere
        assert(data);
        assert(out);
        assert(adrt::adrt_step_is_valid_shape(shape));
        assert(adrt::adrt_step_is_valid_iter(shape, iter));
        assert(std::ranges::equal(shape, adrt::adrt_step_result_shape(shape)));

        const size_t iter_exp = 1_uz << iter;
        const size_t iter_exp_next = 1_uz << (iter + 1);
        const size_t num_col_blocks = adrt::_common::ceil_div(adrt::_common::get<3>(shape), iter_exp_next);

        ADRT_OPENMP("omp parallel for collapse(4) default(none) shared(data, shape, out, iter_exp, iter_exp_next, num_col_blocks)")
        for(size_t batch = 0; batch < adrt::_common::get<0>(shape); ++batch) {
            for(size_t quadrant = 0; quadrant < 4u; ++quadrant) {
                for(size_t row = 0; row < adrt::_common::get<2>(shape); ++row) {
                    for(size_t col_block = 0; col_block < num_col_blocks; ++col_block) {
                        const size_t col_start = col_block * iter_exp_next;
                        const size_t out_offset = col_block;
                        const size_t max_col_i = std::min(iter_exp_next, adrt::_common::get<3>(shape) - col_start);
                        const size_t col_split = std::min(2_uz * row + 1_uz, max_col_i);
                        for(size_t col_i = 0; col_i < col_split; ++col_i) {
                            const size_t col = col_start + col_i;
                            const size_t out_angle = col_i;
                            const size_t a_col_idx = (2_uz * out_offset) * iter_exp + adrt::_common::floor_div2(out_angle);
                            const adrt_scalar aval = adrt::_common::array_access(data, shape, batch, quadrant, row, a_col_idx);
                            const size_t b_col_idx = (2_uz * out_offset + 1_uz) * iter_exp + adrt::_common::floor_div2(out_angle);
                            const size_t b_row_idx = row - adrt::_common::ceil_div2(out_angle);
                            const adrt_scalar bval = adrt::_common::array_access(data, shape, batch, quadrant, b_row_idx, b_col_idx);
                            adrt::_common::array_access(out, shape, batch, quadrant, row, col) = aval + bval;
                        }
                        for(size_t col_i = col_split; col_i < max_col_i; ++col_i) {
                            const size_t col = col_start + col_i;
                            const size_t out_angle = col_i;
                            const size_t a_col_idx = (2_uz * out_offset) * iter_exp + adrt::_common::floor_div2(out_angle);
                            const adrt_scalar aval = adrt::_common::array_access(data, shape, batch, quadrant, row, a_col_idx);
                            adrt::_common::array_access(out, shape, batch, quadrant, row, col) = aval;
                        }
                    }
                }
            }
        }
    }

}

#endif // ADRT_CDEFS_ADRT_H
