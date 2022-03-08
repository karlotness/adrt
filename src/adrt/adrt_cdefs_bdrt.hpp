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
#ifndef ADRT_CDEFS_BDRT_H
#define ADRT_CDEFS_BDRT_H

#include <array>
#include <utility>
#include <algorithm>
#include "adrt_cdefs_common.hpp"

namespace adrt {

    // Defined in: adrt_cdefs_common.cpp
    bool bdrt_is_valid_shape(const std::array<size_t, 4> &shape);
    bool bdrt_step_is_valid_shape(const std::array<size_t, 4> &shape);
    bool bdrt_step_is_valid_iter(const std::array<size_t, 4> &shape, int iter);
    std::array<size_t, 5> bdrt_buffer_shape(const std::array<size_t, 4> &shape);
    std::array<size_t, 4> bdrt_result_shape(const std::array<size_t, 4> &shape);
    std::array<size_t, 4> bdrt_step_result_shape(const std::array<size_t, 4> &shape);

    namespace _impl {

    template <typename adrt_scalar>
    std::array<size_t, 5> bdrt_core(const adrt_scalar *const ADRT_RESTRICT data, const std::array<size_t, 5> &in_shape, adrt_scalar *const ADRT_RESTRICT out) {
        ADRT_ASSERT(data)
        ADRT_ASSERT(out)

        const std::array<size_t, 5> curr_shape = {
            std::get<0>(in_shape), // Keep batch dimension
            4, // Always 4 quadrants
            std::get<2>(in_shape),
            adrt::_common::floor_div2(std::get<3>(in_shape)), // Halve the size of each section
            std::get<4>(in_shape) * size_t{2}, // Double the number of sections
        };

        ADRT_ASSERT(adrt::_assert::same_total_size(in_shape, curr_shape))

        ADRT_OPENMP("omp for collapse(5)")
        for(size_t batch = 0; batch < std::get<0>(curr_shape); ++batch) {
            for(size_t quadrant = 0; quadrant < 4u; ++quadrant) {
                for(size_t row = 0; row < std::get<2>(curr_shape); ++row) {
                    for(size_t sec_i = 0; sec_i < std::get<3>(curr_shape); ++sec_i) {
                        // We double the sections we store to so loop over the previous (half count) value
                        for(size_t section = 0; section < std::get<4>(in_shape); ++section) {
                            const size_t sec_left = size_t{2} * section;
                            const size_t sec_right = sec_left + size_t{1};

                            // Left section
                            const adrt_scalar la_val = adrt::_common::array_access(data, in_shape, batch, quadrant, row, size_t{2} * sec_i, section);
                            const adrt_scalar lb_val = adrt::_common::array_access(data, in_shape, batch, quadrant, row, size_t{2} * sec_i + size_t{1}, section);
                            adrt::_common::array_access(out, curr_shape, batch, quadrant, row, sec_i, sec_left) = la_val + lb_val;

                            // Right section
                            adrt_scalar ra_val = 0, rb_val = 0;
                            if(row + sec_i < std::get<2>(in_shape)) {
                                ra_val = adrt::_common::array_access(data, in_shape, batch, quadrant, row + sec_i, size_t{2} * sec_i, section);
                            }
                            if(row + sec_i + size_t{1} < std::get<2>(in_shape)) {
                                rb_val = adrt::_common::array_access(data, in_shape, batch, quadrant, row + sec_i + size_t{1}, size_t{2} * sec_i + size_t{1}, section);
                            }
                            adrt::_common::array_access(out, curr_shape, batch, quadrant, row, sec_i, sec_right) = ra_val + rb_val;
                        }
                    }
                }
            }
        }

        return curr_shape;
    }

    } // end namespace: adrt::_impl

    template <typename adrt_scalar>
    void bdrt_basic(const adrt_scalar *const ADRT_RESTRICT data, const std::array<size_t, 4> &shape, adrt_scalar *const ADRT_RESTRICT tmp, adrt_scalar *const ADRT_RESTRICT out) {
        ADRT_ASSERT(data)
        ADRT_ASSERT(tmp)
        ADRT_ASSERT(out)
        ADRT_ASSERT(adrt::bdrt_is_valid_shape(shape))
        ADRT_ASSERT(adrt::_assert::same_total_size(adrt::bdrt_result_shape(shape), adrt::bdrt_buffer_shape(shape)))

        const int num_iters = adrt::num_iters(std::get<3>(shape));
        const std::array<size_t, 4> output_shape = adrt::bdrt_result_shape(shape);

        ADRT_OPENMP("omp parallel default(none) shared(data, shape, tmp, out, num_iters, output_shape)")
        {
            // Choose the ordering of the two buffers so that we always end with result in tmp (ready to copy out)
            adrt_scalar *buf_a = tmp;
            adrt_scalar *buf_b = out;
            if(num_iters % 2 != 0) {
                std::swap(buf_a, buf_b);
            }
            std::array<size_t, 5> buf_shape = adrt::bdrt_buffer_shape(shape);

            // Copy data to tmp buffer (always load into buf_a)
            ADRT_OPENMP("omp for collapse(4)")
            for(size_t batch = 0; batch < std::get<0>(shape); ++batch) {
                for(size_t quadrant = 0; quadrant < 4u; ++quadrant) {
                    for(size_t row = 0; row < std::get<2>(shape); ++row) {
                        for(size_t col = 0; col < std::get<3>(shape); ++col) {
                            adrt::_common::array_access(buf_a, buf_shape, batch, quadrant, row, col, size_t{0}) =
                                adrt::_common::array_access(data, shape, batch, quadrant, row, col);
                        }
                    }
                }
            }

            // Perform computations
            for(int i = 0; i < num_iters; ++i) {
                buf_shape = adrt::_impl::bdrt_core(buf_a, buf_shape, buf_b);
                std::swap(buf_a, buf_b);
            }

            // Copy result to out buffer (always tmp -> out)
            ADRT_OPENMP("omp for collapse(4) nowait")
            for(size_t batch = 0; batch < std::get<0>(output_shape); ++batch) {
                for(size_t quadrant = 0; quadrant < 4u; ++quadrant) {
                    for(size_t row = 0; row < std::get<2>(output_shape); ++row) {
                        for(size_t col = 0; col < std::get<3>(output_shape); ++col) {
                            adrt::_common::array_access(out, output_shape, batch, quadrant, row, col) =
                                adrt::_common::array_access(tmp, buf_shape, batch, quadrant, row, size_t{0}, col);
                        }
                    }
                }
            }
        }
    }

    template <typename adrt_scalar>
    void bdrt_step(const adrt_scalar *const ADRT_RESTRICT data, const std::array<size_t, 4> &shape, adrt_scalar *const ADRT_RESTRICT out, int iter) {
        // Requires 0 <= iter < num_iters(n), must be checked elsewhere
        ADRT_ASSERT(data)
        ADRT_ASSERT(out)
        ADRT_ASSERT(adrt::bdrt_step_is_valid_shape(shape))
        ADRT_ASSERT(adrt::bdrt_step_is_valid_iter(shape, iter))
        ADRT_ASSERT(shape == adrt::bdrt_step_result_shape(shape))

        const int adrt_iter = adrt::num_iters(std::get<3>(shape)) - iter - 1;
        const size_t iter_exp = size_t{1} << adrt_iter;
        const size_t num_col_blocks = adrt::_common::ceil_div(std::get<3>(shape), iter_exp);

        ADRT_OPENMP("omp parallel for collapse(4) default(none) shared(data, shape, out, iter_exp, num_col_blocks)")
        for(size_t batch = 0; batch < std::get<0>(shape); ++batch) {
            for(size_t quadrant = 0; quadrant < 4u; ++quadrant) {
                for(size_t row = 0; row < std::get<2>(shape); ++row) {
                    for(size_t col_block = 0; col_block < num_col_blocks; ++col_block) {
                        const size_t col_start = col_block * iter_exp;
                        const size_t max_col_i = std::min(iter_exp, std::get<3>(shape) - col_start);
                        const size_t base_col_sec_i = iter_exp * adrt::_common::floor_div2(col_block);
                        if(col_block % size_t{2} == 0u) {
                            // Case: same row
                            for(size_t col_i = 0; col_i < max_col_i; ++col_i) {
                                const size_t col = col_start + col_i;
                                const size_t col_sec_i = col_i + base_col_sec_i;
                                const size_t base_col = size_t{2} * col_sec_i;
                                const adrt_scalar aval = adrt::_common::array_access(data, shape, batch, quadrant, row, base_col);
                                const adrt_scalar bval = adrt::_common::array_access(data, shape, batch, quadrant, row, base_col + size_t{1});
                                adrt::_common::array_access(out, shape, batch, quadrant, row, col) = aval + bval;
                            }
                        }
                        else {
                            // Case: different rows
                            for(size_t col_i = 0; col_i < max_col_i; ++col_i) {
                                const size_t col = col_start + col_i;
                                const size_t col_sec_i = col_i + base_col_sec_i;
                                const size_t base_col = size_t{2} * col_sec_i;
                                const size_t base_row = row + col_i;
                                adrt_scalar aval = 0;
                                adrt_scalar bval = 0;
                                if(base_row < std::get<2>(shape)) {
                                    aval = adrt::_common::array_access(data, shape, batch, quadrant, base_row, base_col);
                                }
                                if(base_row + size_t{1} < std::get<2>(shape)) {
                                    bval = adrt::_common::array_access(data, shape, batch, quadrant, base_row + size_t{1}, base_col + size_t{1});
                                }
                                adrt::_common::array_access(out, shape, batch, quadrant, row, col) = aval + bval;
                            }
                        }
                    }
                }
            }
        }
    }

}

#endif // ADRT_CDEFS_BDRT_H
