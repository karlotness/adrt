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
#ifndef ADRT_CDEFS_ADRT_H
#define ADRT_CDEFS_ADRT_H

#include <array>
#include <utility>
#include <algorithm>
#include "adrt_cdefs_common.hpp"

namespace adrt {

    // Defined in: adrt_cdefs_common.cpp
    bool adrt_is_valid_shape(const std::array<size_t, 3> &shape);
    std::array<size_t, 5> adrt_buffer_shape(const std::array<size_t, 3> &shape);
    std::array<size_t, 4> adrt_result_shape(const std::array<size_t, 3> &shape);
    // TODO: bool adrt_core_is_valid_shape(const std::array<size_t, 5> &shape);

    template <typename adrt_scalar>
    std::array<size_t, 5> adrt_core(const adrt_scalar *const ADRT_RESTRICT data, const std::array<size_t, 5> &in_shape, adrt_scalar *const ADRT_RESTRICT out) {
        const std::array<size_t, 5> curr_shape = {
            in_shape[0], // Keep batch dimension
            4, // Always 4 quadrants
            adrt::_common::floor_div2(in_shape[2]), // We halve the number of rows
            in_shape[3], // Keep the same number of columns
            in_shape[4] * 2, // The number of angles doubles
        };

        const size_t block_stride = 16;

        ADRT_OPENMP("omp for collapse(5)")
        for(size_t batch = 0; batch < curr_shape[0]; ++batch) {
            for(size_t quadrant = 0; quadrant < 4; ++quadrant) {
                for(size_t row = 0; row < curr_shape[2]; ++row) {
                    for(size_t col_start = 0; col_start < curr_shape[3]; col_start += block_stride) {
                        for(size_t angle_start = 0; angle_start < curr_shape[4]; angle_start += block_stride) {
                            // Inner loops inside each bock
                            for(size_t col = col_start; col < std::min(col_start + block_stride, curr_shape[3]); ++col) {
                                for(size_t angle = angle_start; angle < std::min(angle_start + block_stride, curr_shape[4]); ++angle) {
                                    // TODO: Adjust loop bounds to avoid operations on all zeros. Will likely have to fuse the iterations by hand.
                                    adrt_scalar aval = adrt::_common::array_access(data, in_shape, batch, quadrant, 2 * row, col, adrt::_common::floor_div2(angle));
                                    // Need to check the index access for x
                                    const size_t b_col_idx = col - adrt::_common::ceil_div2(angle);
                                    adrt_scalar bval = 0;
                                    if(col >= adrt::_common::ceil_div2(angle) && b_col_idx < curr_shape[3]) {
                                        bval = adrt::_common::array_access(data, in_shape, batch, quadrant, (2 * row) + 1, b_col_idx, adrt::_common::floor_div2(angle));
                                    }
                                    adrt::_common::array_access(out, curr_shape, batch, quadrant, row, col, angle) = aval + bval;
                                }
                            }
                        }
                    }
                }
            }
        }

        return curr_shape;
    }

    template <typename adrt_scalar>
    void adrt_basic(const adrt_scalar *const ADRT_RESTRICT data, const std::array<size_t, 3> &shape, adrt_scalar *const ADRT_RESTRICT tmp, adrt_scalar *const ADRT_RESTRICT out) {
        const int num_iters = adrt::num_iters(shape[1]);
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
            for(size_t batch = 0; batch < shape[0]; ++batch) {
                for(size_t row = 0; row < shape[1]; ++row) {
                    for(size_t col = 0; col < shape[2]; ++col) {
                        adrt::_common::array_access(buf_a, buf_shape, batch, size_t{0}, row, shape[2] - col - 1, size_t{0}) = \
                            adrt::_common::array_access(data, shape, batch, row, col);
                    }
                }
            }
            // QUADRANT 1 (Transpose the squares)
            ADRT_OPENMP("omp for collapse(3) nowait")
            for(size_t batch = 0; batch < shape[0]; ++batch) {
                for(size_t row_start = 0; row_start < shape[1]; row_start += block_stride) {
                    for(size_t col_start = 0; col_start < shape[2]; col_start += block_stride) {
                        // Transpose inside each block
                        for(size_t row = row_start; row < std::min(row_start + block_stride, shape[1]); ++row) {
                            for(size_t col = col_start; col < std::min(col_start + block_stride, shape[2]); ++col) {
                                adrt::_common::array_access(buf_a, buf_shape, batch, size_t{1}, shape[1] - row - 1, shape[2] - col - 1, size_t{0}) = \
                                    adrt::_common::array_access(data, shape, batch, col, shape[1] - row - 1);
                            }
                        }
                    }
                }
            }
            // QUADRANT 2 (Transpose the squares and flip along x)
            ADRT_OPENMP("omp for collapse(3) nowait")
            for(size_t batch = 0; batch < shape[0]; ++batch) {
                for(size_t row_start = 0; row_start < shape[1]; row_start += block_stride) {
                    for(size_t col_start = 0; col_start < shape[2]; col_start += block_stride) {
                        // Transpose inside each block
                        for(size_t row = row_start; row < std::min(row_start + block_stride, shape[1]); ++row) {
                            for(size_t col = col_start; col < std::min(col_start + block_stride, shape[2]); ++col) {
                                adrt::_common::array_access(buf_a, buf_shape, batch, size_t{2}, shape[1] - row - 1, shape[2] - col - 1, size_t{0}) = \
                                    adrt::_common::array_access(data, shape, batch, shape[2] - col - 1, shape[1] - row - 1);
                            }
                        }
                    }
                }
            }
            // QUADRANT 3 (Flip along y)
            ADRT_OPENMP("omp for collapse(3) nowait")
            for(size_t batch = 0; batch < shape[0]; ++batch) {
                for(size_t row = 0; row < shape[1]; ++row) {
                    for(size_t col = 0; col < shape[2]; ++col) {
                        adrt::_common::array_access(buf_a, buf_shape, batch, size_t{3}, row, shape[2] - col - 1, size_t{0}) = \
                            adrt::_common::array_access(data, shape, batch, shape[1] - row - 1, col);
                    }
                }
            }
            // Fill rest with zeros
            ADRT_OPENMP("omp for collapse(4)")
            for(size_t batch = 0; batch < shape[0]; ++batch) {
                for(size_t quadrant = 0; quadrant < 4; ++quadrant) {
                    for(size_t row = 0; row < shape[1]; ++row) {
                        for(size_t col = shape[2]; col < 2 * shape[2] - 1; ++col) {
                            adrt::_common::array_access(buf_a, buf_shape, batch, quadrant, row, col, size_t{0}) = 0;
                        }
                    }
                }
            }

            // Perform computations
            for(int i = 0; i < num_iters; ++i) {
                buf_shape = adrt::adrt_core(buf_a, buf_shape, buf_b);
                std::swap(buf_a, buf_b);
            }

            // Copy result to out buffer (always tmp -> out)
            ADRT_OPENMP("omp for collapse(4) nowait")
            for(size_t batch = 0; batch < output_shape[0]; ++batch) {
                for(size_t quadrant = 0; quadrant < 4; ++quadrant) {
                    for(size_t d = 0; d < output_shape[2]; ++d) {
                        for(size_t a = 0; a < output_shape[3]; ++a) {
                            const adrt_scalar val = adrt::_common::array_access(tmp, buf_shape, batch, quadrant, size_t{0}, d, a);
                            adrt::_common::array_access(out, output_shape, batch, quadrant, d, a) = val;
                        }
                    }
                }
            }
        }
    }

}

#endif // ADRT_CDEFS_ADRT_H
