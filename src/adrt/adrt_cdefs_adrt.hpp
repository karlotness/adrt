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

#include "adrt_cdefs_common.hpp"
#include <array>
#include <utility>

namespace adrt {

    // Defined in: adrt_cdefs_common.cpp
    bool adrt_is_valid_shape(const std::array<size_t, 3> &shape);
    std::array<size_t, 5> adrt_buffer_shape(const std::array<size_t, 3> &shape);
    std::array<size_t, 4> adrt_result_shape(const std::array<size_t, 3> &shape);
    // TODO: bool adrt_core_is_valid_shape(const std::array<size_t, 5> &shape);

    template <typename adrt_scalar>
    std::array<size_t, 5> adrt_core(const adrt_scalar *const data, const std::array<size_t, 5> &in_shape, adrt_scalar *const out) {
        const std::array<size_t, 5> curr_shape = {
            in_shape[0], // Keep batch dimension
            4, // Always 4 quadrants
            adrt::_common::floor_div2(in_shape[2]), // We halve the number of rows
            in_shape[3], // Keep the same number of columns
            in_shape[4] * 2, // The number of angles doubles
        };

        #pragma omp parallel for collapse(5) default(none) shared(data, out, in_shape, curr_shape)
        for(size_t batch = 0; batch < curr_shape[0]; ++batch) {
            for(size_t quadrant = 0; quadrant < 4; ++quadrant) {
                for(size_t row = 0; row < curr_shape[2]; ++row) {
                    for(size_t col = 0; col < curr_shape[3]; ++col) {
                        for(size_t angle = 0; angle < curr_shape[4]; ++angle) {
                            const size_t j = row, x = col, a = angle;
                            // TODO: Adjust loop bounds to avoid operations on all zeros. This will make x depend on the angle.
                            // Will likely have to fuse the iterations by hand
                            adrt_scalar aval = adrt::_common::array_access(data, in_shape, batch, quadrant, 2 * j, x, adrt::_common::floor_div2(a));
                            // Need to check the index access for x
                            const size_t xb_idx = x - adrt::_common::ceil_div2(a);
                            adrt_scalar bval = 0;
                            if(x >= adrt::_common::ceil_div2(a) && xb_idx < curr_shape[3]) {
                                bval = adrt::_common::array_access(data, in_shape, batch, quadrant, (2 * j) + 1, xb_idx, adrt::_common::floor_div2(a));
                            }
                            adrt::_common::array_access(out, curr_shape, batch, quadrant, j, x, a) = aval + bval;
                        }
                    }
                }
            }
        }

        return curr_shape;
    }

    template <typename adrt_scalar>
    void adrt_basic(const adrt_scalar *const data, const std::array<size_t, 3> &shape, adrt_scalar *const tmp, adrt_scalar *const out) {
        const int num_iters = adrt::num_iters(shape[1]);

        // Choose the ordering of the two buffers so that we always end with result in tmp (ready to copy out)
        adrt_scalar *buf_a = tmp;
        adrt_scalar *buf_b = out;
        if(num_iters % 2 != 0) {
            std::swap(buf_a, buf_b);
        }
        std::array<size_t, 5> buf_shape = adrt::adrt_buffer_shape(shape);

        // Copy data to tmp buffer (always load into buf_a)
        // QUADRANT 0 (Direct copy row by row)
        for(size_t batch = 0; batch < shape[0]; ++batch) {
            for(size_t row = 0; row < shape[1]; ++row) {
                for(size_t col = 0; col < shape[2]; ++col) {
                    adrt::_common::array_access(buf_a, buf_shape, batch, size_t{0}, row, shape[2] - col - 1, size_t{0}) = \
                        adrt::_common::array_access(data, shape, batch, row, col);
                }
            }
        }
        // QUADRANT 1 (Transpose the squares)
        for(size_t batch = 0; batch < shape[0]; ++batch) {
            for(size_t row = 0; row < shape[1]; ++row) {
                for(size_t col = 0; col < shape[2]; ++col) {
                    adrt::_common::array_access(buf_a, buf_shape, batch, size_t{1}, shape[1] - row - 1, shape[2] - col - 1, size_t{0}) = \
                        adrt::_common::array_access(data, shape, batch, col, shape[1] - row - 1);
                }
            }
        }
        // QUADRANT 2 (Transpose the squares and flip along x)
        for(size_t batch = 0; batch < shape[0]; ++batch) {
            for(size_t row = 0; row < shape[1]; ++row) {
                for(size_t col = 0; col < shape[2]; ++col) {
                    adrt::_common::array_access(buf_a, buf_shape, batch, size_t{2}, shape[1] - row - 1, shape[2] - col - 1, size_t{0}) = \
                        adrt::_common::array_access(data, shape, batch, shape[2] - col - 1, shape[1] - row - 1);
                }
            }
        }
        // QUADRANT 3 (Flip along y)
        for(size_t batch = 0; batch < shape[0]; ++batch) {
            for(size_t row = 0; row < shape[1]; ++row) {
                for(size_t col = 0; col < shape[2]; ++col) {
                    adrt::_common::array_access(buf_a, buf_shape, batch, size_t{3}, row, shape[2] - col - 1, size_t{0}) = \
                        adrt::_common::array_access(data, shape, batch, shape[1] - row - 1, col);
                }
            }
        }
        // Fill rest with zeros
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
        std::array<size_t, 4> output_shape = adrt::adrt_result_shape(shape);
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

#endif // ADRT_CDEFS_ADRT_H
