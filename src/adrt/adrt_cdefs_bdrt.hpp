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
#ifndef ADRT_CDEFS_BDRT_H
#define ADRT_CDEFS_BDRT_H

#include "adrt_cdefs_common.hpp"
#include <array>
#include <utility>

namespace adrt {

    // Defined in: adrt_cdefs_common.cpp
    bool bdrt_is_valid_shape(const std::array<size_t, 4> &shape);
    std::array<size_t, 5> bdrt_buffer_shape(const std::array<size_t, 4> &shape);
    std::array<size_t, 4> bdrt_result_shape(const std::array<size_t, 4> &shape);
    // TODO: bool bdrt_core_is_valid_shape(const std::array<size_t, 5> &shape);

    template <typename adrt_scalar>
    std::array<size_t, 5> bdrt_core(const adrt_scalar *const data, const std::array<size_t, 5> &in_shape, adrt_scalar *const out) {
        const std::array<size_t, 5> curr_shape = {
            in_shape[0], // Keep batch dimension
            4, // Always 4 quadrants
            in_shape[2],
            adrt::_common::floor_div2(in_shape[3]), // Halve the size of each section
            in_shape[4] * 2, // Double the number of sections
        };

        ADRT_OPENMP("omp parallel for collapse(5) default(none) shared(data, out, in_shape, curr_shape)")
        for(size_t batch = 0; batch < curr_shape[0]; ++batch) {
            for(size_t quadrant = 0; quadrant < 4; ++quadrant) {
                for(size_t row = 0; row < curr_shape[2]; ++row) {
                    for(size_t sec_i = 0; sec_i < curr_shape[3]; ++sec_i) {
                        // We double the sections we store to so loop over the previous (half count) value
                        for(size_t section = 0; section < in_shape[4]; ++section) {
                            const size_t sec_left = 2 * section, sec_right = 2 * section + 1;

                            // Left section
                            const adrt_scalar la_val = adrt::_common::array_access(data, in_shape, batch, quadrant, row, 2 * sec_i, section);
                            const adrt_scalar lb_val = adrt::_common::array_access(data, in_shape, batch, quadrant, row, 2 * sec_i + 1, section);
                            adrt::_common::array_access(out, curr_shape, batch, quadrant, row, sec_i, sec_left) = la_val + lb_val;

                            // Right section
                            adrt_scalar ra_val = 0, rb_val = 0;
                            if(row + sec_i < in_shape[2]) {
                                ra_val = adrt::_common::array_access(data, in_shape, batch, quadrant, row + sec_i, 2 * sec_i, section);
                            }
                            if(row + sec_i + 1 < in_shape[2]) {
                                rb_val = adrt::_common::array_access(data, in_shape, batch, quadrant, row + sec_i + 1, 2 * sec_i + 1, section);
                            }
                            adrt::_common::array_access(out, curr_shape, batch, quadrant, row, sec_i, sec_right) = ra_val + rb_val;
                        }
                    }
                }
            }
        }

        return curr_shape;
    }

    template <typename adrt_scalar>
    void bdrt_basic(const adrt_scalar *const data, const std::array<size_t, 4> &shape, adrt_scalar *const tmp, adrt_scalar *const out) {
        const int num_iters = adrt::num_iters(shape[3]);

        // Choose the ordering of the two buffers so that we always end with result in tmp (ready to copy out)
        adrt_scalar *buf_a = tmp;
        adrt_scalar *buf_b = out;
        if(num_iters % 2 != 0) {
            std::swap(buf_a, buf_b);
        }
        std::array<size_t, 5> buf_shape = adrt::bdrt_buffer_shape(shape);

        // Copy data to tmp buffer (always load into buf_a)
        for(size_t batch = 0; batch < shape[0]; ++batch) {
            for(size_t quadrant = 0; quadrant < 4; ++quadrant) {
                for(size_t row = 0; row < shape[2]; ++row) {
                    for(size_t col = 0; col < shape[3]; ++col) {
                        adrt::_common::array_access(buf_a, buf_shape, batch, quadrant, row, col, size_t{0}) =
                            adrt::_common::array_access(data, shape, batch, quadrant, row, col);
                    }
                }
            }
        }

        // Perform computations
        for(int i = 0; i < num_iters; ++i) {
            buf_shape = adrt::bdrt_core(buf_a, buf_shape, buf_b);
            std::swap(buf_a, buf_b);
        }

        // Copy result to out buffer (always tmp -> out)
        std::array<size_t, 4> output_shape = adrt::bdrt_result_shape(shape);
        for(size_t batch = 0; batch < output_shape[0]; ++batch) {
            for(size_t quadrant = 0; quadrant < 4; ++quadrant) {
                for(size_t row = 0; row < output_shape[2]; ++row) {
                    for(size_t col = 0; col < output_shape[3]; ++col) {
                        adrt::_common::array_access(out, output_shape, batch, quadrant, row, col) =
                            adrt::_common::array_access(tmp, buf_shape, batch, quadrant, row, size_t{0}, col);
                    }
                }
            }
        }
    }

}

#endif // ADRT_CDEFS_BDRT_H
