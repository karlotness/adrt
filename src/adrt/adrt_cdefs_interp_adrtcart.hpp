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

#ifndef ADRT_CDEFS_INTERP_ADRTCART_H
#define ADRT_CDEFS_INTERP_ADRTCART_H

#include <cmath>
#include <array>
#include <span>
#include <type_traits>
#include <concepts>
#include <limits>
#include <algorithm>
#include <cassert>
#include "adrt_cdefs_common.hpp"

namespace adrt {

    // Defined in: adrt_cdefs_common.cpp
    bool interp_adrtcart_is_valid_shape(std::span<const size_t, 4> shape);
    std::array<size_t, 3> interp_adrtcart_result_shape(std::span<const size_t, 4> shape);

    template <std::floating_point float_index = double>
    bool interp_adrtcart_is_valid_float_index(std::span<const size_t, 4> in_shape) {
        // The input shape is (batch, 4, 2*n-1, n)
        // The output shape is (batch, n, 4*n)
        // We ensure that 4*n fits in a float, so n and 2*n-1 will fit as well
        return adrt::_common::get<3>(in_shape) <= adrt::_common::floor_div(adrt::_const::largest_consecutive_float_size_t<float_index>, 4_uz);
    }

    // DOC ANCHOR: adrt.utils.interp_to_cart +2
    template <std::floating_point adrt_scalar, std::floating_point float_index = double>
    void interp_adrtcart(const adrt_scalar *const ADRT_RESTRICT data, std::span<const size_t, 4> in_shape, adrt_scalar *const ADRT_RESTRICT out) {
        // The current implementation performs floating point arithmetic
        assert(data);
        assert(out);
        assert(adrt::interp_adrtcart_is_valid_shape(in_shape));
        assert(adrt::interp_adrtcart_is_valid_float_index<float_index>(in_shape));

        using larger_float = std::conditional_t<(std::numeric_limits<adrt_scalar>::digits > std::numeric_limits<float_index>::digits), adrt_scalar, float_index>;
        const std::array<size_t, 3> output_shape = adrt::interp_adrtcart_result_shape(in_shape);

        const size_t shape_n = adrt::_common::get<3>(in_shape);
        const float_index t_left = adrt::_const::sqrt2_2<float_index> - (adrt::_const::sqrt2_2<float_index> / static_cast<float_index>(shape_n));
        const float_index th_left = adrt::_const::pi_2<float_index> - (adrt::_const::pi_8<float_index> / static_cast<float_index>(shape_n));

        ADRT_OPENMP("omp parallel for collapse(3) default(none) shared(data, in_shape, out, output_shape, shape_n, t_left, th_left)")
        for(size_t batch = 0; batch < adrt::_common::get<0>(output_shape); ++batch) {
            for(size_t offset = 0; offset < adrt::_common::get<1>(output_shape); ++offset) {
                for(size_t angle = 0; angle < adrt::_common::get<2>(output_shape); ++angle) {
                    const float_index offset_fraction = static_cast<float_index>(offset) / static_cast<float_index>(adrt::_common::get<1>(output_shape) - 1_uz);
                    const float_index angle_fraction = static_cast<float_index>(angle) / static_cast<float_index>(adrt::_common::get<2>(output_shape) - 1_uz);
                    const float_index t = t_left * std::lerp(-static_cast<float_index>(1), static_cast<float_index>(1), offset_fraction);
                    const float_index th = th_left * std::lerp(static_cast<float_index>(1), -static_cast<float_index>(1), angle_fraction);
                    // Compute the quadrant and parity of the angle th
                    const int q = static_cast<int>(std::floor(std::clamp(-th / adrt::_const::pi_4<float_index>, -static_cast<float_index>(2), static_cast<float_index>(1))) + 2);
                    const int sgn = q % 2 == 0 ? 1 : -1;
                    // Compute angle and offset for indexing
                    // We know th is in [-pi/2, pi/2] so we can compute th0 with some arithmetic
                    const float_index th0 = adrt::_const::pi_4<float_index> - std::abs(std::abs(th) - adrt::_const::pi_4<float_index>);
                    const float_index tan_theta = std::clamp(std::tan(th0), static_cast<float_index>(0), static_cast<float_index>(1));
                    const float_index si = std::round(tan_theta * static_cast<float_index>(shape_n - 1_uz));
                    assert(std::isfinite(si));
                    assert(si >= static_cast<float_index>(0));
                    assert(si < static_cast<float_index>(adrt::_common::get<3>(in_shape)));
                    // Compute the scaling factor
                    const larger_float sidea = static_cast<larger_float>(si) / static_cast<larger_float>(shape_n - 1_uz);
                    const larger_float sideb = static_cast<larger_float>(1);
                    const adrt_scalar factor = static_cast<adrt_scalar>(std::sqrt(sidea * sidea + sideb));
                    const float_index h0 = (static_cast<float_index>(0.5L) + (tan_theta / static_cast<float_index>(2))) + ((sgn >= 0 ? t : -t) / std::cos(th0));
                    const float_index hi = (std::round(h0 * static_cast<float_index>(2_uz * shape_n)) - static_cast<float_index>(1)) / static_cast<float_index>(2);
                    assert(std::isfinite(hi));
                    // Perform the updates
                    if(hi >= static_cast<float_index>(0) && hi < static_cast<float_index>(adrt::_common::get<2>(in_shape))) {
                        // Intended access is in bounds
                        adrt::_common::array_access(out, output_shape, batch, offset, angle) = factor * adrt::_common::array_access(data, in_shape, batch, static_cast<size_t>(q), static_cast<size_t>(hi), static_cast<size_t>(si));
                    }
                    else {
                        // Access is out of bounds, fill with zero
                        adrt::_common::array_access(out, output_shape, batch, offset, angle) = 0;
                    }
                }
            }
        }
    }

}

#endif // ADRT_CDEFS_INTERP_ADRTCART_H
