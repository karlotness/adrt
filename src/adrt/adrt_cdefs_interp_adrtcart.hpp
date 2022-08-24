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

#ifndef ADRT_CDEFS_INTERP_ADRTCART_H
#define ADRT_CDEFS_INTERP_ADRTCART_H

#include <cmath>
#include <array>
#include <utility>
#include <type_traits>
#include <cassert>
#include "adrt_cdefs_common.hpp"

namespace adrt {

    // Defined in: adrt_cdefs_common.cpp
    bool interp_adrtcart_is_valid_shape(const std::array<size_t, 4> &shape);
    std::array<size_t, 3> interp_adrtcart_result_shape(const std::array<size_t, 4> &shape);

    template <typename float_index = double>
    bool interp_adrtcart_is_valid_float_index(const std::array<size_t, 4> &in_shape) {
        return std::get<3>(in_shape) <= adrt::_common::floor_div(adrt::_const::largest_consecutive_float_size_t<float_index>(), 4_uz);
    }

    template <typename adrt_scalar, typename float_index = double>
    void interp_adrtcart(const adrt_scalar *const ADRT_RESTRICT data, const std::array<size_t, 4> &in_shape, adrt_scalar *const ADRT_RESTRICT out) {
        // The current implementation performs floating point arithmetic
        static_assert(std::is_floating_point<adrt_scalar>::value, "Cartesian interpolation requires floating point");
        static_assert(std::is_floating_point<float_index>::value, "Floating point index type must be a floating point type");

        assert(data);
        assert(out);
        assert(adrt::interp_adrtcart_is_valid_shape(in_shape));
        assert(adrt::interp_adrtcart_is_valid_float_index<float_index>(in_shape));

        const std::array<size_t, 3> output_shape = adrt::interp_adrtcart_result_shape(in_shape);

        const size_t N = std::get<3>(in_shape);

        const adrt_scalar Nf = static_cast<adrt_scalar>(N);
        const adrt_scalar Nf4 = static_cast<adrt_scalar>(4.0) * Nf;
        const adrt_scalar pi = adrt::_const::pi<adrt_scalar>();
        const adrt_scalar pi_2 = adrt::_const::pi_2<adrt_scalar>();
        const adrt_scalar pi_4 = adrt::_const::pi_4<adrt_scalar>();
        const adrt_scalar sqrt2 = adrt::_const::sqrt2<adrt_scalar>();
        const adrt_scalar sqrt2_2 = adrt::_const::sqrt2_2<adrt_scalar>();

        const adrt_scalar half = static_cast<adrt_scalar>(0.5);
        const adrt_scalar one = static_cast<adrt_scalar>(1.0);
        const adrt_scalar two = static_cast<adrt_scalar>(2.0);

        const adrt_scalar dth = pi / Nf4;
        const adrt_scalar th_left = -pi_2 + half*dth;

        const adrt_scalar ds = sqrt2 / Nf;
        const adrt_scalar s_left = -sqrt2_2 + half*ds;

        ADRT_OPENMP("omp parallel for collapse(3)")
        for(size_t batch = 0; batch < std::get<0>(in_shape); ++batch) {
            for(size_t offset = 0; offset < N; ++offset) {
                for(size_t angle = 0; angle < 4_uz*N; ++angle) {
                    const adrt_scalar j = static_cast<adrt_scalar>(N - 1_uz - offset);
                    const adrt_scalar s = s_left +  j*ds;

                    const adrt_scalar i = static_cast<adrt_scalar>(4_uz*N - 1_uz - angle);
                    const adrt_scalar th = th_left +  i*dth;

                    const adrt_scalar sgn = two*static_cast<adrt_scalar>(th > 0.0) - two*static_cast<adrt_scalar>(th > pi_4) - two*static_cast<adrt_scalar>(th > -pi_4) + one;

                    const adrt_scalar th0 = std::abs(th) - std::abs(th - pi_4) - std::abs(th + pi_4) + pi_2;
                    const adrt_scalar s_sgnd = sgn*s;

                    const size_t q = static_cast<size_t>(th > 0.0) + static_cast<size_t>(th >  -pi_4) + static_cast<size_t>(th > pi_4);
                    const adrt_scalar tif = static_cast<adrt_scalar>(std::floor(static_cast<adrt_scalar>(std::tan(th0))*(Nf - one)));

                    const adrt_scalar sidea = tif/Nf;
                    const adrt_scalar sideb = one - one/Nf;
                    const adrt_scalar factor = static_cast<adrt_scalar>(std::sqrt( sidea*sidea + sideb*sideb ));

                    const adrt_scalar h0 = half*(one + static_cast<adrt_scalar>(std::tan(th0))) - s_sgnd / (static_cast<adrt_scalar>(std::cos(th0)));

                    const adrt_scalar hif = static_cast<adrt_scalar>(std::floor(h0*Nf - half*(sgn + one)));

                    const size_t ti = static_cast<size_t>(tif);
                    const size_t hi = static_cast<size_t>(hif);

                    if ((hi >= 0_uz) && (hi < 2_uz*N - 1_uz) && (ti >= 0_uz) && (ti < N)){
                        adrt::_common::array_access(out, output_shape, batch, offset, angle) = factor*(adrt::_common::array_access(data, in_shape, batch, q, hi, ti)/Nf);
                    }
                    else {
                        adrt::_common::array_access(out, output_shape, batch, offset, angle) = 0.0;
                    };
                }
            }
        }
    }
}

#endif // ADRT_CDEFS_INTERP_ADRTCART_H
