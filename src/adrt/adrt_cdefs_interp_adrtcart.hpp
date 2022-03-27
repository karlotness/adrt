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

    template <typename adrt_scalar>
    void interp_adrtcart(const adrt_scalar *const ADRT_RESTRICT data, const std::array<size_t, 4> &shape, adrt_scalar *const ADRT_RESTRICT out) {
        // The current implementation performs floating point arithmetic
        static_assert(std::is_floating_point<adrt_scalar>::value, "Cartesian interpolation requires floating point");

        assert(data);
        assert(out);
        assert(adrt::interp_adrtcart_is_valid_shape(shape));

        const std::array<size_t, 3> output_shape = adrt::interp_adrtcart_result_shape(shape);

    }

}


#endif // ADRT_CDEFS_INTERP_ADRTCART_H
