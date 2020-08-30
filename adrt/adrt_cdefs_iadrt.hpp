/*
 * Copyright (C) 2020 Karl Otness, Donsub Rim
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
#ifndef ADRTC_CDEFS_IADRT_H
#define ADRTC_CDEFS_IADRT_H

#include "adrt_cdefs_common.hpp"

template <typename adrt_scalar, typename adrt_shape>
static bool iadrt_impl(const adrt_scalar *const data, const unsigned char ndims, const adrt_shape *const shape, adrt_scalar *const out,
                       const adrt_shape *const base_output_shape) {
    // Shape (plane, row, col)
    const adrt_shape corrected_shape[4] =
        {(ndims > 3 ? shape[0] : 1),
         (ndims > 3 ? shape[1] : shape[0]),
         (ndims > 3 ? shape[2] : shape[1]),
         (ndims > 3 ? shape[3] : shape[2])};

    const adrt_shape output_shape[3] =
        {(ndims > 3 ? base_output_shape[0] : 1),
         (ndims > 3 ? base_output_shape[1] : base_output_shape[0]),
         (ndims > 3 ? base_output_shape[2] : base_output_shape[1])};

    return true;
}

#endif // ADRTC_CDEFS_IADRT_H
