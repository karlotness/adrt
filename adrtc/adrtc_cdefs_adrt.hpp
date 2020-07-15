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
#ifndef ADRTC_CDEFS_ADRT_H
#define ADRTC_CDEFS_ADRT_H

#include "adrtc_cdefs_common.hpp"

template <typename adrt_scalar, typename adrt_shape>
static bool _adrt(const adrt_scalar *const data, unsigned char ndims, const adrt_shape *const shape, adrt_scalar *out) {
    // Shape (plane, row, col)
    const adrt_shape corrected_shape[3] =
        {(ndims > 2 ? shape[0] : 1),
         (ndims > 2 ? shape[1] : shape[0]),
         (ndims > 2 ? shape[2] : shape[1])};

    // Allocate auxiliary memory
    adrt_scalar *aux = PyMem_New(adrt_scalar, 2 * shape[0] * shape[1]);
    if(!aux) {
        PyErr_NoMemory();
        return false;
    }

    // NO PYTHON API BELOW THIS POINT
    Py_BEGIN_ALLOW_THREADS;

    // PYTHON API ALLOWED BELOW THIS POINT
    Py_END_ALLOW_THREADS;

    PyMem_Free(aux);
    return true;
}

#endif // ADRTC_CDEFS_ADRT_H
