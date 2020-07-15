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
