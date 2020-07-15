#pragma once
#ifndef ADRTC_CDEFS_ADRT_H
#define ADRTC_CDEFS_ADRT_H

#include "adrtc_cdefs_common.h"

template <typename adrt_scalar, typename adrt_shape>
static bool _adrt(const adrt_scalar *const data, adrt_shape *shape, adrt_scalar *out) {
    // Allocate auxiliary memory
    adrt_scalar *aux = PyMem_New(adrt_scalar, 2 * shape[0] * shape[1]);
    if(!aux) {
        PyErr_NoMemory();
        return false;
    }

    // NO PYTHON API BELOW THIS POINT
    Py_BEGIN_ALLOW_THREADS;

    for (adrt_shape i = 0; i < shape[0]; i++) {
        for (adrt_shape j = 0; j < shape[1]; j++) {
            adrt_array_2d_access(out, shape, i, j) = 10 * adrt_array_2d_access(data, shape, i, j);
        }
    }

    // PYTHON API ALLOWED BELOW THIS POINT
    Py_END_ALLOW_THREADS;

    PyMem_Free(aux);
    return true;
}

#endif // ADRTC_CDEFS_ADRT_H
