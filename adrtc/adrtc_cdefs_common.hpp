#pragma once
#ifndef ADRTC_CDEFS_COMMON_H
#define ADRTC_CDEFS_COMMON_H

#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x03040000
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include "numpy/arrayobject.h"

template <typename adrt_scalar, typename adrt_shape>
inline adrt_scalar& adrt_array_3d_access(adrt_scalar *buf, const adrt_shape shape[3],
                                         adrt_shape plane, adrt_shape row, adrt_shape col) {
    return buf[(shape[1] * shape[2]) * plane + shape[2] * row + col];
}

#endif //ADRTC_CDEFS_COMMON_H
