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

#include "adrtc_cdefs_common.hpp" // Include this first
#include "adrtc_cdefs_adrt.hpp"

static PyArrayObject *adrt_validate_array(PyObject *args) {
    PyArrayObject *I;
    if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &I)) {
        return nullptr;
    }
    if(!PyArray_CHKFLAGS(I, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must be C-order, contiguous, and aligned");
        return nullptr;
    }
    if(PyArray_ISBYTESWAPPED(I)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must have native byte order");
        return nullptr;
    }
    int ndim = PyArray_NDIM(I);
    if(ndim != 2 && ndim != 3) {
        PyErr_Format(PyExc_ValueError, "Invalid dimensionality %d, array must have two or three dimensions", ndim);
        return nullptr;
    }
    return I;
}

static bool adrt_is_square_power_of_two(const int ndim, const npy_intp *shape) {
    if(ndim < 2 || ndim > 3 || shape[ndim - 1] != shape[ndim - 2]) {
        return false;
    }
    for(int i = ndim - 2; i < ndim; ++i) {
        if(shape[i] <= 0) {
            return false;
        }
        npy_intp val = 1;
        while(val < shape[i] && val > 0) {
            val *= 2;
        }
        if(val != shape[i]) {
            return false;
        }
    }
    return true;
}

extern "C" {

static PyObject *adrt(__attribute__((unused)) PyObject *self, PyObject *args){
    // Process function arguments
    PyObject *ret = nullptr;
    PyArrayObject *I = adrt_validate_array(args); // Input array
    npy_intp *old_shape = nullptr;
    npy_intp new_shape[3] = {0};
    int ndim = 2;
    if(!I) {
        goto fail;
    }
    ndim = PyArray_NDIM(I);
    old_shape = PyArray_SHAPE(I);
    if(!adrt_is_square_power_of_two(ndim, PyArray_SHAPE(I))) {
        PyErr_SetString(PyExc_ValueError, "Provided array be square with power of two shapes");
        goto fail;
    }
    // Compute new array shape
    for(int i = 0; i < ndim; ++i) {
        new_shape[i] = old_shape[i];
    }
    new_shape[ndim - 1] = 4 * (new_shape[ndim - 1] - 1);
    // Process input array
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
        ret = PyArray_SimpleNewFromDescr(ndim, new_shape, PyArray_DescrFromType(NPY_FLOAT32));
        if(!ret ||
           !_adrt(static_cast<npy_float32*>(PyArray_DATA(I)),
                  ndim,
                  PyArray_SHAPE(I),
                  static_cast<npy_float32*>(PyArray_DATA((PyArrayObject *) ret)))) {
            goto fail;
        }
        break;
    case NPY_FLOAT64:
        ret = PyArray_SimpleNewFromDescr(ndim, new_shape, PyArray_DescrFromType(NPY_FLOAT64));
        if(!ret ||
           !_adrt(static_cast<npy_float64*>(PyArray_DATA(I)),
                  ndim,
                  PyArray_SHAPE(I),
                  static_cast<npy_float64*>(PyArray_DATA((PyArrayObject *) ret)))) {
            goto fail;
        }
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "Unsupported array type");
        goto fail;
    }
    return ret;
  fail:
    Py_XDECREF(ret);
    return nullptr;
}

static PyMethodDef adrtc_cdefs_methods[] = {
    {"adrt", adrt, METH_VARARGS, "Compute the ADRT"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef adrtc_cdefs_module = {
    PyModuleDef_HEAD_INIT,
    "adrtc._adrtc_cdefs",
    "C routines for ADRTC. These should not be called directly by module users.",
    0,
    adrtc_cdefs_methods,
    nullptr,
    // GC hooks below, unused
    nullptr, nullptr, nullptr
};

PyMODINIT_FUNC
PyInit__adrtc_cdefs(void)
{
    PyObject *module = PyModule_Create(&adrtc_cdefs_module);
    if(!module) {
        return nullptr;
    }
    import_array();
    return module;
}

} // extern "C"
