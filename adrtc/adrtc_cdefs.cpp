/*
 * Copyright (C) 2020 by the ADRT Development Team
 * All right reserved
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

#include "adrtc_cdefs_common.h" // Include this first
#include "adrtc_cdefs_adrt.h"

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

extern "C" {

static PyObject *adrt(__attribute__((unused)) PyObject *self, PyObject *args){
    // Process function arguments
    PyObject *ret = nullptr;
    PyArrayObject *I = adrt_validate_array(args); // Input array
    if(!I) {
        goto fail;
    }
    ret = PyArray_NewLikeArray(I, NPY_CORDER, nullptr, 0);
    if(!ret) {
        // Allocation failed
        goto fail;
    }
    // Process input array
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
        if(!_adrt(static_cast<npy_float32*>(PyArray_DATA(I)),
                  PyArray_SHAPE(I),
                  static_cast<npy_float32*>(PyArray_DATA((PyArrayObject *) ret)))) {
            goto fail;
        }
        break;
    case NPY_FLOAT64:
        if(!_adrt(static_cast<npy_float64*>(PyArray_DATA(I)),
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
    "_adrtc_cdefs",
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
