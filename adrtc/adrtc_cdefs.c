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

#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x03040000
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include "numpy/arrayobject.h"

#include <stdbool.h>

static bool Sadrt(const npy_float32 *const data, npy_intp *shape, npy_float32 *out) {
    // Allocate auxiliary memory
    npy_float32 *aux = malloc(sizeof(npy_float32) * 2 * shape[0] * shape[1]);
    if(!aux) {
        PyErr_NoMemory();
        return false;
    }

    // NO PYTHON API BELOW THIS POINT
    NPY_BEGIN_ALLOW_THREADS;

    NPY_END_ALLOW_THREADS;

    return true;
}

static PyObject *adrt(__attribute__((unused)) PyObject *self, PyObject *args){
    PyArrayObject *I; // Input array

    if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &I)) {
        return NULL;
    }
    if(!PyArray_CHKFLAGS(I, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must be C-order, contiguous, and aligned");
        return NULL;
    }
    if(PyArray_ISBYTESWAPPED(I)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must have native byte order");
        return NULL;
    }
    if(PyArray_NDIM(I) != 2) {
        PyErr_Format(PyExc_ValueError, "Invalid dimensionality %d, array must have two dimensions", PyArray_NDIM(I));
        return NULL;
    }

    PyObject *ret = NULL;
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
        ret = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_FLOAT32),
                                   PyArray_NDIM(I), PyArray_SHAPE(I), NULL,
                                   NULL,  NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_OWNDATA | NPY_ARRAY_WRITEABLE,
                                   NULL);
        if(!ret || !Sadrt((npy_float*)PyArray_DATA(I), PyArray_SHAPE(I), PyArray_DATA((PyArrayObject *) ret))) {
            Py_XDECREF(ret);
            return NULL;
        }
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "Unsupported array type");
        return NULL;
    }

    return ret;
}

static PyMethodDef adrtc_cdefs_methods[] = {
    {"adrt", adrt, METH_VARARGS, "Compute the ADRT"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef adrtc_cdefs_module = {
    PyModuleDef_HEAD_INIT,
    "_adrtc_cdefs",
    "C routines for ADRTC. These should not be called directly by module users.",
    0,
    adrtc_cdefs_methods,
    NULL,
    // GC hooks below, unused
    NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__adrtc_cdefs(void)
{
    PyObject *module = PyModule_Create(&adrtc_cdefs_module);
    if(!module) {
        return NULL;
    }
    import_array();
    return module;
}
