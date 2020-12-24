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

#include "adrt_cdefs_common.hpp" // Include this first
#include "adrt_cdefs_adrt.hpp"
#include "adrt_cdefs_iadrt.hpp"
#include "adrt_cdefs_bdrt.hpp"

static bool adrt_validate_array(PyObject *args, PyArrayObject*& array_out,
                  int& iter_start_out, int& iter_end_out) {
    PyArrayObject *I;
    int iter_start = 0, iter_end = -1;

    if(!PyArg_ParseTuple(args, "O!|ii", &PyArray_Type, &I, &iter_start, &iter_end)) {
        return false;
    }

    if(!PyArray_CHKFLAGS(I, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must be C-order, contiguous, and aligned");
        return false;
    }

    if(PyArray_ISBYTESWAPPED(I)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must have native byte order");
        return false;
    }

    //
    int ndim = PyArray_NDIM(I);
    npy_intp * I_shape = PyArray_SHAPE(I);
    int num_iters = 1 + (int) adrt_num_iters(I_shape[ndim-1]);

    if(iter_start < -num_iters || iter_start >= num_iters || iter_end < -num_iters || iter_end >= num_iters) {
        PyErr_SetString(PyExc_ValueError,"Provided start and end iteration numbers are out of bounds");
        return false;
    }

    array_out = I;
    iter_start_out = iter_start;
    iter_end_out = iter_end;
    return true;
}

static bool adrt_is_valid_adrt_shape(const int ndim, const npy_intp *shape) {
    if(ndim < 3 || ndim > 4 || shape[ndim-2] != (shape[ndim-1] * 2 - 1)) {
        return false;
    }
    for(int i = 0; i < ndim; ++i) {
        if(shape[i] <= 0) {
            return false;
        }
    }
    npy_intp val = 1;
    while(val < shape[ndim - 1] && val > 0) {
        val *= 2;
    }
    if(val != shape[ndim - 1]) {
        return false;
    }
    return true;
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

static PyObject *adrt(PyObject* /* self */, PyObject *args){
    // Process function arguments
    PyObject *ret = nullptr;
    PyArrayObject * I;
    int iter_start = 0, iter_end = -1;
    npy_intp *old_shape = nullptr;
    npy_intp new_shape[4] = {0};

    int ndim = 2;
    int new_dim;

    if(!adrt_validate_array(args, I, iter_start, iter_end)) {
        goto fail;
    }

    if(!I) {
        goto fail;
    }
    ndim = PyArray_NDIM(I);
    old_shape = PyArray_SHAPE(I);
    if(iter_start == 0 && !adrt_is_square_power_of_two(ndim, PyArray_SHAPE(I))) {
        PyErr_SetString(PyExc_ValueError, "Provided array be square with power of two shapes");
        goto fail;
    }
    else if (iter_start != 0 && !adrt_is_valid_adrt_shape(ndim, PyArray_SHAPE(I))) {
        PyErr_SetString(PyExc_ValueError, "Provided array must have a valid ADRT shape");
        goto fail;
    }

    if (iter_start == 0){
        // Compute output shape: [plane?, 4, 2N, N] (batch, quadrant, row, col)
        if(ndim == 2) {
            new_shape[0] = 4;
            new_shape[1] = 2 * old_shape[1] - 1; // TODO use old_shape[0]
            new_shape[2] = old_shape[1];
            new_dim = 3;
        }
        else {
            new_shape[0] = old_shape[0];
            new_shape[1] = 4;
            new_shape[2] = 2 * old_shape[2] - 1; // TODO use old_shape[1]
            new_shape[3] = old_shape[2];
            new_dim = 4;
        }
    }
    else {
        // Compute output shape: [plane?, 4, 2N, N] (batch, quadrant, row, col)
        if(ndim == 3) {
            new_shape[0] = 4;
            new_shape[1] = old_shape[1];
            new_shape[2] = old_shape[2];
            new_dim = 3;
        }
        else {
            new_shape[0] = old_shape[0];
            new_shape[1] = 4;
            new_shape[2] = old_shape[1];
            new_shape[3] = old_shape[2];
            new_dim = 4;
        }
    }

    // Process input array
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
        ret = PyArray_SimpleNewFromDescr(new_dim, new_shape, PyArray_DescrFromType(NPY_FLOAT32));
        if(!ret ||
           !adrt_impl(static_cast<npy_float32*>(PyArray_DATA(I)),
                      ndim,
                      PyArray_SHAPE(I),
                      iter_start, iter_end, 
                      static_cast<npy_float32*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret))),
                      new_shape)) {
            goto fail;
        }
        break;
    case NPY_FLOAT64:
        ret = PyArray_SimpleNewFromDescr(new_dim, new_shape, PyArray_DescrFromType(NPY_FLOAT64));
        if(!ret ||
           !adrt_impl(static_cast<npy_float64*>(PyArray_DATA(I)),
                      ndim,
                      PyArray_SHAPE(I),
                      iter_start, iter_end, 
                      static_cast<npy_float64*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret))),
                      new_shape)) {
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

static PyObject *iadrt(PyObject* /* self */, PyObject *args){
    // Process function arguments
    PyObject *ret = nullptr;
    PyArrayObject * I;

    npy_intp *old_shape = nullptr;
    npy_intp new_shape[4] = {0};
    int ndim = 3;

    int iter_start = 0, iter_end = -1;
    if(!adrt_validate_array(args, I, iter_start, iter_end)) {
        goto fail;
    }

    if(!I) {
        goto fail;
    }
    ndim = PyArray_NDIM(I);
    old_shape = PyArray_SHAPE(I);
    if(!adrt_is_valid_adrt_shape(ndim, PyArray_SHAPE(I))) {
        PyErr_SetString(PyExc_ValueError, "Provided array must have a valid ADRT shape");
        goto fail;
    }


    // Compute output shape: [plane?, N, N] (batch, row, col)
    if(ndim == 3) {
        // Output has size (1,2*N-1, N)
        new_shape[0] = old_shape[0];
        new_shape[1] = old_shape[1];
        new_shape[2] = old_shape[2];
    }
    else {
        // Output has size (batch, 4, N, N)
        new_shape[0] = old_shape[0];
        new_shape[1] = old_shape[1];
        new_shape[2] = old_shape[2];
        new_shape[3] = old_shape[3];
    }

    // Process input array
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
        ret = PyArray_SimpleNewFromDescr(ndim, new_shape, PyArray_DescrFromType(NPY_FLOAT32));
        if(!ret ||
           !iadrt_impl(static_cast<npy_float32*>(PyArray_DATA(I)),
                      ndim,
                      PyArray_SHAPE(I),
                      iter_start, iter_end, 
                      static_cast<npy_float32*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret))),
                      new_shape)) {
            goto fail;
        }
        break;
    case NPY_FLOAT64:
        ret = PyArray_SimpleNewFromDescr(ndim, new_shape, PyArray_DescrFromType(NPY_FLOAT64));
        if(!ret ||
           !iadrt_impl(static_cast<npy_float64*>(PyArray_DATA(I)),
                      ndim,
                      PyArray_SHAPE(I),
                      iter_start, iter_end, 
                      static_cast<npy_float64*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret))),
                      new_shape)) {
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

static PyObject *bdrt(PyObject* /* self */, PyObject *args){
    // Process function arguments
    PyObject *ret = nullptr;
    npy_intp *old_shape = nullptr;
    npy_intp new_shape[3] = {0};
    PyArrayObject * I;
    int ndim = 3;

    int iter_start = 0, iter_end = -1;
    if(!adrt_validate_array(args, I, iter_start, iter_end)) {
        goto fail;
    }

    if(!I) {
        goto fail;
    }
    ndim = PyArray_NDIM(I);
    old_shape = PyArray_SHAPE(I);
    if(!adrt_is_valid_adrt_shape(ndim, PyArray_SHAPE(I))) {
        PyErr_SetString(PyExc_ValueError, "Provided array must have a valid ADRT shape");
        goto fail;
    }

    // Compute output shape: [plane?, N, N] (batch, row, col)
    if(ndim == 3) {
        new_shape[0] = 1;
        new_shape[1] = old_shape[1];
        new_shape[2] = old_shape[2];
    }
    else {
        new_shape[0] = old_shape[0];
        new_shape[1] = old_shape[2];
        new_shape[2] = old_shape[3];
    }

    // Process input array
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
        ret = PyArray_SimpleNewFromDescr(ndim, new_shape, PyArray_DescrFromType(NPY_FLOAT32));
        if(!ret ||
           !bdrt_impl(static_cast<npy_float32*>(PyArray_DATA(I)),
                      ndim,
                      PyArray_SHAPE(I),
                      static_cast<npy_float32*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret))), new_shape)) {
            goto fail;
        }
        break;
    case NPY_FLOAT64:
        ret = PyArray_SimpleNewFromDescr(ndim, new_shape, PyArray_DescrFromType(NPY_FLOAT64));
        if(!ret ||
           !bdrt_impl(static_cast<npy_float64*>(PyArray_DATA(I)),
                      ndim,
                      PyArray_SHAPE(I),
                      static_cast<npy_float64*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret))), new_shape)) {
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

static PyMethodDef adrt_cdefs_methods[] = {
    {"adrt", adrt, METH_VARARGS, "Compute the ADRT"},
    {"iadrt", iadrt, METH_VARARGS, "Compute the inverse ADRT"},
    {"bdrt", bdrt, METH_VARARGS, "Compute the backprojection of the ADRT"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef adrt_cdefs_module = {
    PyModuleDef_HEAD_INIT,
    "adrt._adrt_cdefs",
    "C routines for ADRT. These should not be called directly by module users.",
    0,
    adrt_cdefs_methods,
    nullptr,
    // GC hooks below, unused
    nullptr, nullptr, nullptr
};

PyMODINIT_FUNC
PyInit__adrt_cdefs(void)
{
    PyObject *module = PyModule_Create(&adrt_cdefs_module);
    if(!module) {
        return nullptr;
    }
    import_array();
    return module;
}

} // extern "C"
