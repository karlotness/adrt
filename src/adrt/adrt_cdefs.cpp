/*
 * Copyright (c) 2020, 2021 Karl Otness, Donsub Rim
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

#include "adrt_cdefs_py.hpp" // Include this first
#include "adrt_cdefs_common.hpp"
#include "adrt_cdefs_adrt.hpp"
#include "adrt_cdefs_iadrt.hpp"
#include "adrt_cdefs_bdrt.hpp"
#include "adrt_cdefs_interp_adrtcart.hpp"
#include <array>

static PyArrayObject *adrt_extract_array(PyObject *arg) {
    if(!PyArray_Check(arg)) {
        // This isn't an array
        PyErr_SetString(PyExc_TypeError, "Argument must be a NumPy array or compatible subclass");
        return nullptr;
    }
    PyArrayObject *arr = reinterpret_cast<PyArrayObject*>(arg);
    if(!PyArray_ISCARRAY_RO(arr)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must be C-order, contiguous, aligned, and native byte order");
        return nullptr;
    }
    return arr;
}

template <size_t min_dim, size_t max_dim>
static bool adrt_shape_to_array(PyArrayObject *arr, std::array<size_t, max_dim> &shape_arr) {
    static_assert(min_dim <= max_dim, "Min dimensions must be less than max dimensions.");
    int sndim = PyArray_NDIM(arr);
    size_t ndim = sndim;
    if(sndim < 0 || ndim < min_dim || ndim > max_dim) {
        PyErr_SetString(PyExc_ValueError, "Invalid number of dimensions for input array");
        return false;
    }
    npy_intp *numpy_shape = PyArray_SHAPE(arr);
    // Prepend trivial dimensions
    for(size_t i = 0; i < max_dim - ndim; ++i) {
        shape_arr[i] = 1;
    }
    // Fill rest of array
    for(size_t i = 0; i < ndim; ++i) {
        npy_intp shape = numpy_shape[i];
        if(shape <= 0) {
            PyErr_SetString(PyExc_ValueError, "Array must not have shape with dimension of zero");
            return false;
        }
        shape_arr[i + (max_dim - ndim)] = shape;
    }
    return true;
}

template <size_t n_virtual_dim>
static PyArrayObject *adrt_new_array(int ndim, const std::array<size_t, n_virtual_dim> &virtual_shape, int typenum) {
    if(ndim > static_cast<int>(n_virtual_dim)) {
        PyErr_SetString(PyExc_ValueError, "Invalid number of dimensions computed for output array");
        return nullptr;
    }
    npy_intp new_shape[n_virtual_dim] = {0};
    for(int i = 0; i < ndim; ++i) {
        new_shape[i] = virtual_shape[(n_virtual_dim - ndim) + i];
    }
    PyObject *arr = PyArray_SimpleNew(ndim, new_shape, typenum);
    return reinterpret_cast<PyArrayObject*>(arr);
}

static bool adrt_validate_array(PyObject *args, PyArrayObject*& array_out) {
    // validate_array without iteration bounds
    PyArrayObject *I;

    if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &I)) {
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

    array_out = I;
    return true;
}

static bool adrt_validate_array_iters(PyObject *args, PyArrayObject*& array_out,
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

extern "C" {

static PyObject *adrt_py(PyObject* /* self */, PyObject *arg) {
    // Process function arguments
    PyArrayObject *I = adrt_extract_array(arg);
    if(!I) {
        return nullptr;
    }
    // Extract shapes and check sizes
    std::array<size_t, 3> input_shape;
    if(!adrt_shape_to_array<2, 3>(I, input_shape)) {
        return nullptr;
    }
    if(!adrt::adrt_is_valid_shape(input_shape)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must be square with a power of two shape");
        return nullptr;
    }
    // Compute effective output shape
    const std::array<size_t, 4> output_shape = adrt::adrt_result_shape(input_shape);
    const size_t tmp_buf_elems = adrt::_common::array_product(adrt::adrt_buffer_shape(input_shape));
    // Process input array
    int ndim = PyArray_NDIM(I);
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
    {
        PyArrayObject *ret = adrt_new_array(ndim + 1, output_shape, NPY_FLOAT32);
        npy_float32 *tmp_buf = PyMem_New(npy_float32, tmp_buf_elems);
        if(!ret || !tmp_buf) {
            PyMem_Free(tmp_buf);
            Py_XDECREF(ret);
            return nullptr;
        }
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS;
        adrt::adrt_basic(static_cast<npy_float32*>(PyArray_DATA(I)), input_shape, tmp_buf, static_cast<npy_float32*>(PyArray_DATA(ret)));
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS;
        PyMem_Free(tmp_buf);
        return reinterpret_cast<PyObject*>(ret);
    }
    case NPY_FLOAT64:
    {
        PyArrayObject *ret = adrt_new_array(ndim + 1, output_shape, NPY_FLOAT64);
        npy_float64 *tmp_buf = PyMem_New(npy_float64, tmp_buf_elems);
        if(!ret || !tmp_buf) {
            PyMem_Free(tmp_buf);
            Py_XDECREF(ret);
            return nullptr;
        }
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS;
        adrt::adrt_basic(static_cast<npy_float64*>(PyArray_DATA(I)), input_shape, tmp_buf, static_cast<npy_float64*>(PyArray_DATA(ret)));
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS;
        PyMem_Free(tmp_buf);
        return reinterpret_cast<PyObject*>(ret);
    }
    default:
        PyErr_SetString(PyExc_TypeError, "Unsupported array type");
        return nullptr;
    }
}

static PyObject *iadrt(PyObject* /* self */, PyObject *args){
    // Process function arguments
    PyObject *ret = nullptr;
    PyArrayObject * I;

    npy_intp *old_shape = nullptr;
    npy_intp new_shape[4] = {0};
    int ndim = 3;

    int iter_start = 0, iter_end = -1;
    if(!adrt_validate_array_iters(args, I, iter_start, iter_end)) {
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
        ret = PyArray_SimpleNew(ndim, new_shape, NPY_FLOAT32);
        if(!ret ||
           !iadrt_impl(static_cast<npy_float32*>(PyArray_DATA(I)),
                      static_cast<unsigned char>(ndim),
                      PyArray_SHAPE(I),
                      iter_start, iter_end,
                      static_cast<npy_float32*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret))),
                      new_shape)) {
            goto fail;
        }
        break;
    case NPY_FLOAT64:
        ret = PyArray_SimpleNew(ndim, new_shape, NPY_FLOAT64);
        if(!ret ||
           !iadrt_impl(static_cast<npy_float64*>(PyArray_DATA(I)),
                      static_cast<unsigned char>(ndim),
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
    npy_intp new_shape[4] = {0};
    PyArrayObject * I;
    int ndim = 3;

    int iter_start = 0, iter_end = -1;
    if(!adrt_validate_array_iters(args, I, iter_start, iter_end)) {
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

    // Compute output shape: [plane?, 2*N-1, N] (batch, row, col)
    if(ndim == 3) {
        new_shape[0] = old_shape[0];
        new_shape[1] = old_shape[1];
        new_shape[2] = old_shape[2];
    }
    else {
        new_shape[0] = old_shape[0];
        new_shape[1] = old_shape[1];
        new_shape[2] = old_shape[2];
        new_shape[3] = old_shape[3];
    }

    // Process input array
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
        ret = PyArray_SimpleNew(ndim, new_shape, NPY_FLOAT32);
        if(!ret ||
           !bdrt_impl(static_cast<npy_float32*>(PyArray_DATA(I)),
                      static_cast<unsigned char>(ndim),
                      PyArray_SHAPE(I),
                      iter_start, iter_end,
                      static_cast<npy_float32*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret))), new_shape)) {
            goto fail;
        }
        break;
    case NPY_FLOAT64:
        ret = PyArray_SimpleNew(ndim, new_shape, NPY_FLOAT64);
        if(!ret ||
           !bdrt_impl(static_cast<npy_float64*>(PyArray_DATA(I)),
                      static_cast<unsigned char>(ndim),
                      PyArray_SHAPE(I),
                      iter_start, iter_end,
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

static PyObject *interp_adrtcart(PyObject* /* self */, PyObject *args){
    // interpolate adrt data to Cartesian coordinates 

    // Process function arguments
    PyObject *ret = nullptr;
    npy_intp *old_shape = nullptr;
    npy_intp new_shape[4] = {0};
    PyArrayObject * I;
    int ndim = 3;

    if(!adrt_validate_array(args, I)) {
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

    // Compute output shape: [plane?, 4, N, N] (batch, row, col)
    if(ndim == 3) {
        new_shape[0] = old_shape[0];    // 4
        new_shape[1] = old_shape[2];    // N
        new_shape[2] = old_shape[2];    // N
    }
    else {
        new_shape[0] = old_shape[0];    // plane?
        new_shape[1] = old_shape[1];    // 4
        new_shape[2] = old_shape[3];    // N
        new_shape[3] = old_shape[3];    // N
    }

    // Process input array
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
        ret = PyArray_SimpleNew(ndim, new_shape, NPY_FLOAT32);
        if(!ret ||
           !interp_adrtcart_impl(static_cast<npy_float32*>(PyArray_DATA(I)),
                      static_cast<unsigned char>(ndim),
                      PyArray_SHAPE(I),
                      static_cast<npy_float32*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret))), new_shape)) {
            goto fail;
        }
        break;
    case NPY_FLOAT64:
        ret = PyArray_SimpleNew(ndim, new_shape, NPY_FLOAT64);
        if(!ret ||
           !interp_adrtcart_impl(static_cast<npy_float64*>(PyArray_DATA(I)),
                      static_cast<unsigned char>(ndim),
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

static PyObject *num_iters(PyObject* /* self */, PyObject *arg){
    size_t val = PyLong_AsSize_t(arg);
    if(PyErr_Occurred()) {
        return nullptr;
    }
    return PyLong_FromLong(adrt_num_iters(val));
}

static PyMethodDef adrt_cdefs_methods[] = {
    {"adrt", adrt_py, METH_O, "Compute the ADRT"},
    {"iadrt", iadrt, METH_VARARGS, "Compute the inverse ADRT"},
    {"bdrt", bdrt, METH_VARARGS, "Compute the backprojection of the ADRT"},
    {"num_iters", num_iters, METH_O, "Compute the number of iterations needed for the ADRT"},
    {"interp_adrtcart", interp_adrtcart, METH_VARARGS, 
     "Interpolate ADRT output to Cartesian coordinate system"},
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

// Support Python<3.9 which doesn't set visibility
#pragma GCC visibility push(default)

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

#pragma GCC visibility pop

} // extern "C"
