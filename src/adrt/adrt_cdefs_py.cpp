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

#include <array>
#include <limits>

#include "adrt_cdefs_common.hpp"
#include "adrt_cdefs_adrt.hpp"
#include "adrt_cdefs_iadrt.hpp"
#include "adrt_cdefs_bdrt.hpp"
#include "adrt_cdefs_interp_adrtcart.hpp"

namespace adrt { namespace _py { namespace {

PyArrayObject *extract_array(PyObject *arg) {
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
adrt::_common::Optional<std::array<size_t, max_dim>> array_shape(PyArrayObject *arr) {
    static_assert(min_dim <= max_dim, "Min dimensions must be less than max dimensions.");
    std::array<size_t, max_dim> shape_arr;
    const int sndim = PyArray_NDIM(arr);
    const unsigned int ndim = static_cast<unsigned int>(sndim);
    if(sndim < 0 || ndim < min_dim || ndim > max_dim) {
        PyErr_SetString(PyExc_ValueError, "Invalid number of dimensions for input array");
        return {};
    }
    const npy_intp *const numpy_shape = PyArray_SHAPE(arr);
    // Prepend trivial dimensions
    for(size_t i = 0; i < max_dim - ndim; ++i) {
        shape_arr[i] = 1;
    }
    // Fill rest of array
    for(size_t i = 0; i < ndim; ++i) {
        const npy_intp shape = numpy_shape[i];
        if(shape <= 0) {
            PyErr_SetString(PyExc_ValueError, "Array must not have shape with dimension of zero");
            return {};
        }
        shape_arr[i + (max_dim - ndim)] = static_cast<size_t>(shape);
    }
    return {shape_arr};
}

template <size_t n_virtual_dim>
PyArrayObject *new_array(int ndim, const std::array<size_t, n_virtual_dim> &virtual_shape, int typenum) {
    const unsigned int undim = static_cast<unsigned int>(ndim);
    if(undim > n_virtual_dim || ndim <= 0) {
        // This would be a bug and should have been caught earlier. Handle it as well as we can.
        PyErr_SetString(PyExc_RuntimeError, "Invalid number of dimensions computed for output array");
        return nullptr;
    }
    npy_intp new_shape[n_virtual_dim] = {0};
    for(size_t i = 0; i < undim; ++i) {
        const size_t shape_val = virtual_shape[(n_virtual_dim - undim) + i];
        if(shape_val <= static_cast<npy_uintp>(std::numeric_limits<npy_intp>::max())) {
            new_shape[i] = static_cast<npy_intp>(shape_val);
        }
        else {
            PyErr_SetString(PyExc_ValueError, "Maximum allowed dimension exceeded");
            return nullptr;
        }
    }
    PyObject *arr = PyArray_SimpleNew(ndim, new_shape, typenum);
    if(!arr) {
        if(!PyErr_Occurred()) {
            // Don't shadow errors that NumPy may have set
            PyErr_NoMemory();
        }
        return nullptr;
    }
    return reinterpret_cast<PyArrayObject*>(arr);
}

template <size_t ndim>
adrt::_common::Optional<size_t> shape_product(const std::array<size_t, ndim> &shape) {
    static_assert(ndim > 0, "Need at least one shape dimension");
    size_t n_elem = shape[0];
    for(size_t i = 1; i < ndim; ++i) {
        if(!adrt::_common::mul_check(n_elem, shape[i]).store_value(n_elem)) {
            PyErr_SetString(PyExc_ValueError, "Array is too big; unable to allocate temporary space");
            return {};
        }
    }
    return {n_elem};
}

void *py_malloc(size_t n_elem, size_t elem_size) {
    size_t alloc_size;
    if(!adrt::_common::mul_check(n_elem, elem_size).store_value(alloc_size)) {
        PyErr_SetString(PyExc_ValueError, "Array is too big; unable to allocate temporary space");
        return nullptr;
    }
    void *ret = PyMem_Malloc(alloc_size);
    if(!ret) {
        PyErr_NoMemory();
        return nullptr;
    }
    return ret;
}

template <typename scalar>
scalar *py_malloc(size_t n_elem) {
    return static_cast<scalar*>(adrt::_py::py_malloc(n_elem, sizeof(scalar)));
}

void py_free(void *ptr) {
    PyMem_Free(ptr);
}

}}} // End namespace adrt::_py

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

static PyObject *adrt_py_adrt(PyObject* /* self */, PyObject *arg) {
    // Process function arguments
    PyArrayObject *const I = adrt::_py::extract_array(arg);
    if(!I) {
        return nullptr;
    }
    // Extract shapes and check sizes
    std::array<size_t, 3> input_shape;
    if(!adrt::_py::array_shape<2, 3>(I).store_value(input_shape)) {
        return nullptr;
    }
    if(!adrt::adrt_is_valid_shape(input_shape)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must be square with a power of two shape");
        return nullptr;
    }
    // Compute effective output shape
    const std::array<size_t, 4> output_shape = adrt::adrt_result_shape(input_shape);
    size_t tmp_buf_elems;
    if(!adrt::_py::shape_product(adrt::adrt_buffer_shape(input_shape)).store_value(tmp_buf_elems)) {
        return nullptr;
    }
    // Process input array
    const int ndim = PyArray_NDIM(I);
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
    {
        PyArrayObject *ret = adrt::_py::new_array(ndim + 1, output_shape, NPY_FLOAT32);
        npy_float32 *tmp_buf = adrt::_py::py_malloc<npy_float32>(tmp_buf_elems);
        if(!ret || !tmp_buf) {
            adrt::_py::py_free(tmp_buf);
            Py_XDECREF(ret);
            return nullptr;
        }
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS;
        adrt::adrt_basic(static_cast<const npy_float32*>(PyArray_DATA(I)), input_shape, tmp_buf, static_cast<npy_float32*>(PyArray_DATA(ret)));
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS;
        adrt::_py::py_free(tmp_buf);
        return reinterpret_cast<PyObject*>(ret);
    }
    case NPY_FLOAT64:
    {
        PyArrayObject *ret = adrt::_py::new_array(ndim + 1, output_shape, NPY_FLOAT64);
        npy_float64 *tmp_buf = adrt::_py::py_malloc<npy_float64>(tmp_buf_elems);
        if(!ret || !tmp_buf) {
            adrt::_py::py_free(tmp_buf);
            Py_XDECREF(ret);
            return nullptr;
        }
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS;
        adrt::adrt_basic(static_cast<const npy_float64*>(PyArray_DATA(I)), input_shape, tmp_buf, static_cast<npy_float64*>(PyArray_DATA(ret)));
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS;
        adrt::_py::py_free(tmp_buf);
        return reinterpret_cast<PyObject*>(ret);
    }
    default:
        PyErr_SetString(PyExc_TypeError, "Unsupported array type");
        return nullptr;
    }
}

static PyObject *adrt_py_iadrt(PyObject* /* self */, PyObject *arg){
    // Process function arguments
    PyArrayObject *const I = adrt::_py::extract_array(arg);
    if(!I) {
        return nullptr;
    }
    // Extract shapes and check sizes
    std::array<size_t, 4> input_shape;
    if(!adrt::_py::array_shape<3, 4>(I).store_value(input_shape)) {
        return nullptr;
    }
    if(!adrt::iadrt_is_valid_shape(input_shape)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must have a valid ADRT shape");
        return nullptr;
    }
    // Compute effective output shape
    const std::array<size_t, 4> output_shape = adrt::iadrt_result_shape(input_shape);
    std::size_t tmp_buf_elems;
    if(!adrt::_py::shape_product(adrt::iadrt_buffer_shape(input_shape)).store_value(tmp_buf_elems)) {
        return nullptr;
    }
    // Process input array
    const int ndim = PyArray_NDIM(I);
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
    {
        PyArrayObject *ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT32);
        npy_float32 *tmp_buf = adrt::_py::py_malloc<npy_float32>(tmp_buf_elems);
        if(!ret || !tmp_buf) {
            adrt::_py::py_free(tmp_buf);
            Py_XDECREF(ret);
            return nullptr;
        }
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS;
        adrt::iadrt_basic(static_cast<const npy_float32*>(PyArray_DATA(I)), input_shape, tmp_buf, static_cast<npy_float32*>(PyArray_DATA(ret)));
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS;
        adrt::_py::py_free(tmp_buf);
        return reinterpret_cast<PyObject*>(ret);
    }
    case NPY_FLOAT64:
    {
        PyArrayObject *ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT64);
        npy_float64 *tmp_buf = adrt::_py::py_malloc<npy_float64>(tmp_buf_elems);
        if(!ret || !tmp_buf) {
            adrt::_py::py_free(tmp_buf);
            Py_XDECREF(ret);
            return nullptr;
        }
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS;
        adrt::iadrt_basic(static_cast<const npy_float64*>(PyArray_DATA(I)), input_shape, tmp_buf, static_cast<npy_float64*>(PyArray_DATA(ret)));
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS;
        adrt::_py::py_free(tmp_buf);
        return reinterpret_cast<PyObject*>(ret);
    }
    default:
        PyErr_SetString(PyExc_TypeError, "Unsupported array type");
        return nullptr;
    }
}

static PyObject *adrt_py_bdrt(PyObject* /* self */, PyObject *arg) {
    // Process function arguments
    PyArrayObject *const I = adrt::_py::extract_array(arg);
    if(!I) {
        return nullptr;
    }
    // Extract shapes and check sizes
    std::array<size_t, 4> input_shape;
    if(!adrt::_py::array_shape<3, 4>(I).store_value(input_shape)) {
        return nullptr;
    }
    if(!adrt::bdrt_is_valid_shape(input_shape)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must have a valid ADRT shape");
        return nullptr;
    }
    // Compute effective output shape
    const std::array<size_t, 4> output_shape = adrt::bdrt_result_shape(input_shape);
    std::size_t tmp_buf_elems;
    if(!adrt::_py::shape_product(adrt::bdrt_buffer_shape(input_shape)).store_value(tmp_buf_elems)) {
        return nullptr;
    }
    // Process input array
    const int ndim = PyArray_NDIM(I);
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
    {
        PyArrayObject *ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT32);
        npy_float32 *tmp_buf = adrt::_py::py_malloc<npy_float32>(tmp_buf_elems);
        if(!ret || !tmp_buf) {
            adrt::_py::py_free(tmp_buf);
            Py_XDECREF(ret);
            return nullptr;
        }
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS;
        adrt::bdrt_basic(static_cast<const npy_float32*>(PyArray_DATA(I)), input_shape, tmp_buf, static_cast<npy_float32*>(PyArray_DATA(ret)));
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS;
        adrt::_py::py_free(tmp_buf);
        return reinterpret_cast<PyObject*>(ret);
    }
    case NPY_FLOAT64:
    {
        PyArrayObject *ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT64);
        npy_float64 *tmp_buf = adrt::_py::py_malloc<npy_float64>(tmp_buf_elems);
        if(!ret || !tmp_buf) {
            adrt::_py::py_free(tmp_buf);
            Py_XDECREF(ret);
            return nullptr;
        }
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS;
        adrt::bdrt_basic(static_cast<const npy_float64*>(PyArray_DATA(I)), input_shape, tmp_buf, static_cast<npy_float64*>(PyArray_DATA(ret)));
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS;
        adrt::_py::py_free(tmp_buf);
        return reinterpret_cast<PyObject*>(ret);
    }
    default:
        PyErr_SetString(PyExc_TypeError, "Unsupported array type");
        return nullptr;
    }
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

static PyObject *adrt_py_num_iters(PyObject* /* self */, PyObject *arg){
    size_t val = PyLong_AsSize_t(arg);
    if(PyErr_Occurred()) {
        return nullptr;
    }
    return PyLong_FromLong(adrt::num_iters(val));
}

static PyMethodDef adrt_cdefs_methods[] = {
    {"adrt", adrt_py_adrt, METH_O, "Compute the ADRT"},
    {"iadrt", adrt_py_iadrt, METH_O, "Compute the inverse ADRT"},
    {"bdrt", adrt_py_bdrt, METH_O, "Compute the backprojection of the ADRT"},
    {"num_iters", adrt_py_num_iters, METH_O, "Compute the number of iterations needed for the ADRT"},
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
ADRT_BEGIN_EXPORT

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

ADRT_END_EXPORT

} // extern "C"
