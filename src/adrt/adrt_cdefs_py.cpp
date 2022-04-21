/*
 * Copyright (c) 2022 Karl Otness, Donsub Rim
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

#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x03080000
// Include this first
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_15_API_VERSION
#include <numpy/arrayobject.h>

#include <array>
#include <limits>
#include <type_traits>
#include <cassert>
#include <cstddef>

#include "adrt_cdefs_common.hpp"
#include "adrt_cdefs_adrt.hpp"
#include "adrt_cdefs_iadrt.hpp"
#include "adrt_cdefs_bdrt.hpp"
#include "adrt_cdefs_interp_adrtcart.hpp"
#include "adrt_cdefs_fmg.hpp"

#ifndef NDEBUG
#pragma message ("Building with assertions enabled")
#endif

using namespace adrt::_literals;
using std::size_t;

namespace adrt { namespace _py { namespace {

PyArrayObject *extract_array(PyObject *arg) {
    assert(arg);
    if(!PyArray_Check(arg)) {
        // This isn't an array
        PyErr_SetString(PyExc_TypeError, "Argument must be a NumPy array or compatible subclass");
        return nullptr;
    }
    PyArrayObject *const arr = reinterpret_cast<PyArrayObject*>(arg);
    if(!PyArray_ISCARRAY_RO(arr)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must be C-order, contiguous, aligned, and native byte order");
        return nullptr;
    }
    return arr;
}

PyObject *array_to_pyobject(PyArrayObject *arr) {
    return reinterpret_cast<PyObject*>(arr);
}

adrt::_common::Optional<size_t> extract_size_t(PyObject *arg) {
    assert(arg);
    const size_t val = PyLong_AsSize_t(arg);
    if(val == static_cast<size_t>(-1) && PyErr_Occurred()) {
        return {};
    }
    return {val};
}

adrt::_common::Optional<int> extract_int(PyObject *arg) {
    assert(arg);
    const long val = PyLong_AsLong(arg);
    if(val == -1L) {
        PyObject *const exc = PyErr_Occurred();
        if(exc) {
            // Error occurred
            if(PyErr_GivenExceptionMatches(exc, PyExc_OverflowError)) {
                // If it's an OverflowError replace the message with one about int
                PyErr_SetString(PyExc_OverflowError, "Python int too large to convert to C int");
            }
            return {};
        }
    }
    else if(val < std::numeric_limits<int>::min() || val > std::numeric_limits<int>::max()) {
        PyErr_SetString(PyExc_OverflowError, "Python int too large to convert to C int");
        return {};
    }
    return {static_cast<int>(val)};
}

template <size_t min_dim, size_t max_dim>
adrt::_common::Optional<std::array<size_t, max_dim>> array_shape(PyArrayObject *arr) {
    static_assert(min_dim <= max_dim, "Min dimensions must be less than max dimensions.");
    static_assert(min_dim > 0u, "Min dimensions must be positive.");
    assert(arr);
    std::array<size_t, max_dim> shape_arr;
    const int sndim = PyArray_NDIM(arr);
    const unsigned int ndim = static_cast<unsigned int>(sndim);
    if(sndim < 0 || ndim < min_dim || ndim > max_dim) {
        PyErr_Format(PyExc_ValueError, "Invalid number of dimensions for input array: %d (must be between %zu and %zu)", sndim, min_dim, max_dim);
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
        else if(static_cast<npy_uintp>(shape) > std::numeric_limits<size_t>::max()) {
            PyErr_SetString(PyExc_ValueError, "Maximum allowed dimension exceeded");
            return {};
        }
        shape_arr[i + (max_dim - ndim)] = static_cast<size_t>(shape);
    }
    return {shape_arr};
}

template <size_t n_virtual_dim>
PyArrayObject *new_array(int ndim, const std::array<size_t, n_virtual_dim> &virtual_shape, int typenum) {
    static_assert(n_virtual_dim > 0u, "Need at least one shape dimension");
    const unsigned int undim = static_cast<unsigned int>(ndim);
    if(undim > n_virtual_dim || ndim <= 0) {
        // This would be a bug and should have been caught earlier. Handle it as well as we can.
        PyErr_Format(PyExc_AssertionError, "BUG: Invalid number of dimensions computed for output array (requested %d but should be between 1 and %zu)", ndim, n_virtual_dim);
        return nullptr;
    }
    std::array<npy_intp, n_virtual_dim> new_shape;
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
    PyObject *const arr = PyArray_SimpleNew(ndim, new_shape.data(), typenum);
    if(!arr) {
        if(!PyErr_Occurred()) {
            // Don't shadow errors that NumPy may have set
            PyErr_NoMemory();
        }
        return nullptr;
    }
    assert(PyArray_Check(arr));
    assert(PyArray_ISCARRAY(reinterpret_cast<PyArrayObject*>(arr)));
    return reinterpret_cast<PyArrayObject*>(arr);
}

template <size_t ndim>
adrt::_common::Optional<size_t> shape_product(const std::array<size_t, ndim> &shape) {
    static_assert(ndim > 0u, "Need at least one shape dimension");
    const adrt::_common::Optional<size_t> n_elem = adrt::_common::shape_product(shape);
    if(!n_elem) {
        PyErr_SetString(PyExc_ValueError, "Array is too big; unable to allocate temporary space");
    }
    return n_elem;
}

template <size_t N, size_t... Ints>
adrt::_common::Optional<std::array<PyObject*, N>> unpack_tuple(PyObject *tuple, const char *name, adrt::_common::index_sequence<Ints...>) {
    static_assert(N >= 1u, "Must accept at least one argument");
    static_assert(N <= static_cast<size_t>(std::numeric_limits<Py_ssize_t>::max()), "Required tuple size is too large for Py_ssize_t");
    static_assert(std::is_same<adrt::_common::index_sequence<Ints...>, adrt::_common::make_index_sequence<N>>::value, "Wrong list of indices. Do not call this overload directly!");
    assert(tuple);
    assert(name);
    std::array<PyObject*, N> ret;
    const bool ok = PyArg_UnpackTuple(tuple, name, static_cast<Py_ssize_t>(N), static_cast<Py_ssize_t>(N), &std::get<Ints>(ret)...);
    if(!ok) {
        return {};
    }
    return {ret};
}

template <size_t N>
adrt::_common::Optional<std::array<PyObject*, N>> unpack_tuple(PyObject *tuple, const char *name) {
    return adrt::_py::unpack_tuple<N>(tuple, name, adrt::_common::make_index_sequence<N>{});
}

void *py_malloc(size_t n_elem, size_t elem_size) {
    const adrt::_common::Optional<size_t> alloc_size = adrt::_common::mul_check(n_elem, elem_size);
    if(!alloc_size) {
        PyErr_SetString(PyExc_ValueError, "Array is too big; unable to allocate temporary space");
        return nullptr;
    }
    void *const ret = PyMem_Malloc(*alloc_size);
    if(!ret) {
        PyErr_NoMemory();
        return nullptr;
    }
    return ret;
}

bool module_add_object_ref(PyObject *module, const char *name, PyObject *value) {
    assert(module);
    assert(name);
    assert(value);
    assert(PyModule_Check(module));
    Py_INCREF(value);
    if(PyModule_AddObject(module, name, value) != 0) {
        Py_DECREF(value);
        return false;
    }
    return true;
}

bool module_add_bool(PyObject *module, const char *name, bool value) {
    return adrt::_py::module_add_object_ref(module, name, value ? Py_True : Py_False);
}

template <typename scalar>
scalar *py_malloc(size_t n_elem) {
    static_assert(!std::is_same<PyObject*, scalar*>::value && !std::is_same<PyArrayObject*, scalar*>::value, "Do not malloc Python objects!");
    return static_cast<scalar*>(adrt::_py::py_malloc(n_elem, sizeof(scalar)));
}

void py_free(void *ptr) {
    PyMem_Free(ptr);
}

void py_free(PyObject *obj) = delete;
void py_free(PyArrayObject *obj) = delete;

void xdecref(PyObject *obj) {
    if(obj) {
        Py_DECREF(obj);
    }
}

void xdecref(PyArrayObject *arr) {
    adrt::_py::xdecref(adrt::_py::array_to_pyobject(arr));
}

}}} // End namespace adrt::_py

extern "C" {

static PyObject *adrt_py_adrt(PyObject* /* self */, PyObject *arg) {
    // Process function arguments
    PyArrayObject *const I = adrt::_py::extract_array(arg);
    if(!I) {
        return nullptr;
    }
    // Extract shapes and check sizes
    const adrt::_common::Optional<std::array<size_t, 3>> input_shape = adrt::_py::array_shape<2, 3>(I);
    if(!input_shape) {
        return nullptr;
    }
    if(!adrt::adrt_is_valid_shape(*input_shape)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must be square with a power of two shape");
        return nullptr;
    }
    // Compute effective output shape
    const std::array<size_t, 4> output_shape = adrt::adrt_result_shape(*input_shape);
    const adrt::_common::Optional<size_t> tmp_buf_elems = adrt::_py::shape_product(adrt::adrt_buffer_shape(*input_shape));
    if(!tmp_buf_elems) {
        return nullptr;
    }
    // Process input array
    const int ndim = PyArray_NDIM(I);
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim + 1, output_shape, NPY_FLOAT32);
        npy_float32 *const tmp_buf = adrt::_py::py_malloc<npy_float32>(*tmp_buf_elems);
        if(!ret || !tmp_buf) {
            adrt::_py::py_free(tmp_buf);
            adrt::_py::xdecref(ret);
            return nullptr;
        }
        const npy_float32 *const in_data = static_cast<npy_float32*>(PyArray_DATA(I));
        npy_float32 *const out_data = static_cast<npy_float32*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::adrt_basic(in_data, *input_shape, tmp_buf, out_data);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        adrt::_py::py_free(tmp_buf);
        return adrt::_py::array_to_pyobject(ret);
    }
    case NPY_FLOAT64:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim + 1, output_shape, NPY_FLOAT64);
        npy_float64 *const tmp_buf = adrt::_py::py_malloc<npy_float64>(*tmp_buf_elems);
        if(!ret || !tmp_buf) {
            adrt::_py::py_free(tmp_buf);
            adrt::_py::xdecref(ret);
            return nullptr;
        }
        const npy_float64 *const in_data = static_cast<npy_float64*>(PyArray_DATA(I));
        npy_float64 *const out_data = static_cast<npy_float64*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::adrt_basic(in_data, *input_shape, tmp_buf, out_data);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        adrt::_py::py_free(tmp_buf);
        return adrt::_py::array_to_pyobject(ret);
    }
    default:
        PyErr_SetString(PyExc_TypeError, "Unsupported array type");
        return nullptr;
    }
}

static PyObject *adrt_py_adrt_step(PyObject* /* self */, PyObject *args) {
    // Unpack function arguments
    const adrt::_common::Optional<std::array<PyObject*, 2>> unpacked_args = adrt::_py::unpack_tuple<2>(args, "adrt_step");
    if(!unpacked_args) {
        return nullptr;
    }
    // Process array argument
    PyArrayObject *const I = adrt::_py::extract_array(std::get<0>(*unpacked_args));
    if(!I) {
        return nullptr;
    }
    // Extract shape and check sizes
    const adrt::_common::Optional<std::array<size_t, 4>> input_shape = adrt::_py::array_shape<3, 4>(I);
    if(!input_shape) {
        return nullptr;
    }
    if(!adrt::adrt_step_is_valid_shape(*input_shape)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must have valid shape for ADRT, use adrt_init");
        return nullptr;
    }
    // Process int argument
    const adrt::_common::Optional<int> iter = adrt::_py::extract_int(std::get<1>(*unpacked_args));
    if(!iter) {
        return nullptr;
    }
    // Check range of iter
    if(!adrt::adrt_step_is_valid_iter(*input_shape, *iter)) {
        PyErr_SetString(PyExc_ValueError, "Parameter step is out of range for provided array shape, use num_iters");
        return nullptr;
    }
    // Compute effective output shape
    const std::array<size_t, 4> output_shape = adrt::adrt_step_result_shape(*input_shape);
    // Process input array
    const int ndim = PyArray_NDIM(I);
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT32);
        if(!ret) {
            return nullptr;
        }
        const npy_float32 *const in_data = static_cast<npy_float32*>(PyArray_DATA(I));
        npy_float32 *const out_data = static_cast<npy_float32*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::adrt_step(in_data, *input_shape, out_data, *iter);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        return adrt::_py::array_to_pyobject(ret);
    }
    case NPY_FLOAT64:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT64);
        if(!ret) {
            return nullptr;
        }
        const npy_float64 *const in_data = static_cast<npy_float64*>(PyArray_DATA(I));
        npy_float64 *const out_data = static_cast<npy_float64*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::adrt_step(in_data, *input_shape, out_data, *iter);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        return adrt::_py::array_to_pyobject(ret);
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
    const adrt::_common::Optional<std::array<size_t, 4>> input_shape = adrt::_py::array_shape<3, 4>(I);
    if(!input_shape) {
        return nullptr;
    }
    if(!adrt::iadrt_is_valid_shape(*input_shape)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must have a valid ADRT output shape");
        return nullptr;
    }
    // Compute effective output shape
    const std::array<size_t, 4> output_shape = adrt::iadrt_result_shape(*input_shape);
    const adrt::_common::Optional<size_t> tmp_buf_elems = adrt::_py::shape_product(adrt::iadrt_buffer_shape(*input_shape));
    if(!tmp_buf_elems) {
        return nullptr;
    }
    // Process input array
    const int ndim = PyArray_NDIM(I);
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT32);
        npy_float32 *const tmp_buf = adrt::_py::py_malloc<npy_float32>(*tmp_buf_elems);
        if(!ret || !tmp_buf) {
            adrt::_py::py_free(tmp_buf);
            adrt::_py::xdecref(ret);
            return nullptr;
        }
        const npy_float32 *const in_data = static_cast<npy_float32*>(PyArray_DATA(I));
        npy_float32 *const out_data = static_cast<npy_float32*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::iadrt_basic(in_data, *input_shape, tmp_buf, out_data);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        adrt::_py::py_free(tmp_buf);
        return adrt::_py::array_to_pyobject(ret);
    }
    case NPY_FLOAT64:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT64);
        npy_float64 *const tmp_buf = adrt::_py::py_malloc<npy_float64>(*tmp_buf_elems);
        if(!ret || !tmp_buf) {
            adrt::_py::py_free(tmp_buf);
            adrt::_py::xdecref(ret);
            return nullptr;
        }
        const npy_float64 *const in_data = static_cast<npy_float64*>(PyArray_DATA(I));
        npy_float64 *const out_data = static_cast<npy_float64*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::iadrt_basic(in_data, *input_shape, tmp_buf, out_data);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        adrt::_py::py_free(tmp_buf);
        return adrt::_py::array_to_pyobject(ret);
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
    const adrt::_common::Optional<std::array<size_t, 4>> input_shape = adrt::_py::array_shape<3, 4>(I);
    if(!input_shape) {
        return nullptr;
    }
    if(!adrt::bdrt_is_valid_shape(*input_shape)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must have a valid ADRT output shape");
        return nullptr;
    }
    // Compute effective output shape
    const std::array<size_t, 4> output_shape = adrt::bdrt_result_shape(*input_shape);
    const adrt::_common::Optional<size_t> tmp_buf_elems = adrt::_py::shape_product(adrt::bdrt_buffer_shape(*input_shape));
    if(!tmp_buf_elems) {
        return nullptr;
    }
    // Process input array
    const int ndim = PyArray_NDIM(I);
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT32);
        npy_float32 *const tmp_buf = adrt::_py::py_malloc<npy_float32>(*tmp_buf_elems);
        if(!ret || !tmp_buf) {
            adrt::_py::py_free(tmp_buf);
            adrt::_py::xdecref(ret);
            return nullptr;
        }
        const npy_float32 *const in_data = static_cast<npy_float32*>(PyArray_DATA(I));
        npy_float32 *const out_data = static_cast<npy_float32*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::bdrt_basic(in_data, *input_shape, tmp_buf, out_data);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        adrt::_py::py_free(tmp_buf);
        return adrt::_py::array_to_pyobject(ret);
    }
    case NPY_FLOAT64:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT64);
        npy_float64 *const tmp_buf = adrt::_py::py_malloc<npy_float64>(*tmp_buf_elems);
        if(!ret || !tmp_buf) {
            adrt::_py::py_free(tmp_buf);
            adrt::_py::xdecref(ret);
            return nullptr;
        }
        const npy_float64 *const in_data = static_cast<npy_float64*>(PyArray_DATA(I));
        npy_float64 *const out_data = static_cast<npy_float64*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::bdrt_basic(in_data, *input_shape, tmp_buf, out_data);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        adrt::_py::py_free(tmp_buf);
        return adrt::_py::array_to_pyobject(ret);
    }
    default:
        PyErr_SetString(PyExc_TypeError, "Unsupported array type");
        return nullptr;
    }
}

static PyObject *adrt_py_bdrt_step(PyObject* /* self */, PyObject *args) {
    // Unpack function arguments
    const adrt::_common::Optional<std::array<PyObject*, 2>> unpacked_args = adrt::_py::unpack_tuple<2>(args, "bdrt_step");
    if(!unpacked_args) {
        return nullptr;
    }
    // Process array argument
    PyArrayObject *const I = adrt::_py::extract_array(std::get<0>(*unpacked_args));
    if(!I) {
        return nullptr;
    }
    // Extract shape and check sizes
    const adrt::_common::Optional<std::array<size_t, 4>> input_shape = adrt::_py::array_shape<3, 4>(I);
    if(!input_shape) {
        return nullptr;
    }
    if(!adrt::bdrt_step_is_valid_shape(*input_shape)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must have a valid ADRT output shape");
        return nullptr;
    }
    // Process int argument
    const adrt::_common::Optional<int> iter = adrt::_py::extract_int(std::get<1>(*unpacked_args));
    if(!iter) {
        return nullptr;
    }
    // Check range of iter
    if(!adrt::bdrt_step_is_valid_iter(*input_shape, *iter)) {
        PyErr_SetString(PyExc_ValueError, "Parameter step is out of range for provided array shape, use num_iters");
        return nullptr;
    }
    // Compute effective output shape
    const std::array<size_t, 4> output_shape = adrt::bdrt_step_result_shape(*input_shape);
    // Process input array
    const int ndim = PyArray_NDIM(I);
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT32);
        if(!ret) {
            return nullptr;
        }
        const npy_float32 *const in_data = static_cast<npy_float32*>(PyArray_DATA(I));
        npy_float32 *const out_data = static_cast<npy_float32*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::bdrt_step(in_data, *input_shape, out_data, *iter);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        return adrt::_py::array_to_pyobject(ret);
    }
    case NPY_FLOAT64:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT64);
        if(!ret) {
            return nullptr;
        }
        const npy_float64 *const in_data = static_cast<npy_float64*>(PyArray_DATA(I));
        npy_float64 *const out_data = static_cast<npy_float64*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::bdrt_step(in_data, *input_shape, out_data, *iter);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        return adrt::_py::array_to_pyobject(ret);
    }
    default:
        PyErr_SetString(PyExc_TypeError, "Unsupported array type");
        return nullptr;
    }
}

static PyObject *adrt_py_interp_adrtcart(PyObject* /* self */, PyObject *arg) {
    // Process function arguments
    PyArrayObject *const I = adrt::_py::extract_array(arg);
    if(!I) {
        return nullptr;
    }
    // Extract shapes and check sizes
    const adrt::_common::Optional<std::array<size_t, 4>> input_shape = adrt::_py::array_shape<3, 4>(I);
    if(!input_shape) {
        return nullptr;
    }
    if(!adrt::interp_adrtcart_is_valid_shape(*input_shape)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must have a valid ADRT output shape");
        return nullptr;
    }
    // Compute effective output shape
    const std::array<size_t, 3> output_shape = adrt::interp_adrtcart_result_shape(*input_shape);
    // Process input array
    const int ndim = PyArray_NDIM(I);
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim - 1, output_shape, NPY_FLOAT32);
        if(!ret) {
            return nullptr;
        }
        const npy_float32 *const in_data = static_cast<npy_float32*>(PyArray_DATA(I));
        npy_float32 *const out_data = static_cast<npy_float32*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::interp_adrtcart(in_data, *input_shape, out_data);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        return adrt::_py::array_to_pyobject(ret);
    }
    case NPY_FLOAT64:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim - 1, output_shape, NPY_FLOAT64);
        if(!ret) {
            return nullptr;
        }
        const npy_float64 *const in_data = static_cast<npy_float64*>(PyArray_DATA(I));
        npy_float64 *const out_data = static_cast<npy_float64*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::interp_adrtcart(in_data, *input_shape, out_data);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        return adrt::_py::array_to_pyobject(ret);
    }
    default:
        PyErr_SetString(PyExc_TypeError, "Unsupported array type");
        return nullptr;
    }
}

static PyObject *adrt_py_fmg_restriction(PyObject* /* self */, PyObject *arg) {
    // Process function arguments
    PyArrayObject *const I = adrt::_py::extract_array(arg);
    if(!I) {
        return nullptr;
    }
    // Extract shapes and check sizes
    const adrt::_common::Optional<std::array<size_t, 4>> input_shape = adrt::_py::array_shape<3, 4>(I);
    if(!input_shape) {
        return nullptr;
    }
    if(!adrt::fmg_restriction_is_valid_shape(*input_shape)) {
        PyErr_SetString(PyExc_ValueError, "Provided array must have a valid ADRT output shape");
        return nullptr;
    }
    // Compute effective output shape
    const std::array<size_t, 4> output_shape = adrt::fmg_restriction_result_shape(*input_shape);
    // Process input array
    const int ndim = PyArray_NDIM(I);
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT32);
        if(!ret) {
            return nullptr;
        }
        const npy_float32 *const in_data = static_cast<npy_float32*>(PyArray_DATA(I));
        npy_float32 *const out_data = static_cast<npy_float32*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::fmg_restriction(in_data, *input_shape, out_data);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        return adrt::_py::array_to_pyobject(ret);
    }
    case NPY_FLOAT64:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT64);
        if(!ret) {
            return nullptr;
        }
        const npy_float64 *const in_data = static_cast<npy_float64*>(PyArray_DATA(I));
        npy_float64 *const out_data = static_cast<npy_float64*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::fmg_restriction(in_data, *input_shape, out_data);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        return adrt::_py::array_to_pyobject(ret);
    }
    default:
        PyErr_SetString(PyExc_TypeError, "Unsupported array type");
        return nullptr;
    }
}

static PyObject *adrt_py_fmg_prolongation(PyObject* /* self */, PyObject *arg) {
    // Process function arguments
    PyArrayObject *const I = adrt::_py::extract_array(arg);
    if(!I) {
        return nullptr;
    }
    // Extract shapes and check sizes
    const adrt::_common::Optional<std::array<size_t, 3>> input_shape = adrt::_py::array_shape<2, 3>(I);
    if(!input_shape) {
        return nullptr;
    }
    if(!adrt::fmg_prolongation_is_valid_shape(*input_shape)) {
        PyErr_SetString(PyExc_ValueError, "Provided array is too large for prolongation operator");
        return nullptr;
    }
    // Compute effective output shape
    const std::array<size_t, 3> output_shape = adrt::fmg_prolongation_result_shape(*input_shape);
    // Process input array
    const int ndim = PyArray_NDIM(I);
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT32);
        if(!ret) {
            return nullptr;
        }
        const npy_float32 *const in_data = static_cast<npy_float32*>(PyArray_DATA(I));
        npy_float32 *const out_data = static_cast<npy_float32*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::fmg_prolongation(in_data, *input_shape, out_data);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        return adrt::_py::array_to_pyobject(ret);
    }
    case NPY_FLOAT64:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT64);
        if(!ret) {
            return nullptr;
        }
        const npy_float64 *const in_data = static_cast<npy_float64*>(PyArray_DATA(I));
        npy_float64 *const out_data = static_cast<npy_float64*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::fmg_prolongation(in_data, *input_shape, out_data);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        return adrt::_py::array_to_pyobject(ret);
    }
    default:
        PyErr_SetString(PyExc_TypeError, "Unsupported array type");
        return nullptr;
    }
}

static PyObject *adrt_py_fmg_highpass(PyObject* /* self */, PyObject *arg) {
    // Process function arguments
    PyArrayObject *const I = adrt::_py::extract_array(arg);
    if(!I) {
        return nullptr;
    }
    // Extract shapes and check sizes
    const adrt::_common::Optional<std::array<size_t, 3>> input_shape = adrt::_py::array_shape<2, 3>(I);
    if(!input_shape) {
        return nullptr;
    }
    if(!adrt::fmg_highpass_is_valid_shape(*input_shape)) {
        PyErr_SetString(PyExc_ValueError, "Provided array is too small to high-pass filter");
        return nullptr;
    }
    // Compute effective output shape
    const std::array<size_t, 3> output_shape = adrt::fmg_highpass_result_shape(*input_shape);
    // Process input array
    const int ndim = PyArray_NDIM(I);
    switch(PyArray_TYPE(I)) {
    case NPY_FLOAT32:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT32);
        if(!ret) {
            return nullptr;
        }
        const npy_float32 *const in_data = static_cast<npy_float32*>(PyArray_DATA(I));
        npy_float32 *const out_data = static_cast<npy_float32*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::fmg_highpass(in_data, *input_shape, out_data);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        return adrt::_py::array_to_pyobject(ret);
    }
    case NPY_FLOAT64:
    {
        PyArrayObject *const ret = adrt::_py::new_array(ndim, output_shape, NPY_FLOAT64);
        if(!ret) {
            return nullptr;
        }
        const npy_float64 *const in_data = static_cast<npy_float64*>(PyArray_DATA(I));
        npy_float64 *const out_data = static_cast<npy_float64*>(PyArray_DATA(ret));
        // NO PYTHON API BELOW THIS POINT
        Py_BEGIN_ALLOW_THREADS
        adrt::fmg_highpass(in_data, *input_shape, out_data);
        // PYTHON API ALLOWED BELOW THIS POINT
        Py_END_ALLOW_THREADS
        return adrt::_py::array_to_pyobject(ret);
    }
    default:
        PyErr_SetString(PyExc_TypeError, "Unsupported array type");
        return nullptr;
    }
}

static PyObject *adrt_py_num_iters(PyObject* /* self */, PyObject *arg){
    const adrt::_common::Optional<size_t> val = adrt::_py::extract_size_t(arg);
    if(!val) {
        return nullptr;
    }
    return PyLong_FromLong(adrt::num_iters(*val));
}

static PyMethodDef adrt_cdefs_methods[] = {
    {"adrt", adrt_py_adrt, METH_O, "Compute the ADRT"},
    {"adrt_step", adrt_py_adrt_step, METH_VARARGS, "Compute one step of the ADRT"},
    {"iadrt", adrt_py_iadrt, METH_O, "Compute the inverse ADRT"},
    {"bdrt", adrt_py_bdrt, METH_O, "Compute the backprojection of the ADRT"},
    {"bdrt_step", adrt_py_bdrt_step, METH_VARARGS, "Compute one step of the bdrt"},
    {"num_iters", adrt_py_num_iters, METH_O, "Compute the number of iterations needed for the ADRT"},
    {"interp_to_cart", adrt_py_interp_adrtcart, METH_O, "Interpolate ADRT output to Cartesian coordinate system"},
    {"press_fmg_restriction", adrt_py_fmg_restriction, METH_O, "Multigrid restriction operator"},
    {"press_fmg_prolongation", adrt_py_fmg_prolongation, METH_O, "Multigrid prolongation operator"},
    {"press_fmg_highpass", adrt_py_fmg_highpass, METH_O, "Multigrid high-pass filter"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef adrt_cdefs_module = {
    PyModuleDef_HEAD_INIT,
    "adrt._adrt_cdefs",
    "Native implementations of core functionality.\n\n"
    ".. danger::\n"
    "   This module is not part of the public API surface. Do not use it!\n\n"
    "These functions are not part of the public API surface and may be changed\n"
    "or removed. Do not call them directly.",
    0,
    adrt_cdefs_methods,
    nullptr,
    // GC hooks below, unused
    nullptr, nullptr, nullptr
};

// Support Python<3.9 which doesn't set visibility
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(default)
#endif

PyMODINIT_FUNC
PyInit__adrt_cdefs()
{
    import_array();
    PyObject *const module = PyModule_Create(&adrt_cdefs_module);
    if(!module) {
        return nullptr;
    }
    if(!adrt::_py::module_add_bool(module, "openmp_enabled", adrt::_const::openmp_enabled())) {
        adrt::_py::xdecref(module);
        return nullptr;
    }
    return module;
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif

} // extern "C"
