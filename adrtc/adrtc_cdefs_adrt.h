#if !defined(ADRT_SCALAR) || !defined(ADRT_SHAPE)
#error "Must define the scalar types and allocator functions"
#endif

#ifndef ADRT_SCALAR
#define ADRT_SCALAR double
#endif

#ifndef ADRT_SHAPE
#define ADRT_SHAPE int
#endif

// Hack to calm linters: include Python
#ifndef PY_VERSION
#include <Python.h>
#endif

#define ADRT_FUNC_NAME_INNER(SCALAR) _adrt_impl_ ## SCALAR
#define ADRT_FUNC_NAME(SCALAR) ADRT_FUNC_NAME_INNER(SCALAR)

#include <stdbool.h>

static bool ADRT_FUNC_NAME(ADRT_SCALAR) (const ADRT_SCALAR *const data, ADRT_SHAPE *shape, ADRT_SCALAR *out) {
    // Allocate auxiliary memory
    ADRT_SCALAR *aux = PyMem_New(ADRT_SCALAR, 2 * shape[0] * shape[1]);
    if(!aux) {
        PyErr_NoMemory();
        return false;
    }

    // NO PYTHON API BELOW THIS POINT
    Py_BEGIN_ALLOW_THREADS;

    for (ADRT_SHAPE i = 0; i < shape[0]; i++) {
        for (ADRT_SHAPE j = 0; j < shape[1]; j++) {
            out[shape[1] * i + j] = 10 * data[shape[1] * i + j];
        }
    }

    // PYTHON API ALLOWED BELOW THIS POINT
    Py_END_ALLOW_THREADS;

    PyMem_Free(aux);
    return true;
}

#undef ADRT_FUNC_NAME
#undef ADRT_FUNC_NAME_INNER
