/*  C extensions for Approximate  Discrete Radon Transform
 *  See LICENSE
 */

#define PY_SSIZE_T_CLEAN
#define PY_LIMITED_API 0x03040000
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include "numpy/arrayobject.h"
#include <math.h>
#include <stdio.h>

/*
 * compute ADRT of a single quadrant
 */
void adrtq(PyArrayObject *a, npy_double **da, int n, int n1, int n2){

    int h,s,t,l,m;
    int L,M,N;
    npy_intp hs[2]={0,0};     /* integer pair for indices */

    N = (int) PyArray_DIM(a,0);
    M = (int) PyArray_DIM(a,1);

    for (h = N; h < 2*N; h++){
        for (s = 0; s < N; s++){
            hs[0] = h-N;
            hs[1] = s;
            da[h][s] = *(npy_double*) PyArray_GetPtr(a,&hs[0]);
        }
    }

    /* temporary scratch space */
    npy_double ** dat = malloc(sizeof(npy_double*)*3*N);
    for (h=0; h<3*N; h++){
        dat[h] = malloc(sizeof(npy_double*)*N);
        for (s=0; s<N; s++){
            dat[h][s] = 0.0;
        }
    }

    N = (int) pow(2.0, (double) n);
    /* section loop 2^n x 2^m image */
    for (m = n1; m < n2; m++){
        /* l-th 2^n x 2^m image */
        M = (int) pow(2.0, (double) m);
        L = (int) pow(2.0, (double) n-m-1);
        for (l = 0; l < L; l++){
            for (t = 0; t < M; t++){
                for (h = 0; h+t < 3*N; h++){
                    dat[h][2*t+2*l*M] = da[h][t+2*l*M]+da[h+t][t+(2*l+1)*M];
                }
                for (h = 0; h+t+1 < 3*N; h++){
                    dat[h][2*t+1+2*l*M] = da[h][t+2*l*M]+da[h+t+1][t+(2*l+1)*M];
                }
            }
        }
        /* update da */
        for (h = 0; h < 3*N; h++){
            for (s = 0; s < N; s++){
                da[h][s] = dat[h][s];
            }
        }
    }

    /* free scratch memory */
    free(dat);
    }

void bdrtq(PyArrayObject *da, npy_double **ba, int n){

    int h,s,l,m,mh,m0;
    int M,N;
    npy_intp hs[2]={0,0};     /* integer pair for indices */

    M = (int) PyArray_DIM(da,0);
    N = (int) PyArray_DIM(da,1);

    for (h = 0; h < M; h++){
        for (s = 0; s < N; s++){
            hs[0] = h;
            hs[1] = s;
            ba[h][s] = *(npy_double*) PyArray_GetPtr(da,&hs[0]);
        }
    }

    /* temporary scratch space */
    npy_double ** bat = malloc(sizeof(npy_double*)*M);
    for (h=0; h<M; h++){
        bat[h] = malloc(sizeof(npy_double*)*N);
        for (s=0; s<N; s++){
            bat[h][s] = 0.0;
        }
    }

    N = (int) pow(2.0, (double) n);
    /* section loop 2^n x 2^m image */
    for (m = n-1; m > -1; m--){
        /* l-th 2^n x 2^m image */
        mh = (int) pow(2.0, (double) m);   /* 2^m length of half-img */
        m0 = 2*mh;
        for (h = 0; h < M; h++){
            for (s = 0; s < mh; s++){
                for (l = 0; l < N; l += m0){
                    bat[h][l+s] = ba[h][l+2*s] + ba[h][l+2*s+1];
                    if ((h > 0) && (h+s < M)) {
                        bat[h+s][l+s+mh] = ba[h][l+2*s] + ba[h-1][l+2*s+1];
                    }
                    else if (h == 1) {
                        bat[h+s][l+s+mh] = ba[h][l+2*s];
                    }
                }
            }
        }
        /* update da */
        for (h = 0; h < M; h++){
            for (s = 0; s < N; s++){
                ba[h][s] = bat[h][s];
            }
        }
    }

    /* free scratch memory */
    free(bat);
    }

void fliplr(npy_double **a, npy_double **b, int M, int N){

    int i,j;

    for( i=0; i < M; i++){
        for( j=0; j < N; j++){
            b[i][N-1-j] = a[i][j];
        }
    }
    }

void tr(npy_double **a, npy_double **b, int M, int N){

    int i,j;

    for( i=0; i < M; i++){
        for( j=0; j < N; j++){
            b[j][i] = a[i][j];
        }
    }
}


void set_pyarray(npy_double **b, PyArrayObject *a, int M, int N){

    int i,j;
    npy_intp ii[2] = {0,0};

    for( i=0; i < M; i++){
        for( j=0; j < N; j++){
            ii[0] = i;
            ii[1] = j;
            *(npy_double*) PyArray_GetPtr(a, ii) = b[i][j];
        }
    }
}


static PyObject * bdrtpart(PyObject *self, PyObject *args){

    PyArrayObject *da;   /* input image */
    PyArrayObject *ba_out;   /* input image */

    int h,s,n,N,M;
    /* const int ndim = 2; */

    if (!PyArg_ParseTuple(args,"O!",&PyArray_Type,&da))
        return NULL;

    Py_INCREF(da);
    M = (int) PyArray_DIM(da,0);
    N = (int) PyArray_DIM(da,1);
    if (M != 3*N || (int) PyArray_NDIM(da) != 2)
        return NULL;

    n = (int) round(log2( (double) N));

    /* temporary scratch space */
    npy_double ** ba = malloc(sizeof(npy_double*)*M);
    for (h=0; h<M; h++){
        ba[h] = malloc(sizeof(npy_double*)*N);
        for (s=0; s<N; s++){
            ba[h][s] = 0.0;
        }
    }

    bdrtq(da,ba,n);

    Py_DECREF(da);

    npy_intp nmdim[2] = {N,N};
    npy_intp hs[2] = {0,0};

    /* prep for output */
    ba_out =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type,
                            PyArray_DescrFromType(NPY_DOUBLE),
                            PyArray_NDIM(da), nmdim, NULL,NULL,0,NULL);

    for (h = 0; h < N; h++){
        for (s = 0; s < N ; s++){
            hs[0] = h;
            hs[1] = s;
            *(npy_double*) PyArray_GetPtr(ba_out, hs) = ba[h+N][s];
        }
    }

    free(ba);

    return Py_BuildValue("O",ba_out);
    }


static PyObject * bdrt(PyObject *self, PyObject *args){

    PyArrayObject *daa;   /* input image */
    PyArrayObject *dab;   /* input image */
    PyArrayObject *dac;   /* input image */
    PyArrayObject *dad;   /* input image */

    PyArrayObject *baa_out;   /* input image */
    PyArrayObject *bab_out;   /* input image */
    PyArrayObject *bac_out;   /* input image */
    PyArrayObject *bad_out;   /* input image */

    int h,s,n,N,M;
    /* const int ndim = 2; */

    if (!PyArg_ParseTuple(args,"O!O!O!O!",&PyArray_Type,&daa,
        &PyArray_Type,&dab,&PyArray_Type,&dac,&PyArray_Type,&dad))
        return NULL;

    Py_INCREF(daa);
    Py_INCREF(dab);
    Py_INCREF(dac);
    Py_INCREF(dad);

    M = (int) PyArray_DIM(daa,0);
    N = (int) PyArray_DIM(daa,1);
    if (M != 3*N || (int) PyArray_NDIM(daa) != 2)
        return NULL;

    n = (int) round(log2( (double) N));

    /* temporary scratch space */
    npy_double ** baa = malloc(sizeof(npy_double*)*M);
    npy_double ** bab = malloc(sizeof(npy_double*)*M);
    npy_double ** bac = malloc(sizeof(npy_double*)*M);
    npy_double ** bad = malloc(sizeof(npy_double*)*M);
    for (h=0; h<M; h++){
        baa[h] = malloc(sizeof(npy_double*)*N);
        bab[h] = malloc(sizeof(npy_double*)*N);
        bac[h] = malloc(sizeof(npy_double*)*N);
        bad[h] = malloc(sizeof(npy_double*)*N);
        for (s=0; s<N; s++){
            baa[h][s] = 0.0;
            bab[h][s] = 0.0;
            bac[h][s] = 0.0;
            bad[h][s] = 0.0;
        }
    }

    bdrtq(daa,baa,n);
    bdrtq(dab,bab,n);
    bdrtq(dac,bac,n);
    bdrtq(dad,bad,n);

    Py_DECREF(daa);
    Py_DECREF(dab);
    Py_DECREF(dac);
    Py_DECREF(dad);

    npy_intp nmdim[2] = {N,N};
    npy_intp hs[2] = {0,0};

    /* prep for output */
    baa_out =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type,
                                PyArray_DescrFromType(NPY_DOUBLE),
                                PyArray_NDIM(daa), nmdim, NULL,NULL,0,NULL);

    bab_out =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type,
                                PyArray_DescrFromType(NPY_DOUBLE),
                                PyArray_NDIM(daa), nmdim, NULL,NULL,0,NULL);

    bac_out =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type,
                                PyArray_DescrFromType(NPY_DOUBLE),
                                PyArray_NDIM(daa), nmdim, NULL,NULL,0,NULL);

    bad_out =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type,
                                PyArray_DescrFromType(NPY_DOUBLE),
                                PyArray_NDIM(daa), nmdim, NULL,NULL,0,NULL);

    for (h = 0; h < N; h++){
        for (s = 0; s < N ; s++){
            hs[0] = h;
            hs[1] = s;
            /* restrict according to orientation */
            *(npy_double*) PyArray_GetPtr(baa_out, hs) = baa[h+N][s];
            *(npy_double*) PyArray_GetPtr(bab_out, hs) = bab[s+N][h];
            *(npy_double*) PyArray_GetPtr(bac_out, hs) = bac[2*N-1-s][h];
            *(npy_double*) PyArray_GetPtr(bad_out, hs) = bad[h+N][N-1-s];
        }
    }

    free(baa);
    free(bab);
    free(bac);
    free(bad);

    return Py_BuildValue("(OOOO)",baa_out,bab_out,bac_out,bad_out);
    }



/*
 * compute ADRT
 */
static PyObject * adrtpart(PyObject *self, PyObject *args){

    PyArrayObject *a;   /* input image */
    PyArrayObject *da_out;   /* input image */

    int h,s,n,N,M;
    /* const int ndim = 2; */
    int n1,n2;

    if (!PyArg_ParseTuple(args,"O!ii",&PyArray_Type,&a,&n1,&n2))
        return NULL;

    Py_INCREF(a);
    N = (int) PyArray_DIM(a,0);
    M = (int) PyArray_DIM(a,1);
    if (N != M || (int) PyArray_NDIM(a) != 2)
        return NULL;

    n = (int) round(log2( (double) N));

    /* force bounds */
    if (n2 > n) n2 = n;
    if (n1 < 0) n1 = 0;

    /* temporary scratch space */
    npy_double ** da = malloc(sizeof(npy_double*)*3*N);
    for (h=0; h<3*N; h++){
        da[h] = malloc(sizeof(npy_double*)*N);
        for (s=0; s<N; s++){
            da[h][s] = 0.0;
        }
    }

    adrtq(a,da,n,n1,n2);

    Py_DECREF(a);

    npy_intp nmdim[2] = {3*N,M};
    npy_intp hs[2] = {0,0};

    /* prep for output */
    da_out =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type,
                            PyArray_DescrFromType(NPY_DOUBLE),
                            PyArray_NDIM(a), nmdim, NULL,NULL,0,NULL);

    for (h = 0; h < 3*N ; h++){
        for (s = 0; s < M ; s++){
            hs[0] = h;
            hs[1] = s;
            *(npy_double*) PyArray_GetPtr(da_out, hs) = da[h][s];
        }
    }

    free(da);

    return Py_BuildValue("O",da_out);
    }


static PyObject * adrt(PyObject *self, PyObject *args){

    PyArrayObject *a;   /* input image */

    PyArrayObject *daa_out;   /* output adrt quadrant a */
    PyArrayObject *dab_out;   /* output adrt quadrant b */
    PyArrayObject *dac_out;   /* output adrt quadrant c */
    PyArrayObject *dad_out;   /* output adrt quadrant d */

    npy_intp hs[2] = {0,0};

    int h,s,n,N,M;
    /* const int ndim = 2; */

    if (!PyArg_ParseTuple(args,"O!",&PyArray_Type,&a))
        return NULL;

    Py_INCREF(a);

    N = (int) PyArray_DIM(a,0);
    M = (int) PyArray_DIM(a,1);
    if (N != M || (int) PyArray_NDIM(a) != 2)
        return NULL;

    n = (int) round(log2( (double) N));

    /* temporary scratch space */
    npy_double ** daa = malloc(sizeof(npy_double*)*3*N);
    npy_double ** dab = malloc(sizeof(npy_double*)*3*N);
    npy_double ** dac = malloc(sizeof(npy_double*)*3*N);
    npy_double ** dad = malloc(sizeof(npy_double*)*3*N);

    for (h=0; h<3*N; h++){
        daa[h] = malloc(sizeof(npy_double*)*N);
        dab[h] = malloc(sizeof(npy_double*)*N);
        dac[h] = malloc(sizeof(npy_double*)*N);
        dad[h] = malloc(sizeof(npy_double*)*N);
        for (s=0; s<N; s++){
            daa[h][s] = 0.0;
            dab[h][s] = 0.0;
            dac[h][s] = 0.0;
            dad[h][s] = 0.0;
        }
    }

    npy_double ** av = malloc(sizeof(npy_double*)*N);
    npy_double ** aw = malloc(sizeof(npy_double*)*N);
    npy_double ** a0 = malloc(sizeof(npy_double*)*N);

    for (h=0; h<N; h++){
        av[h] = malloc(sizeof(npy_double*)*N);
        aw[h] = malloc(sizeof(npy_double*)*N);
        a0[h] = malloc(sizeof(npy_double*)*N);
        for (s=0; s<N; s++){
            hs[0] = h;
            hs[1] = s;
            a0[h][s] = *(npy_double*) PyArray_GetPtr(a, hs);
            av[h][s] = 0.0;
            aw[h][s] = 0.0;
        }
    }

    adrtq(a,daa,n,0,n);             /* quadrant a */

    tr(a0,aw,N,N);              /* quadrant b */
    set_pyarray(aw,a,N,N);
    adrtq(a,dab,n,0,n);

    fliplr(a0,av,N,N);          /* quadrant d */
    tr(av,aw,N,N);
    set_pyarray(aw,a,N,N);
    adrtq(a,dac,n,0,n);

    fliplr(a0,aw,N,N);          /* quadrant c */
    set_pyarray(aw,a,N,N);
    adrtq(a,dad,n,0,n);

    set_pyarray(a0,a,N,N);
    Py_DECREF(a);

    npy_intp nmdim[2] = {3*N,N};

    /* prep for output */
    daa_out =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type,
                                PyArray_DescrFromType(NPY_DOUBLE),
                                PyArray_NDIM(a), nmdim, NULL,NULL,0,NULL);

    dab_out =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type,
                                PyArray_DescrFromType(NPY_DOUBLE),
                                PyArray_NDIM(a), nmdim, NULL,NULL,0,NULL);

    dac_out =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type,
                                PyArray_DescrFromType(NPY_DOUBLE),
                                PyArray_NDIM(a), nmdim, NULL,NULL,0,NULL);

    dad_out =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type,
                                PyArray_DescrFromType(NPY_DOUBLE),
                                PyArray_NDIM(a), nmdim, NULL,NULL,0,NULL);

    for (h = 0; h < 3*N ; h++){
        for (s = 0; s < N ; s++){
            hs[0] = h;
            hs[1] = s;
            *(npy_double*) PyArray_GetPtr(daa_out, hs) = daa[h][s];
            *(npy_double*) PyArray_GetPtr(dab_out, hs) = dab[h][s];
            *(npy_double*) PyArray_GetPtr(dac_out, hs) = dac[h][s];
            *(npy_double*) PyArray_GetPtr(dad_out, hs) = dad[h][s];
        }
    }

    free(daa);
    free(dab);
    free(dac);
    free(dad);

    free(aw);
    free(av);
    free(a0);

    return Py_BuildValue("(OOOO)",daa_out,dab_out,dac_out,dad_out);
}



/*
 * compute inverse ADRT from single quadrant
 */
void iadrtq(PyArrayObject *da, npy_double **a, int n){

    int h,s,t,l,m;
    int L,M,N;
    npy_double x;

    npy_intp hs[2] = {0,0};

    N = (int) round(pow(2.0, (double) n));

    for (h=0; h < 3*N; h++){
        for (s=0; s < N; s++){
            hs[0] = h;
            hs[1] = s;
            a[h][s] = *(npy_double*) PyArray_GetPtr(da,&hs[0]);
        }
    }

    /* temporary scratch memory */
    double ** at = malloc(sizeof(npy_double*)*3*N);
    for (h=0; h<3*N; h++){
        at[h] = malloc(sizeof(npy_double*)*N);
        for (s=0; s<N; s++){
            at[h][s] = 0.0;
        }
    }

    /* section loop 2^n x 2^m image */
    for (m = n-1; m > -1; m--){
        /* l-th 2^n x 2^m image */
        M = (int) pow(2.0, (double) m);
        L = (int) pow(2.0, (double) n-m-1);
        for (l = 0; l < L; l++){
            for (t = 0; t < M; t++){
                for (h = 0; h+1 < 3*N; h++){
                    at[h][t+(2*l)*M] = a[h+1][2*t  +(2*l)*M]
                                     - a[h  ][2*t+1+(2*l)*M];
                }
                for (h = t; h < 3*N; h++){
                    at[h][t+(2*l+1)*M] = a[h-t][2*t+1+(2*l)*M]
                                       - a[h-t][2*t  +(2*l)*M];
                }
                for (h = 0; h < t; h++){
                    at[h][t+(2*l+1)*M] = 0.0;
                }
            }
        }
        /* update da */
        for (s = 0; s < N; s++){
            x = 0.0;
            for (h = 0; h+1 < 3*N; h++){
                x += at[h][s];
                a[h+1][s] = x;
            }
        }
    }

    /* free scratch memory */
    free(at);
    }



/*
 * compute inverse ADRT
 */
static PyObject * iadrt(PyObject *self, PyObject *args){

    PyArrayObject *da;      /* input transform */
    PyArrayObject *a_out;   /* output image */

    int h,s,n,N,M;

    /* const int ndim = 2; */
    if (!PyArg_ParseTuple(args,"O!",&PyArray_Type,&da))
        return NULL;

    Py_INCREF(da);
    M = (int) PyArray_DIM(da,0);
    N = (int) PyArray_DIM(da,1);
    if (M != 3*N || (int) PyArray_NDIM(da) != 2)
        return NULL;

    n = (int) round(log2((double) N));

    /* allocate scratch memory */
    npy_double ** a = malloc(sizeof(npy_double*)*3*N);
    for (h=0; h<3*N; h++){
        a[h] = malloc(sizeof(npy_double*)*N);
        for (s=0; s<N; s++){
            a[h][s] = 0.0;
        }
    }

    iadrtq(da,a,n);

    Py_DECREF(da);

    npy_intp nmdim[2] = {N,N};
    npy_intp hs[2] = {0,0};

    /* prep for output */
    a_out =  (PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type,
                              PyArray_DescrFromType(NPY_DOUBLE),
                              PyArray_NDIM(da), nmdim, NULL,NULL,0,NULL);

    for (h = 0; h < N ; h++){
        for (s = 0; s < N ; s++){
            hs[0] = h;
            hs[1] = s;
            *(npy_double*) PyArray_GetPtr(a_out, hs) = a[h+N][s];
        }
    }

    free(a);

    return Py_BuildValue("O",a_out);
    }



static PyMethodDef AdrtModuleMethods[] = {
    {"adrt",     adrt, METH_VARARGS, "compute full ADRT of 2^N x 2^N image"},
    {"adrtpart", adrtpart, METH_VARARGS,
                            "compute partial ADRT of 2^N x 2^N image"},
    {"iadrt",   iadrt, METH_VARARGS, "invert ADRT of 2^N x 2^N image"},
    {"bdrt",     bdrt, METH_VARARGS,
             "compute full back-projection from ADRT of 2^N x 2^N image"},
    {"bdrtpart", bdrtpart, METH_VARARGS,
             "compute back-projection from ADRT of 2^N x 2^N image"},
    {NULL, NULL, 0, NULL}   /* Sentinel */
    };

static struct PyModuleDef AdrtModule = {
        PyModuleDef_HEAD_INIT,
        "_adrtc",
        "C routines for ADRT computation",
        0,
        AdrtModuleMethods,
        NULL,
        NULL,
        NULL,
        NULL};

PyMODINIT_FUNC PyInit__adrtc(void) {
    Py_Initialize();
    import_array();
    return PyModule_Create(&AdrtModule);
    }
