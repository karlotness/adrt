.. _cginverse:

Iterative Inverse
=================

This example makes use of facilities from SciPy. In particular, its
:func:`scipy.sparse.linalg.cg` routine. ::

   import numpy as np
   from scipy.sparse.linalg import LinearOperator, cg
   from matplotlib import pyplot as plt
   import adrt

.. plot::
   :context: reset
   :include-source: false
   :nofigs:

   from scipy.sparse.linalg import LinearOperator, cg

The operation defined by :func:`adrt.adrt` is linear. If we consider
its matrix :math:`A`, then the operation :func:`adrt.bdrt` defines its
transpose :math:`A^T`. Using these, we invert the ADRT applying the
conjugate gradient method to the normal equations:
:math:`A^{T}Ax=A^{T}b`.

Here we use SciPy's implementation in particular, provided in
:func:`scipy.sparse.linalg.cg`. To do this we define
``AdrtNormalOperator`` an instance of
:class:`scipy.sparse.linalg.LinearOperator` for the operation
:math:`A^{T}A` and then use this in a function ``cgiadrt`` which
performs the actual inversion operation using conjugate gradients.

.. plot::
   :context: close-figs
   :nofigs:

   class AdrtNormalOperator(LinearOperator):
       def __init__(self, img_size, dtype=None):
           super().__init__(dtype=dtype, shape=(img_size**2, img_size**2))
           self._img_size = img_size

       def _matvec(self, x):
           sqmat = x.reshape((self._img_size, self._img_size))
           ret = adrt.utils.truncate(adrt.bdrt(adrt.adrt(sqmat)))
           return np.mean(ret, axis=0).ravel()

       def _adjoint(self):
           return self


   def cgiadrt(b, **kwargs):
       img_size = b.shape[-1]
       linop = AdrtNormalOperator(img_size=img_size, dtype=b.dtype)
       tb = np.mean(adrt.utils.truncate(adrt.bdrt(b)), axis=0).ravel()
       x, info = cg(linop, tb, x0=tb, **kwargs)
       if info != 0:
           raise ValueError(f"Convergence failed (cg status {info})")
       return x.reshape((img_size, img_size))
