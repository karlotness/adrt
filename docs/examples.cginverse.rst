.. _cginverse:

Iterative Inverse
=================

This example makes use of facilities from SciPy. In particular, its
:func:`scipy.sparse.linalg.cg` routine. ::

   import numpy as np
   from scipy.sparse.linalg import LinearOperator, cg
   from matplotlib import pyplot as plt
   import adrt

   def cgiadrt(da, **kwargs):
       def _matmul(x):
           n2 = x.shape[0]
           n = int(np.round(np.sqrt(n2)))
           x2 = x.reshape((n, n))

           da = adrt(x2)
           ba = bdrt(da)
           ta = truncate(ba)
           x_out = np.mean(ta, axis=0).flatten()

           return x_out

       n = da.shape[-1]

       ba = bdrt(da)
       ta = truncate(ba)
       ta = np.mean(ta, axis=0)

       if "x0" not in kwargs.keys():
           kwargs["x0"] = ta.flatten()

       ta = ta.flatten()
       linop = LinearOperator((n ** 2, n ** 2), matvec=_matmul, dtype=da.dtype)
       out = cg(linop, ta, **kwargs)
       ia_out = out[0].reshape(n, n)
       cg_out = out[1:]

       return ia_out, cg_out
