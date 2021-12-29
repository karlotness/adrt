.. _cginverse:

Iterative Inverse
=================

This example presents a method for inverting the forward ADRT which
takes a different approach to the inverse implemented in
:func:`adrt.iadrt`. The inverse here uses an iterative solver, in
particular the SciPy's :func:`scipy.sparse.linalg.cg` routine, but
another implementation could be used instead, if desired.

To begin our implementation we import the modules we need, including
values from :mod:`scipy.sparse.linalg`. ::

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
           super().__init__(dtype=dtype, shape=(img_size ** 2, img_size ** 2))
           self._img_size = img_size

       def _matmat(self, x):
           # Use batch dimensions to handle columns of matrix x
           n_batch = x.shape[-1]
           batch_img = np.moveaxis(x, -1, 0).reshape(
               (n_batch, self._img_size, self._img_size)
           )
           ret = adrt.utils.truncate(adrt.bdrt(adrt.adrt(batch_img))).mean(axis=1)
           return np.moveaxis(ret, 0, -1).reshape((self._img_size ** 2, n_batch))

       def _adjoint(self):
           return self


   def cgiadrt(b, **kwargs):
       img_size = b.shape[-1]
       linop = AdrtNormalOperator(img_size=img_size, dtype=b.dtype)
       tb = adrt.utils.truncate(adrt.bdrt(b)).mean(axis=0).ravel()
       x, info = cg(linop, tb, x0=tb, **kwargs)
       if info != 0:
           raise ValueError(f"Convergence failed (cg status {info})")
       return x.reshape((img_size, img_size))

We'll use the same starting image as in the :ref:`quickstart`, but we
will apply a small amount of normal noise to its adrt to illustrate
the difference in behavior between the iterative inverse here and
:func:`adrt.iadrt`.

.. plot::
   :context: close-figs
   :align: center

   # Generate input image
   n = 16
   xs = np.linspace(-1, 1, n)
   x, y = np.meshgrid(xs, xs)
   img = 0.5 * ((np.abs(x - 0.25) + np.abs(y)) < 0.7).astype(np.float32)
   img[:, 3] = 1
   img[1, :] = 1

   # Compute ADRT and add noise
   img_plain_adrt = adrt.adrt(img)
   noise_mask = np.random.default_rng(seed=0).normal(scale=1e-4, size=img_plain_adrt.shape)
   img_noise_adrt = img_plain_adrt + noise_mask

   # Plot noisy ADRT
   vmin = np.min(img_noise_adrt)
   vmax = np.max(img_noise_adrt)
   fig, axs = plt.subplots(1, 4, sharey=True)
   for i, ax in enumerate(axs.ravel()):
       im_plot = ax.imshow(img_noise_adrt[i], vmin=vmin, vmax=vmax)
   fig.tight_layout()
   fig.colorbar(im_plot, ax=axs, orientation="horizontal", pad=0.1)


If you compare this against the ADRT in :ref:`quickstart`, you should
see that the differences are visually imperceptible. However, the two
inverses produce very different results.

.. plot::
   :context: close-figs
   :align: center

   iadrt_inv = adrt.utils.truncate(adrt.iadrt(img_noise_adrt)).mean(axis=0)
   cg_inv = cgiadrt(img_noise_adrt)

   fig, axs = plt.subplots(1, 3, sharey=True)
   plot_elements = [(img, "Original"), (cg_inv, "CG Inverse"), (iadrt_inv, "iadrt Inverse")]
   for ax, (data, title) in zip(axs.ravel(), plot_elements):
       im_plot = ax.imshow(data)
       fig.colorbar(im_plot, ax=ax, orientation="horizontal", pad=0.08)
       ax.set_title(title)
   fig.tight_layout()

The inverse provided by :func:`adrt.iadrt` is an exact inverse to the
forward ADRT, but it is very sensitive to noise in its input. It is
therefore not suitable for cases where the forward ADRT was not
exactly applied, or where noise may be present. In such cases, a
different approach such as the ``cgiadrt`` illustrated here may be
more suitable.

Multiple Noise Levels
---------------------

We repeat the above demonstration of the ``cgiadrt`` iterative inverse
for several noise levels. For each example a new noise mask is drawn
from a normal distribution :math:`\mathcal{N}(0, \sigma I)`.

.. plot::
   :context: close-figs
   :align: center

   rng = np.random.default_rng(seed=0)
   fig, axs = plt.subplots(2, 2, sharey=True)
   fig.suptitle("CG Inverses at Several Noise Levels")
   for scale, ax in zip([1e-2, 1e-1, 1, 10], axs.ravel()):
       noise = rng.normal(scale=scale, size=img_plain_adrt.shape)
       cg_inv = cgiadrt(img_plain_adrt + noise)
       im_plot = ax.imshow(cg_inv)
       fig.colorbar(im_plot, ax=ax)
       ax.set_title(f"$\\sigma = {scale}$")
   fig.tight_layout()

The results produced by ``cgiadrt`` remain relatively clean even at
noise with scales much larger than those used for the comparison with
:func:`adrt.iadrt`. While exact, :func:`adrt.iadrt`, is unstable and
so an iterative approach such as the one demonstrated here may be
advantageous for certain applications and can be assembled with the
help of routines in this package.
