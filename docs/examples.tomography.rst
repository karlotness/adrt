Computerized Tomography
=======================

Using the routines in :py:mod:`adrt`, we can solve the inverse problem arising
from Computerized Tomography (CT). Here we will be using SciPy's
:func:`scipy.sparse.linalg.cg` routine as illustrated in the :ref:`Iterative
Inverse Section <inverse page>`.

We first import requisite modules, and define the ``ADRTNormalOperator`` and the
function ``iadrt_cg``. However, here we modify ``ADRTNormalOperator`` by adding
a multiple of the identity term. This modifies our operator so that the CG
inverse computes the ridge regression problem, given by
:math:`(A^{T}A + \lambda I)x = A^{T}b`. We name the new operator ``ADRTRidgeOperator`` below.

.. plot:: code/iadrt_cg.py
   :context: reset
   :include-source: false
   :nofigs:

.. plot::
   :context: close-figs
   :align: center

   # Using ADRTNormalOperator from the Iterative Inverse example
   class ADRTRidgeOperator(ADRTNormalOperator):
      def __init__(self, img_size, dtype=None, ridge_param=40.0):
         super().__init__(dtype=dtype, img_size=img_size)
         self._ridge_param = ridge_param

      def _matmat(self, x):
         # Use batch dimensions to handle columns of matrix x
         return super()._matmat(x) + self._ridge_param * x

Forward Data
-------------

We first obtain the forward measurements of the CT problem, by computing the
Radon transform of the Shepp-Logan phantom.

.. plot::
   :context: close-figs
   :align: center

   phantom = np.load("data/shepp-logan.npz")["phantom"]
   n = phantom.shape[0]

We sample the CT data on a uniform Cartesian grid in the :math:`(\theta, t)`
coordinates, using the routine provided in :func:`skimage.transform.radon`:

.. plot::
   :context: close-figs
   :align: center

   from skimage.transform import radon

   th_array1 = np.unique(adrt.utils.coord_adrt(n).angle)
   theta = 90.0 + np.rad2deg(th_array1.squeeze())
   sinogram = radon(phantom, theta=theta)

The sinogram is plotted below. Although this sinogram is similar to that which
appeared in :ref:`sinogram computation section <adrt shepplogan page>` the grid
in the :math:`(\theta, t)` coordinates used is different, and they used two
different discrete approximations of the continuous transform.

.. plot::
   :context: close-figs
   :align: center

   plt.imshow(sinogram, aspect="auto")
   plt.colorbar()

Then we use :class:`scipy.interpolate.RectBivariateSpline` to
interpolate the sampled forward data at the ADRT coordinates, forming
the ADRT data.

.. plot::
   :context: close-figs
   :align: center

   from scipy import interpolate

   t_array = np.linspace(-0.5, 0.5, n)
   spline = interpolate.RectBivariateSpline(t_array, th_array1, sinogram)
   s_array, th_array = adrt.utils.coord_adrt(n)
   adrt_data = spline(s_array, th_array, grid=False)


Inversion result
----------------

Now, we compute the inverse problem by solving the ridge regression
problem. We also show the inverse computed with :func:`adrt.iadrt_fmg`
included in the package.

.. plot::
   :context: close-figs
   :align: center

   # Using iadrt_cg from the Iterative Inverse example
   cg_inv = iadrt_cg(adrt_data, op_cls=ADRTRidgeOperator)
   fmg_inv = adrt.iadrt_fmg(adrt_data)

   # Display inversion result
   fig, axs = plt.subplots(1, 2, sharey=True)
   for ax, data, title in zip(
       axs.ravel(),
       [cg_inv, fmg_inv],
       ["CG Inverse", "FMG Inverse"],
   ):
       im_plot = ax.imshow(data, cmap="bone", extent=(0, 1, 0, 1))
       fig.colorbar(im_plot, ax=ax, orientation="horizontal", pad=0.08)
       ax.set_title(title)
   fig.tight_layout()

The inversion result, together with a slice plot in the horizontal direction is
displayed below.

.. plot::
   :context: close-figs
   :align: center

   fig, axs = plt.subplots(
       2, 3, sharex=True, sharey="row",
   )
   vmin = min(map(np.min, [phantom, cg_inv, fmg_inv]))
   vmax = max(map(np.max, [phantom, cg_inv, fmg_inv]))
   plot_row = n // 5 * 2
   plot_x = np.linspace(0.0, 1.0, n)

   for ax, data, title in zip(
       axs.T,
       [phantom, cg_inv, fmg_inv],
       ["Original", "CG Ridge Inverse", "FMG Inverse"],
   ):
       im_ax = ax[0]
       plot_ax = ax[1]
       im_ax.imshow(
           data,
           cmap="bone",
           extent=(0, 1, 0, 1),
           vmin=vmin,
           vmax=vmax,
       )
       im_ax.axhline(0.6, color="C0")
       im_ax.set_title(title)
       plot_ax.plot(plot_x, data[plot_row, :], "C0")
       plot_ax.grid(True)
   fig.tight_layout()
