Computerized Tomography
=======================

Using the routines in :py:mod:`adrt`, we can solve the inverse problem arising
from Computerized Tomography (CT). Here we will be using SciPy's
:func:`scipy.sparse.linalg.cg` routine as illustrated in the :ref:`Iterative
Inverse Section <inverse page>`.

We first import requisite modules, and define the ``AdrtNormalOperator`` and the
function ``iadrt_cg``. However, here we modify ``AdrtNormalOperator`` by adding
a multiple of the identity term. This modifies our operator so that the CG
inverse computes the ridge regression problem, given by
:math:`(A^{T}A + \lambda A)x = A^{T}b`. We name the new operator ``AdrtRidgeOperator`` below.

.. plot:: code/iadrt_cg.py
   :context: reset
   :include-source: false
   :nofigs:

.. plot::
   :context: close-figs
   :align: center

   class AdrtRidgeOperator(AdrtNormalOperator):
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
coordinates, using the routine provided in :func:`skimage.transform.radon()`

.. plot::
   :context: close-figs
   :align: center

   from skimage.transform import radon

   th_array, s_array = adrt.utils.coord_adrt_to_cart(n)
   th_array = np.broadcast_to(th_array, (2 * n - 1, 4 * n))

   th_array1 = th_array[0, :]
   theta = 90.0 + np.rad2deg(th_array1)
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

Then we define a function ``cart_to_adrt`` that interpolates the sampled forward
data into the ADRT data format.

.. plot::
   :context: close-figs
   :align: center

   def cart_to_adrt(th_array, s_array, sinogram):

      n = th_array.shape[1] // 4
      m = sinogram.shape[0]

      nq = 4
      adrt_data = np.zeros((nq, 2*n-1, n))
      theta = th_array[0, :]

      theta_q = np.abs(theta) - np.abs(theta - np.pi/4) - np.abs(theta + np.pi/4) + np.pi/2

      t_coords, step = np.linspace(-0.5, 0.5, m, retstep=True, endpoint=True)

      for q in range(nq):
         for i in range(n):
            if q % 2 == 0:
               j = q*n + i
            else:
               j = (q+1)*n - i - 1
            s_coords = s_array[:, j]
            factor = np.cos(theta_q[j])
            vals = np.interp(s_coords,
                             t_coords - step*j/(4*n),  # offset correction
                             sinogram[:, j],
                             left=0.0, right=0.0)

            adrt_data[q, :, i] = vals*factor

      return adrt_data


Inversion result
----------------

Now, we compute the inverse problem by solving the ridge regression problem.

.. plot::
   :context: close-figs
   :align: center

   adrt_data = cart_to_adrt(th_array, s_array, sinogram)
   cg_inv = iadrt_cg(adrt_data, op_cls=AdrtRidgeOperator)

   # Display inversion result
   plt.imshow(cg_inv, cmap="bone")
   plt.colorbar()
   plt.tight_layout()

The inversion result, together with a slice plot in the horizontal direction is
displayed below.

.. plot::
   :context: close-figs
   :align: center

   fig, axs = plt.subplots(nrows=2,
                           ncols=2,
                           gridspec_kw={'height_ratios' : (3,1)})

   ax = axs[0, 0]
   ax.imshow(phantom, cmap='Greys_r', extent=(0, 1, 0, 1))
   ax.hlines(0.6, 0, 1, 'b')
   ax.set_title('original')

   ax = axs[0, 1]
   ax.imshow(cg_inv, cmap='Greys_r', extent=(0, 1, 0, 1))
   ax.hlines(0.6, 0, 1, 'b')
   ax.set_title('CG inverse (ridge)')

   ax = axs[1, 0]
   x = np.linspace(0.0, 1.0, n)
   ax.plot(x, phantom[n // 5 * 2, :], 'b')
   ax.set_ylim([-0.1, 1.1])

   ax = axs[1, 1]
   x = np.linspace(0.0, 1.0, n)
   ax.plot(x, cg_inv[n // 5 * 2, :], 'b')
   ax.set_ylim([-0.1, 1.1])
