Examples
========

Below, we show some examples of basic usage of ``adrt``. In each of
the examples below, we will make use of a few basic Python libraries:

.. code::

   import numpy as np
   from matplotlib import pyplot as plt
   import adrt


Simple ADRT
-----------

.. plot::
   :context: reset
   :align: center

   # Generate image
   n = 16
   xs = np.linspace(-1, 1, n)
   x, y = np.meshgrid(xs, xs)
   img = ((np.abs(x - 0.25) + np.abs(y)) < 0.7).astype(np.float32)

   # Display
   plt.imshow(img)
   plt.colorbar()
   plt.tight_layout()

The result can be computed by :func:`adrt.adrt`.

.. plot::
   :context: close-figs
   :align: center

   adrt_result = adrt.adrt(img)

   # Display result
   fig, axs = plt.subplots(1, 4)
   for i, ax in enumerate(axs.ravel()):
       im_plot = ax.imshow(adrt_result[i], vmin=0, vmax=np.max(adrt_result))
   plt.tight_layout()
   fig.colorbar(im_plot, ax=axs, orientation="horizontal")

This result can be stitched together using :func:`adrt.utils.stitch_adrt`.

.. plot::
   :context: close-figs
   :align: center

   adrt_stitched = adrt.utils.stitch_adrt(adrt_result)

   # Display result
   plt.imshow(adrt_stitched)
   plt.colorbar()
   for i in range(1, 4):
       plt.axvline(n * i - 0.5, color="white", linestyle="--")
   plt.tight_layout()

Sinograms of an image
---------------------

As a more involved example we can consider the Shepp-Logan phantom
(:download:`data file <data/shepp-logan.npz>`).

.. plot::
   :context: reset
   :align: center

   phantom = np.load("data/shepp-logan.npz")["phantom"].astype(np.float32)
   n = phantom.shape[0]

   # Display the image
   plt.imshow(phantom, cmap="bone")
   plt.colorbar()
   plt.tight_layout()

We can start by computing the adrt of this image

.. plot::
   :context: close-figs
   :align: center

   adrt_result = adrt.adrt(phantom)
   adrt_stitched = adrt.utils.stitch_adrt(adrt_result)

   plt.imshow(adrt_stitched)
   plt.colorbar()
   for i in range(1, 4):
       plt.axvline(n * i - 0.5, color="white", linestyle="--")
   plt.tight_layout()

These can be interpolated to a Cartesian grid with
:func:`adrt.utils.interp_to_cart`.

.. plot::
   :context: close-figs
   :align: center

   theta, s, img_cart = adrt.utils.interp_to_cart(adrt_result)

   plt.imshow(img_cart, aspect="auto")
   plt.colorbar()
   plt.ylabel("$t$")
   plt.xlabel("$\\theta$")
