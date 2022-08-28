Sinograms of an Image
=====================

As a more involved example we can consider the Shepp-Logan phantom
(:download:`data file <data/shepp-logan.npz>`).

We make use of this package as well as a few other fundamental
libraries. ::

   import numpy as np
   from matplotlib import pyplot as plt
   import adrt

First, we load and preview the data.

.. plot::
   :context: reset
   :align: center

   phantom = np.load("data/shepp-logan.npz")["phantom"]
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

   img_cart = adrt.utils.interp_to_cart(adrt_result)

   plt.imshow(img_cart, aspect="auto")
   plt.colorbar()
   plt.ylabel("$t$")
   plt.xlabel("$\\theta$")
