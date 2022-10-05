Sinograms of an Image
=====================

The discretized Radon transforms are often used to approximate sinograms of a
known image. :py:mod:`adrt` provides utilities that map the ADRT output which is
in the ADRT coordinates :math:`(h, s)` to a sinogram which is in the Cartesian
coordinates :math:`(\theta, t)`.

We make use of this package as well as a few other fundamental
libraries. ::

   import numpy as np
   from matplotlib import pyplot as plt
   import adrt

We will illustrate the computation of sinograms with a couple examples. The simple geometry relating the two coordinates :math:`(h, s)`  and :math:`(\theta, t)` is given in a :ref:`diagram below<adrt_to_cart diagram>`.

Gaussian humps
--------------

We compute an image that is a sum of two Gaussian humps.

.. plot::
   :context: close-figs
   :align: center

   n = 2**9
   x1 = np.linspace(0.0, 1.0, n)
   X, Y = np.meshgrid(x1, x1)
   s1 = 200
   s2 = 100

   gaussians = np.exp( -s1*(X - 0.75)**2 - s1*(Y - 0.3)**2) \
             + np.exp( -s2*(X - 0.25)**2 - s2*(Y - 0.8)**2)

   plt.imshow(gaussians, extent=(0, 1, 0, 1))
   plt.colorbar()

We compute the ADRT of this image and plot the image.

.. plot::
   :context: close-figs
   :align: center

   adrt_result = adrt.adrt(gaussians)
   adrt_stitched = adrt.utils.stitch_adrt(adrt_result)

   plt.imshow(adrt_stitched)
   plt.colorbar()
   for i in range(1, 4):
       plt.axvline(n * i - 0.5, color="white", linestyle="--")
   plt.ylabel('$h$')
   plt.xlabel('$s$')
   plt.tight_layout()

From the ADRT data we compute the sinogram by using the function
:func:`adrt.utils.interp_to_cart`. Each isotropic Gaussian hump correspond to
a sinusoidal curve of commensurate width in the sinogram.

.. plot::
   :context: close-figs
   :align: center

   img_cart = adrt.utils.interp_to_cart(adrt_result)
   img_extent = np.array([ -90, 90, -1/np.sqrt(2), 1/np.sqrt(2)])

   plt.imshow(img_cart, aspect="auto", extent=img_extent)
   plt.colorbar()
   plt.ylabel("$t$")
   plt.xlabel("$\\theta$")


.. _adrt shepplogan page:

Shepp-Logan Phantom
-------------------

As a more involved example we can consider the Shepp-Logan phantom
(:download:`data file <data/shepp-logan.npz>`).

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
   plt.ylabel('$h$')
   plt.xlabel('$s$')
   plt.tight_layout()

These can be interpolated to a Cartesian grid with
:func:`adrt.utils.interp_to_cart`.

.. plot::
   :context: close-figs
   :align: center

   img_cart = adrt.utils.interp_to_cart(adrt_result)
   img_extent = np.array([ -90, 90, -1/np.sqrt(2), 1/np.sqrt(2)])

   plt.imshow(img_cart, aspect="auto", extent=img_extent)
   plt.colorbar()
   plt.ylabel("$t$")
   plt.xlabel("$\\theta$")

.. _adrt_to_cart diagram:

The coordinate transform
------------------------

The coordinates :math:`(h, s)` and :math:`(\theta, t)` are related by a simple
geometric relation, depicted in the following diagram. This is shown for
quadrant 3, the transform for the other quadrants are derived by flipping and
transposing the image.


.. plot::
   :context: close-figs
   :align: center


   fig, ax = plt.subplots()
   
   theta, t = (0.2*np.pi, -0.25)
   
   ax.set_aspect(1)
   xoffset, yoffset = (0.075, 0.025)
   
   # add rectangle
   ax.fill(np.array([0, 1, 1, 0, 0]) - 0.5, 
           np.array([0, 0, 1, 1, 0]) - 0.5,
           color = (0, 0, 0, 0.15))
   
   # draw auxiliary line normal to the hyperplane
   z = np.array([-1.0, 1.0])
   ax.plot([0, - t*np.sin(theta)], 
           [0, + t*np.cos(theta)], 'r')
   
   # draw t-coordinate
   ax.annotate('$t$', 
           xy=(-0.5*t*np.sin(theta)+0.02, 0.5*t*np.cos(theta)))
   
   # draw line
   ax.plot(z*np.cos(theta) - t*np.sin(theta), 
           z*np.sin(theta) + t*np.cos(theta), 'k')
   
   # mark origin
   ax.plot(0, 0, 'k.', markersize=10)
   
   # left-intercept
   x0, y0 = (-0.5, -0.5*np.tan(theta) + t/np.cos(theta))
   # right-intercept
   x1, y1 = \
       (0.5, (0.5 + t*np.sin(theta))/np.cos(theta)*np.sin(theta) + t*np.cos(theta))
   
   # mark intercept
   ax.plot(x0, y0, 'k+')
   
   # draw legs
   ax.hlines(y0, -0.5, 0.5, 'k', linestyles='--')
   ax.vlines(0.5, y0, y1, 'k', linestyles='--')
   
   # display s calculation
   ax.annotate('$s = \\arctan(\\theta)$', 
               xy=(0.5 + xoffset, 0.5*(y0 + y1)), 
               color='b')
   ax.vlines(0.5 + 0.5*xoffset, y0, y1, 'b')
   
   ax.annotate('$\\theta$', xy=(-0.5 + 1.5*xoffset, y0 + 0.02))
   ax.annotate('$\\frac{h}{N} = \\frac{t}{\cos(\\theta)} - \\frac{\\tan(\\theta)}{2}$', 
               xy=(-0.5 - 8*xoffset, y0), 
               color='b')
   
   # show h-coordinates for bottom and top edges
   ax.annotate('$h=0$', xy=(-0.5 - 2.5*xoffset, -0.5 - yoffset), alpha=0.5)
   ax.annotate('$h=N$', xy=(-0.5 - 2.5*xoffset,  0.5 - yoffset), alpha=0.5)
   
   ax.hlines(-0.5, -0.5 - xoffset/3, -0.5 + xoffset/3, 'k')
   ax.hlines( 0.5, -0.5 - xoffset/3, -0.5 + xoffset/3, 'k')
   
   ax.spines['top'].set_visible(False)
   ax.spines['right'].set_visible(False)
   ax.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
   ax.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
   ax.set_xlim([-1.2, 1.2])
   ax.set_xlabel('$x$')
   ax.set_ylabel('$y$')
   ax.grid(True)
