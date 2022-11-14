Wave Equation
=============

The Radon transform allows one to solve the multi-dimensional wave equation
:math:`\partial_t^2 u = \Delta u` by transforming the problem into a family of
1D wave equations in the Radon domain, due to its intertwining property.
[#natterer01]_ As a result, one can solve the 1D wave equations in the Radon
domain then transform back into the physical variables to obtain the solution.
Note that this solution is essentially identical to the Lax-Philips translation
representation. [#lax67]_

For this solution, we will need to invert the Radon transform: we will again be
using SciPy's :func:`scipy.sparse.linalg.cg` routine as illustrated in the
:ref:`Iterative Inverse Section <inverse page>` and make use of the function
``iadrt_cg`` from that example.

.. plot:: code/iadrt_cg.py
   :context: reset
   :include-source: false
   :nofigs:

We choose a superposition of two cosine peaks as the initial condition and form
its discretization.

.. plot::
   :context: close-figs
   :align: center

   n = 2**9
   xx = np.linspace(0.0, 1.0, n)
   X, Y = np.meshgrid(xx, xx)

   alph1 = 16.0
   alph2 = 8.0

   x1, y1 = 0.6, 0.65
   x2, y2 = 0.4, 0.35

   R1 = np.sqrt((X - x1)**2 + (Y - y1)**2)
   R2 = np.sqrt((X - x2)**2 + (Y - y2)**2)

   init = 0.5*(np.cos(np.pi*alph1*R1) + 1.0)*(R1 < 1.0/alph1) \
        + 0.5*(np.cos(np.pi*alph2*R2) + 1.0)*(R2 < 1.0/alph2)


We then approximate the Radon transform of the initial condition using :func:`adrt.adrt`.

.. plot::
   :context: close-figs
   :align: center

   init_adrt = adrt.adrt(init)

For each angular slice, we translate the initial condition following the
d'Alembert formula.

.. plot::
   :context: close-figs
   :align: center

   sol_adrt = np.zeros(init_adrt.shape)

   m = init_adrt.shape[1]

   # Eulerian grid
   yy = np.linspace(-1.0, 1.0, m)

   time = 0.20
   for q in range(4):
       for i in range(n):
           th = np.arctan(i/(n-1))

           # Construct Lagrangian grid then interpolate
           xx = yy + time/np.cos(th)
           sol_adrt[q, :, i] += 0.5*np.interp(yy, xx, init_adrt[q, :, i])
           xx = yy - time/np.cos(th)
           sol_adrt[q, :, i] += 0.5*np.interp(yy, xx, init_adrt[q, :, i])


Finally, we plot the solution.

.. plot::
   :context: close-figs
   :align: center


   plt.plot(init_adrt[0, :, m//2], label='initial ADRT slice')
   plt.plot(sol_adrt[0, :, m//2], label='solution ADRT slice')
   plt.legend()

Finally, we invert the ADRT.

.. plot::
   :context: close-figs
   :align: center

   # Using iadrt_cg from the Iterative Inverse example
   sol = iadrt_cg(sol_adrt)

We plot the solution, and also show the Cartesian view of the ADRT data.

.. plot::
   :context: close-figs
   :align: center

   fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))

   cart_extent = 0.5*np.array([-np.sqrt(2), np.sqrt(2), -np.pi, np.pi])

   ax = axs[0, 1]
   im = ax.imshow(adrt.utils.interp_to_cart(init_adrt), aspect='auto', extent=cart_extent)
   plt.colorbar(im, ax=ax)
   ax.set_xlabel('$\\theta$')
   ax.set_ylabel('$t$')

   ax = axs[1, 1]
   im = ax.imshow(adrt.utils.interp_to_cart(sol_adrt), aspect='auto', extent=cart_extent)
   plt.colorbar(im, ax=ax)
   ax.set_xlabel('$x$')
   ax.set_ylabel('$y$')
   ax.set_xlabel('$\\theta$')
   ax.set_ylabel('$t$')

   ax = axs[0, 0]
   im = ax.imshow(init, extent=(0, 1, 0, 1))
   ax.set_title('time = {:1.1f}'.format(0))
   plt.colorbar(im, ax=ax)
   ax.set_aspect(1)
   ax.set_xlabel('$x$')
   ax.set_ylabel('$y$')

   ax = axs[1, 0]
   ax.set_title('time = {:1.1f}'.format(time))
   im = ax.imshow(sol, extent=(0, 1, 0, 1))
   plt.colorbar(im, ax=ax)
   ax.set_aspect(1)
   ax.set_xlabel('$x$')
   ax.set_ylabel('$y$')

   fig.tight_layout()
   fig.show()


.. [#natterer01] Frank Natterer, *The Mathematics of Computerized
                 Tomography*, SIAM 2001. `doi:10.1137/1.9780898719284
                 <https://doi.org/10.1137/1.9780898719284>`_.
.. [#lax67] Peter D. Lax, and Ralph S. Phillips, *Scattering Theory*,
            Bulletin of the American Mathematical Society 1964.
            `doi:10.1090/S0002-9904-1964-11051-X
            <https://doi.org/10.1090/S0002-9904-1964-11051-X>`_.
