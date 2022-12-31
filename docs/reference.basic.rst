Basic Routines
==============

.. automodule:: adrt

Forward Transforms
------------------

These functions implement the fundamental operations of this package:
the forward approximate discrete Radon transform (ADRT) and the
associated backprojection operation.

.. autofunction:: adrt

.. autofunction:: bdrt

.. _basic-inverse-transforms:

Inverse Transforms
------------------

We provide two possible inverses to :func:`adrt`. The first is an
exact (although ill-conditioned) inverse, while the second implements
an approximate inverse by the full multigrid method.

.. autofunction:: iadrt

.. autofunction:: iadrt_fmg

.. [#brady98] Martin L. Brady, *Discrete Radon transform has an exact,
   fast inverse and generalizes to operations other than sums along
   lines*, Proceedings of the National Academy of Sciences, 103
   `doi:10.1137/S0097539793256673
   <https://doi.org/10.1137/S0097539793256673>`_.
.. [#press06] William H. Press, *A Fast Discrete Approximation
   Algorithm for the Radon Transform Related Databases*, SIAM Journal
   on Computing, 27 `doi:10.1073/pnas.0609228103
   <https://doi.org/10.1073/pnas.0609228103>`_.
.. [#rim20] Donsub Rim, *Exact and fast inversion of the approximate
   discrete Radon transform from partial data*, Applied Mathematics
   Letters, 102 `doi:10.1016/j.aml.2019.106159
   <https://doi.org/10.1016/j.aml.2019.106159>`_.
