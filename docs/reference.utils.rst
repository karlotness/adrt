Utilities
=========

.. automodule:: adrt.utils

Output Stitching
----------------

These routines manipulate ADRT outputs by slicing, rotating, and
aligning them to make the results suitable for further processing or
visualization.

.. autofunction:: truncate

.. autofunction:: stitch_adrt

.. autofunction:: unstitch_adrt

Interpolation
-------------

We provide a simple routine which interpolates the unevenly spaced
ADRT angles into evenly spaced angles as in a more conventional
sinogram.

.. autofunction:: interp_to_cart

Coordinate Information
----------------------

These functions provide information on the angle and offset
coordinates which are represented in an ADRT output, or the closest
correspondence between regularly-spaced Cartesian coordinates to the
unevenly spaced ADRT coordinates.

.. autofunction:: coord_adrt

.. autofunction:: coord_cart_to_adrt
