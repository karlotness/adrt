ADRT: Approximate Discrete Radon Transform
==========================================

This is the documentation for the `adrt` package, a Python library
providing an approximate discrete Radon transform operating on `NumPy
<https://numpy.org/>`__ arrays.

This library includes routines for computing the forward ADRT, a
back-projection operation, several inverses, and related utility
functions.

To get started, follow the :doc:`installation instructions <install>`
then consult the :doc:`quickstart guide <quickstart>` for an
introduction to the use of the library.

Our :doc:`examples <examples>` demonstrate applying these routines to
various sample problems and include recipes for developing new
functionality using this package. Detailed information on each
function is included in the :doc:`API reference <reference>`.

The library source code is available in our `repository on GitHub
<https://github.com/karlotness/adrt>`__.

Citation
--------

If you use this software in your research, please cite our associated
`JOSS paper <https://doi.org/10.21105/joss.05083>`__.

.. code-block:: bibtex

   @article{adrt,
     title={adrt: approximate discrete {R}adon transform for {P}ython},
     author={Karl Otness and Donsub Rim},
     journal={Journal of Open Source Software},
     publisher={The Open Journal},
     year=2023,
     doi={10.21105/joss.05083},
     url={https://doi.org/10.21105/joss.05083},
     volume=8,
     number=83,
     pages=5083,
   }

.. toctree::
   :maxdepth: 1
   :caption: Contents

   install
   quickstart
   examples
   reference
   develop
   changes
   license


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
