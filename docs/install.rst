.. highlight:: bash

Installation
============

Source
------

The package can also be built from source code, which is available
from our `GitHub repository <https://github.com/karlotness/adrt>`_.
This makes it possible to customize the compiler flags used when
building the native extensions. In particular, OpenMP support can
enabled as discussed :ref:`below <openmp-build>`.

To install the package from source with default compiler flags run::

  pip install .

Running Tests
~~~~~~~~~~~~~

Once you have an installed version of the package, the test suite can
help confirm whether the routines are operating correctly. We use
:doc:`PyTest <pytest:index>` as a test runner. Once PyTest is
installed, navigate to the root source code directory and run::

  pytest

.. _openmp-build:

OpenMP Support
~~~~~~~~~~~~~~

Several routines in this package support parallelization with
`OpenMP`_. Using these features requires both compiler support and an
OpenMP runtime library. Most compilers already support OpenMP and most
Linux platforms will likely have an OpenMP runtime installed.

To enable OpenMP you must pass the necessary flags to your C++
compiler via the ``CFLAGS`` environment variable. For GCC the correct
flag is ``-fopenmp``. In this case, run pip as::

  CFLAGS="-fopenmp" pip install .

This will install an OpenMP-enabled copy of the package. Other
compilers will require different flags to be passed through ``CFLAGS``
and ``LDFLAGS``.

.. _OpenMP: https://www.openmp.org/
