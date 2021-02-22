.. highlight:: bash

.. _installation:

Installation
============

PyPI
----

This package is available as "`adrt
<https://pypi.org/project/adrt/>`__" on PyPI. You can install it using
pip::

  python -m pip install adrt
  # On Windows:
  py -m pip install adrt

For more details on pip, consult its `user guide
<https://pip.pypa.io/en/stable/user_guide/>`__. Before installing it
is best to ensure your package manager is up to date (`upgrade pip
<https://pip.pypa.io/en/stable/installing/#upgrading-pip>`__).

Source
------

The package can also be built from source code, which is available
from our `GitHub repository <https://github.com/karlotness/adrt>`__.
This makes it possible to customize the compiler flags used when
building the native extensions. In particular, OpenMP support can
enabled as discussed :ref:`below <openmp-build>`.

To install the package from source with default compiler flags run
(note the trailing ``.``)::

  python -m pip install .

Running Tests
~~~~~~~~~~~~~

Once you have an installed version of the package, the test suite can
help confirm whether the routines are operating correctly. We use
`pytest <https://pytest.org/>`__ as a test runner. Once both pytest
and the package are installed, navigate to the root directory of the
repository and run::

  pytest

Alternatively, you can use the built-in :mod:`unittest` module to
run the tests::

  python -m unittest tests/*.py

.. _openmp-build:

OpenMP Support
~~~~~~~~~~~~~~

Several routines in this package support parallelization with `OpenMP
<https://www.openmp.org/>`__. Using these features requires both
compiler support and an OpenMP runtime library. Most compilers already
support OpenMP and most Linux platforms will likely have an OpenMP
runtime installed.

To enable OpenMP you must pass the necessary flags to your C++
compiler via the ``CFLAGS`` environment variable. For GCC the correct
flag is ``-fopenmp``. In this case, run pip as::

  CFLAGS="-fopenmp" python -m pip install .

This will install an OpenMP-enabled copy of the package. Other
compilers will require different flags to be passed through ``CFLAGS``
and ``LDFLAGS``.
