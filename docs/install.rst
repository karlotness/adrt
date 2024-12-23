.. highlight:: shell-session

Installation
============

PyPI
----

This package is available as "`adrt
<https://pypi.org/project/adrt/>`__" on PyPI. You can install it using
pip::

  $ python -m pip install adrt

or on Windows:

.. code-block:: ps1con

   PS> py -m pip install adrt

For more details on pip, consult its `user guide
<https://pip.pypa.io/en/stable/user_guide/>`__. Before installing it
is best to ensure your package manager is up to date (`upgrade pip
<https://pip.pypa.io/en/stable/installation/#upgrading-pip>`__).

Conda-Forge
-----------

This package can also be installed into a `Conda environment
<https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html>`__
from the `conda-forge <https://conda-forge.org/>`__ channel::

  $ conda install -c conda-forge adrt

For more information, consult the `project documentation
<https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge>`__.
The packages distributed by conda-forge for Linux and macOS have
OpenMP multithreading support enabled (see further discussion
:ref:`below <openmp-build>`).

Source
------

The package can also be built from source code, which is available
from our `GitHub repository <https://github.com/karlotness/adrt>`__.
This makes it possible to customize the compiler flags used when
building the native extensions. In particular, OpenMP support can be
enabled as discussed :ref:`below <openmp-build>`.

The code requires C++20 support to build and makes use of a few
compiler built-in functions for performance. We require recent GCC,
Clang, or MSVC.

To install the package from source with default compiler flags,
navigate to the directory containing your copy of the source code and
run (note the trailing ``.``)::

  $ python -m pip install .

Running Tests
~~~~~~~~~~~~~

Once you have an installed version of the package, the test suite can
help confirm whether the routines are operating correctly. We use
`pytest <https://pytest.org/>`__ as a test runner. Once the package is
installed, navigate to the root directory of the repository and run::

  $ python -m pip install -r tests/requirements.txt
  $ python -m pytest

However, if you are trying to modify the package or make a
contribution see the information under :doc:`develop`.

.. _openmp-build:

OpenMP Support
~~~~~~~~~~~~~~

Several routines in this package support parallelization with `OpenMP
<https://www.openmp.org/>`__. Using these features requires both
compiler support and an OpenMP runtime library. Most compilers already
support OpenMP and most Linux platforms will likely have an OpenMP
runtime installed.

To enable OpenMP you must pass the necessary flags to your C++
compiler via the ``CXXFLAGS`` environment variable. For GCC the correct
flag is ``-fopenmp``. In this case, run pip as::

  $ CXXFLAGS="-O3 -fopenmp" CPPFLAGS="-DNDEBUG" python -m pip install .

This will install an OpenMP-enabled copy of the package. Other
compilers will require different flags to be passed through
``CXXFLAGS``, ``CFLAGS``, and ``LDFLAGS``. In the above example
``CPPFLAGS`` is set to suppress `debugging assertions
<https://en.cppreference.com/w/cpp/error/assert>`__ using the
preprocessor. Other compilers may require different flags to ensure
the same preprocessor macro is set.

After building, you can verify that your installed copy supports
OpenMP with :func:`adrt.core.threading_enabled`.
