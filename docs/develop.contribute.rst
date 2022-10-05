.. highlight:: shell-session

Contributing
============

If you have a minor issue with the software, it is likely easiest to
`open an issue <https://github.com/karlotness/adrt/issues>`__ for the
maintainers to address. For larger changes, it may help to discuss
them first either by opening an issue, or in the `discussion section
<https://github.com/karlotness/adrt/discussions>`__.

For development you will need:

* Python
* A C++ compiler with support for C++11 (recent GCC, Clang, or MSVC)
* `tox <https://tox.wiki/>`__
* Git and a GitHub account (if you want to submit your changes to the
  project)

Once you have these installed you have the tools you need to make
changes to the code or documentation, test your changes, and submit
them through a pull request.

To build build :term:`wheels <pypug:Wheel>` or :term:`sdists
<pypug:Source Distribution (or "sdist")>` to distribute to your users
(either original or modified), consider using `build
<https://pypa-build.readthedocs.io/en/stable/>`__::

  $ python -m build

which will produce a source distribution and installable wheel in the
``dist/`` directory.

.. note::

   If you submit changes to this project, they will be distributed
   under the same :doc:`3-clause BSD license <license>` as the rest of
   the project. By submitting changes you accept this licensing, and
   indicate that you have the ability to license your changes in this
   way including authorization from your employer, if necessary.

Workflow
--------

General steps:

#. Fork and clone the repository
#. Create a new local branch
#. Make your changes
#. Run the tests
#. Push your changes to your fork
#. Submit a pull request

Checkout locally
~~~~~~~~~~~~~~~~

For development you will need a local clone of the repository. If you
only want to build the software for your own use you can simply clone
`our repository <https://github.com/karlotness/adrt>`__ directly. For
more information, see GitHub's `instructions for cloning a repository
<https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository>`__.

If you want to submit your changes, you will need to make a pull
request for which you will need a fork. For information on forking
repositories and making pull requests see GitHub's `introduction to
making contributions
<https://docs.github.com/en/get-started/quickstart/contributing-to-projects>`__.

Make your changes
~~~~~~~~~~~~~~~~~

Once you have a local copy of our repository you can begin to edit the
source code and documentation. If you are making code changes try to
do as much as possible in Python since this simplifies development,
testing, and maintenance. Only use C++ in cases where using Python
would have a *large* performance impact, or would require duplicating
functionality in native code.

See the subsections below for some things to keep in mind as you are
editing.

Python
......

We manage Python formatting and style with a few linting tools. All of
these are run with the tests using tox (see information on running
tests below).

As you modify the code, add any new public functions to
:pycode:`__all__` in their modules and make sure that all public
functions have complete docstrings. Private functions should have
names starting with an underscore and should *not* be listed in
:pycode:`__all__`.

For each new function, consider their call signature. Functions of a
single parameter, or for which parameters do not have descriptive
names should use positional-only parameters. Some function
arguments---particularly boolean flags---should be keyword-only. See
information on :term:`parameters <python:parameter>` from the Python
documentation for more information on parameter types.

If you add any new imports make sure not to add any import cycles.

Don't use Python :pycode:`assert` statements except in tests.

C++
...

Use only features from C++11. The standard library can be used, but
only parts which do not require special C++ runtime support. In
particular this means no functionality which throws exceptions or
requires cleanup after exceptions (only trivially-destructible types).
Avoid compiler-specific extensions, but if you use them provide
fallback versions with the preprocessor.

This produces code which can use templates, but otherwise handles
errors and memory allocation C-style. Make sure to handle error flows
and to clean up your memory and decrement Python refcounts where
necessary.

Any errors at runtime should be reported as Python exceptions from the
interface code in ``adrt_cdefs_py.cpp``. Our convention is that
functions in the ``adrt::_py`` namespace set Python exceptions on
error. Other functions in the ``adrt`` namespace do not and will need
extra handling. No Python APIs are used outside of
``adrt_cdefs_py.cpp``. Only use features from the `Limited API
<https://docs.python.org/3/c-api/stable.html>`__ for the oldest
actively-supported version of Python.

Put non-template and non-inline functions in cpp files, likely
``adrt_cdefs_common.cpp``.

Include assertions with :cppcode:`assert` for conditions which are
required for correctness (*not* error handling). In particular,
assertions on function arguments to check precondition and document
requirements. Use :cppcode:`static_assert` to check type-level
requirements.

Documentation
.............

All public functions must have complete docstrings. For functions
which are implemented in C++ we add docstrings with the Python
wrappers in ``_wrappers.py``. Docstrings are written in `NumPy format
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`__.
Make sure that any new functions appear on the correct API reference
page.

The documentation is generated using Sphinx. All documentation source
code is in the ``docs/`` directory and HTML pages can be generated
using tox::

     $ tox -e docs

After which the main page will be at ``docs/_build/html/index.html``.
If you have made documentation changes, look over the generated pages
to check formatting.

Tests
.....

Every public function should have tests. We use `pytest
<https://pytest.org/>`__ for testing from Python. Each function being
tested has a separate test file in the ``tests/`` directory.

As much as possible, test not only expected use, but also error cases
(invalid arguments, array dtypes or shapes, etc.).

Functions in C++ that are *not* exposed to Python, like many of the
functions in ``adrt_cdefs_common.hpp`` can be tested using `Catch2
<https://github.com/catchorg/Catch2/tree/v2.x>`__. Each function being
tested has its own test file under ``tests/cpp/``.

Commit Messages
...............

Include a short summary of the changes on the first line. For changes
that are more involved, include further details in a new paragraph
after a blank line. In particular, discuss *why* the change was made.

Make sure your Git client is configured with your `name
<https://docs.github.com/en/get-started/getting-started-with-git/setting-your-username-in-git>`__
and `email
<https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address#setting-your-commit-email-address-in-git>`__
before committing.

Run tests
~~~~~~~~~

Once you have finished making your changes, you should run tests
locally and ensure that they pass before making a pull request.

We use `tox <https://tox.wiki/>`__ to run our tests and linters. Tox
will install required dependencies then run the tests. First, `install
tox <https://tox.wiki/en/latest/install.html>`__, then run::

  $ tox

This will also run our linters and report any Python style or
formatting issues.

Native Tests
............

You *do not* need to run C++ tests unless you have made changes to
them or have edited the functions they cover (in particular those in
``adrt_cdefs_common.hpp``). In most cases you can disregard this
section.

To run the tests, you will need a copy of `Catch2
<https://github.com/catchorg/Catch2>`__ version 2. We have a Python
script ``tools/download_catch2.py`` to retrieve one (requires
`requests <https://requests.readthedocs.io>`__)::

  $ python tools/download_catch2.py tests/cpp/catch2/catch.hpp

Then on a Linux system, run::

  $ g++ -std=c++11 -g -Wall -Wextra -Wpedantic $(find src/adrt tests/cpp/ -name '*.cpp' -not -name 'adrt_cdefs_py.cpp') -I src/adrt/ -o tests/cpp/test_all
  $ tests/cpp/test_all

A similar compilation process should work on other systems. Generally,
you need to compile all ``*.cpp`` files *except* ``adrt_cdefs_py.cpp``
and add ``src/adrt/`` to the include search path.

Submit a pull request
~~~~~~~~~~~~~~~~~~~~~

Once your changes are ready to submit, push your working branch to
your repository fork. Then create a pull request for branch with your
edits.

Automated tests will run on your pull request (including some that are
not run locally). Check the compilation logs for the automated runs
and fix any compiler warnings or test failures.

Thank you for your contribution!
