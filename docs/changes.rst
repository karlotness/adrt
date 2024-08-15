Changelog
=========

This document provides a brief summary of changes in each released
version of `adrt`. More information and release builds are also
available on the `GitHub releases page
<https://github.com/karlotness/adrt/releases>`__.

v1.2.0 (Unreleased)
-------------------

* Drop support for Python 3.9, now requires version 3.10 or later
* Add support for experimental `free-threaded mode
  <https://docs.python.org/3.13/whatsnew/3.13.html#whatsnew313-free-threaded-cpython>`__
  in Python 3.13 (when built from source)

v1.1.0
------

* Add support for NumPy 2.0
* Drop support for Python 3.8, now requires version 3.9 or later
* Building this version requires a C++20 compiler

v1.0.1
------
* Improve test compatibility with PyPy
* Use generators internally in :func:`adrt.iadrt_fmg`
* Add SPDX license identifiers to source

v1.0.0
------
Initial release
