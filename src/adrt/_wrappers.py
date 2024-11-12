# Copyright 2023 Karl Otness, Donsub Rim
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


r"""Python wrappers for functions from the native extension.

.. danger::
   This module is not part of the public API surface. Do not use it!

Most of these functions are re-exported from other modules and should
be used from there. Users must not import routines from this module;
they are not part of the public API surface and may be changed or
removed.

This module exists to centralize type conversion routines used to make
the native functions more pleasant to use from Python and to attach
Python documentation.
"""


import operator
import typing
import collections.abc
import numpy as np
import numpy.typing as npt
from . import _adrt_cdefs


__all__: typing.Final[collections.abc.Sequence[str]] = []


C = typing.TypeVar("C", bound=collections.abc.Callable[..., typing.Any])
A = typing.TypeVar("A", bound=np.generic)
F = typing.TypeVar("F", np.float32, np.float64)


def _set_module(module: str) -> collections.abc.Callable[[C], C]:
    r"""Override ``__module__`` on functions for documentation.

    This is an internal function. Users should not call it. This
    changes the way :func:`help` describes the function. Without this,
    functions in this module are listed as being in ``adrt._wrappers``
    rather than the module that users observe them in. NumPy does this
    for many of their functions which are defined in submodules but
    appear at the top level.
    """

    def decorate(func: C) -> C:
        func.__module__ = module
        return func

    return decorate


def _format_object_type(obj: object, /) -> str:
    r"""Given an object `obj`, return a formatted string for its type name.

    In particular, for built-in types omit the ``builtins`` module
    name, but for others, include it to give the full name of type
    type (such as "np.ndarray").
    """
    t = type(obj)
    if t.__module__ == "builtins":
        return str(t.__qualname__)
    return f"{t.__module__}.{t.__qualname__}"


def _normalize_array(a: npt.NDArray[A], /) -> npt.NDArray[A]:
    r"""Ensure provided arrays are in a suitable layout.

    This is an internal function. Users should not call it. Make sure
    that arrays being passed to the extension module have the expected
    memory layout.
    """
    if not isinstance(a, np.ndarray):
        # Explicitly fail if not ndarray (or subclass).
        # Users otherwise may get a confusing error related to the dtype attribute.
        raise TypeError(
            f"array must be numpy.ndarray, but got {_format_object_type(a)}"
        )
    a = np.asarray(a, a.dtype.newbyteorder("="), "C")
    if not a.flags.aligned:
        return a.copy("C")
    return a


@_set_module("adrt")
def adrt(a: npt.NDArray[F], /) -> npt.NDArray[F]:
    r"""The Approximate Discrete Radon Transform (ADRT).

    This is the fundamental routine of this package, computing the
    ADRT of the provided array. The array `a` must store square input
    images with sizes a power of two. The input may optionally include
    an additional leading batch dimension, so an array of either two
    or three dimensions.

    If padding is needed for the input array, consider
    :func:`numpy.pad`.

    The returned array will have the shape of an ADRT output of size
    N. The output is divided into four quadrants, each one less than
    twice as tall as the input. The taller height dimension represents
    ADRT offsets, while the final dimension has the same size as the
    input and represents the ADRT angles.

    For more information on the construction of the quadrants and the
    contents of this array see: :ref:`adrt-description`.

    Parameters
    ----------
    a : numpy.ndarray of float
        Array for which the ADRT should be computed. This should be a
        square image with side length a power of two, and optionally a
        leading batch dimension.

    Returns
    -------
    numpy.ndarray of float
        The ADRT of the provided data. For input images of size ``N``,
        each member of the batch will have shape ``(4, 2*N-1, N)``.

    Notes
    -----
    The transform implemented here is an approximation to the Radon
    transform and *approximates* the sums along lines with carefully
    chosen angles. Each quadrant slices along a range of :math:`\pi/4`
    radians. For a detailed description of the algorithm see
    :ref:`adrt-description` and refer to the source papers [#brady98]_
    [#press06]_.
    """
    return _adrt_cdefs.adrt(_normalize_array(a))


@_set_module("adrt.core")
def adrt_step(a: npt.NDArray[F], /, step: typing.SupportsIndex) -> npt.NDArray[F]:
    r"""Compute a single step of the ADRT.

    The ADRT implemented in :func:`adrt.adrt` is internally an
    iterative algorithm. Sums along line segments of a given length
    are approximated by joining sums along line segments of half
    length in a bottom-up fashion from segments of length two.

    This function allows you to run a single step of the ADRT in order
    to observe the outputs (for example to read off sums of partial
    line segments) or to modify the values as the computation proceeds
    (for example, to mask certain values as they grow).

    To use this function correctly, use :func:`adrt.core.adrt_init` to
    initialize your input array. The argument `a` to this function
    should either be the result of :func:`adrt_init` or a previous
    output of this function.

    Parameters
    ----------
    a : numpy.ndarray of float
        The array for which the single ADRT step should be computed.
    step : int
        The step to compute. The upper bound on this value should be
        computed using :func:`num_iters`, then `step` must be between
        :math:`0` and :math:`\mathtt{num\_iters}-1`, inclusive.

    Returns
    -------
    numpy.ndarray of float
        The result of the `step` iteration of the ADRT. The output has
        the same shape as the input.

    Note
    ----
    If you only want the result of the last step (the full ADRT) and
    are not interested in the intermediate steps, use the more
    efficient :func:`adrt.adrt`.
    """
    return _adrt_cdefs.adrt_step(_normalize_array(a), operator.index(step))


@_set_module("adrt")
def iadrt(a: npt.NDArray[F], /) -> npt.NDArray[F]:
    r"""An exact inverse to the ADRT.

    Computes an exact inverse to the ADRT, but only works for exact
    ADRT outputs. The array `a` may have an optional batch dimension
    with the shape of an ADRT output.

    The returned array has the same shape as `a`, but each quadrant
    should have only zeros below the square at the top of the array.
    However, this may not be the case due to conditioning or
    imprecision in the calculations performed by this routine.

    The upper square of each quadrant can be extracted and rotated
    using :func:`adrt.utils.truncate`.

    Parameters
    ----------
    a : numpy.ndarray of float
        An ADRT output for which to compute the inverse.

    Returns
    -------
    numpy.ndarray of float
        The computed inverse with the same shape as `a`.

    Warning
    -------
    This inverse is ill-conditioned and will be exact *only* if `a` is
    an exact output of the forward ADRT and the floating point type
    provides sufficient precision. In other cases this inverse is not
    appropriate.

    For an alternative, see the :doc:`examples.cginverse` example.

    Notes
    -----
    For details of the algorithm see :ref:`iadrt-description` or the
    source paper [#rim20]_.
    """
    return _adrt_cdefs.iadrt(_normalize_array(a))


@_set_module("adrt")
def bdrt(a: npt.NDArray[F], /) -> npt.NDArray[F]:
    r"""Backprojection operator for the ADRT.

    The transform implemented in :func:`adrt` is a linear operation.
    This function computes a generalized transpose of this operation,
    producing an output with the same shape as its input.

    To retrieve the entries of the transpose of the ADRT in the proper
    order, apply :func:`adrt.utils.truncate` to the array produced by
    this function. The truncation operation will remove the extended
    entries and rotate each quadrant into the same orientation as the
    original, square shape ADRT input. The quadrants can then be
    combined, if desired, potentially by :func:`numpy.mean`.

    Parameters
    ----------
    a : numpy.ndarray of float
        An ADRT output array to backproject.

    Returns
    -------
    numpy.ndarray of float
        Backprojection of `a` with the same shape.

    Notes
    -----
    For more details on the backprojection implemented here see the
    source paper [#press06]_.

    Examples
    --------
    As discussed above, this function can be used to compute the
    transpose of the operator applied by :func:`adrt` as follows::

      def adrt_tranpose(a):
          return adrt.utils.truncate(adrt.bdrt(a)).mean(axis=-3)
    """
    return _adrt_cdefs.bdrt(_normalize_array(a))


@_set_module("adrt.core")
def bdrt_step(a: npt.NDArray[F], /, step: typing.SupportsIndex) -> npt.NDArray[F]:
    r"""Compute a single step of the bdrt.

    The implementation of :func:`adrt.bdrt` is internally an iterative
    algorithm. This function allows you to run a single step of the
    bdrt in order to observe the outputs or to modify the values as
    the computation proceeds.

    To use this function correctly, the input `a` should be the result
    of an ADRT operation (:func:`adrt.adrt`) or a previous output of
    this function.

    Parameters
    ----------
    a : numpy.ndarray of float
        The array for which the single bdrt step should be computed.
        This array must have data type :obj:`float32 <numpy.float32>`
        or :obj:`float64 <numpy.float64>`.
    step : int
        The step to compute. The upper bound on this value should be
        computed using :func:`num_iters`, then `step` must be between
        :math:`0` and :math:`\mathtt{num\_iters}-1`, inclusive.

    Returns
    -------
    numpy.ndarray of float
        The result of the `step` iteration of the bdrt. The output has
        the same shape as the input.

    Note
    ----
    If you only want the result of the last step and are not
    interested in the intermediate steps, use the more efficient
    :func:`adrt.bdrt`.
    """
    return _adrt_cdefs.bdrt_step(_normalize_array(a), operator.index(step))


@_set_module("adrt.utils")
def interp_to_cart(a: npt.NDArray[F], /) -> npt.NDArray[F]:
    r"""Interpolate an ADRT output into a regular Cartesian grid.

    The angles and offsets used in an ADRT output are irregularly-spaced to
    enable reuse of intermediate calculations.  This routine provides a basic
    interpolation operation which resamples these angles into an even-spacing.

    The ADRT result is interpolated into a uniform Cartesian grid in the Radon
    domain :math:`(\theta, t)`. Where :math:`\theta` is the normal direction of
    the line, and :math:`t` is the distance of the line to the origin. Upon a
    coordinate transformation, a nearest neighbor interpolation is performed.

    For an ADRT output of size ``N``, the interpolated array has shape
    ``(N, 4*N)`` with an optional batch dimension preserved.

    Parameters
    ----------
    a : numpy.ndarray of float
        ADRT output array to interpolate.

    Returns
    -------
    numpy.ndarray of float
        Interpolated Cartesian grid data.

    Notes
    -----

    See the :doc:`coordinate transform section <examples.coordinate>` for more
    details on the coordinate transform.
    """
    return _adrt_cdefs.interp_to_cart(_normalize_array(a))


@_set_module("adrt.core")
def threading_enabled() -> bool:
    r"""Indicate whether core routines provide multithreading.

    Many of the core routines in this package can optionally be built
    with internal multithreading support using OpenMP. If this is
    enabled these functions will internally split their work across
    multiple threads.

    Returns
    -------
    bool
        :pycode:`True` if this module is built with internal threading
        support, otherwise :pycode:`False`.
    """
    return _adrt_cdefs.OPENMP_ENABLED


def _press_fmg_restriction(a: npt.NDArray[F], /) -> npt.NDArray[F]:
    return _adrt_cdefs.press_fmg_restriction(_normalize_array(a))


def _press_fmg_prolongation(a: npt.NDArray[F], /) -> npt.NDArray[F]:
    return _adrt_cdefs.press_fmg_prolongation(_normalize_array(a))


def _press_fmg_highpass(a: npt.NDArray[F], /) -> npt.NDArray[F]:
    return _adrt_cdefs.press_fmg_highpass(_normalize_array(a))
