# Copyright (c) 2020, 2021 Karl Otness, Donsub Rim
# All rights reserved
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


r"""Low-level routines for observing or influencing the basic algorithms.

The ``adrt.core`` module provides low-level routines that make it
possible to observe the progress of other iterative algorithms
provided in this package, or to intervene in their computation.

Broadly, there are two types of routines provided here: generators
which yield snapshots of each phase in the iterative computation; and
single-step functions which allow executing a single, specified step.

These routines could be used to implement the basic routines from the
:mod:`adrt` module; however the implementations there are more
efficient. If you only want the final result of the basic algorithms,
you should use the functions from the main :mod:`adrt` module.

However, if you want to observe or modify the progress of the basic
algorithms these routines make this possible. If you want only to
observe the individual steps of the iterative computations---but not
modify them---then the iterator routines here may be useful.
Otherwise, if you want to perform more advanced operations and
intervene in and modify the progress of the computations the
single-step routines make that possible.
"""


__all__ = ["num_iters", "adrt_step", "adrt_init", "adrt_iter", "bdrt_step", "bdrt_iter"]


import numpy as np
from ._wrappers import _format_object_type, num_iters, adrt_step, bdrt_step


def adrt_init(a, /):
    r"""Initialize an array for use with :func:`adrt_step`.

    This function processes square arrays with side lengths a power of
    two. These arrays may also optionally have an optional batch
    dimension. This function is intended to be used with
    :func:`adrt.core.adrt_step`.

    Parameters
    ----------
    a : numpy.ndarray
        The array which will be made suitable for further processing
        with the ADRT. This array must have a square shape with sides
        a power of two, optionally with a leading batch dimension.

    Returns
    -------
    numpy.ndarray
        The input array duplicated, stacked, flipped and rotated to
        make it suitable for further processing with the ADRT. The
        output array will have shape :math:`(b, 4, 2n-1, n)` where
        :math:`b` is the optional batch dimension.
    """
    # Explicitly require an ndarray (or subclass).
    if not isinstance(a, np.ndarray):
        raise TypeError(
            f"Array argument must be numpy.ndarray, got {_format_object_type(a)}"
        )
    # Check input shape
    if a.ndim > 3 or a.ndim < 2:
        raise ValueError("ADRT input must have two or three dimensions")
    if (
        a.shape[-1] != a.shape[-2]
        or ((a.shape[-1] - 1) & a.shape[-1]) != 0
        or not all(a.shape)
    ):
        raise ValueError("ADRT input must be square, with shape a power of two")
    # Shape is valid, create new output buffer and copy
    n = a.shape[-1]
    output_shape = a.shape[:-2] + (4, 2 * n - 1, n)
    ret = np.zeros_like(a, shape=output_shape)
    ret[..., 0, :n, :] = np.flip(a, axis=-1).swapaxes(-1, -2)
    ret[..., 1, :n, :] = np.flip(a, axis=-2)
    ret[..., 2, :n, :] = a
    ret[..., 3, :n, :] = np.flip(a, axis=(-1, -2)).swapaxes(-1, -2)
    return ret


def adrt_iter(a, /, *, copy=True):
    r"""Yield individual steps of the ADRT.

    The ADRT implemented in :func:`adrt.adrt` is internally an
    iterative algorithm. Sums along line segments of a given length
    are approximated by joining sums along line segments of half
    length in a bottom-up fashion from segments of length two.

    This function allows you observe individual steps of the ADRT. It
    is a generator which will yield first the initialized array,
    followed by the outputs of each iteration of the ADRT.

    Parameters
    ----------
    a : numpy.ndarray
        The array for which steps of the ADRT will be computed. This
        array must have data type :obj:`float32 <numpy.float32>` or
        :obj:`float64 <numpy.float64>`.
    copy : bool, optional
        If true (default), the arrays produced by this generator are
        independent copies. Otherwise, read-only views are produced
        and these *must not* be modified without making a
        :meth:`copy <numpy.ndarray.copy>` first.

    Yields
    ------
    numpy.ndarray
        Successive stages of the ADRT computation. First, the
        unprocessed array computed by :func:`adrt_init` followed by
        snapshots of progress after each ADRT iteration.

    Note
    ----
    If you only want the result of the last step (the full ADRT) and
    are not interested in the intermediate steps, use the more
    efficient :func:`adrt.adrt`.
    """
    a = adrt_init(a)
    a.setflags(write=False)
    yield a.copy() if copy else a.view()
    for i in range(num_iters(a.shape[-1])):
        a = adrt_step(a, i)
        a.setflags(write=False)
        yield a.copy() if copy else a.view()


def bdrt_iter(a, /, *, copy=True):
    r"""Yield individual steps of the bdrt.

    The implementation of :func:`adrt.bdrt` is internally an iterative
    algorithm. This function allows you observe individual steps of
    the bdrt. It is a generator which will yield the outputs of each
    iteration of the bdrt.

    Parameters
    ----------
    a : numpy.ndarray
        The array for which steps of the bdrt will be computed. This
        array must have data type :obj:`float32 <numpy.float32>` or
        :obj:`float64 <numpy.float64>`.
    copy : bool, optional
        If true (default), the arrays produced by this generator are
        independent copies. Otherwise, read-only views are produced
        and these *must not* be modified without making a
        :meth:`copy <numpy.ndarray.copy>` first.

    Yields
    ------
    numpy.ndarray
        Successive stages of the bdrt computation, a snapshot of the
        progress after each bdrt iteration.

    Note
    ----
    If you only want the result of the last step and are not
    interested in the intermediate steps, use the more efficient
    :func:`adrt.bdrt`.
    """
    for i in range(num_iters(a.shape[-1])):
        a = bdrt_step(a, i)
        a.setflags(write=False)
        yield a.copy() if copy else a.view()
