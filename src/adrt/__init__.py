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


r"""Fast approximate discrete Radon transform for NumPy arrays.

The ``adrt`` package provides a fast implementation of an approximate
discrete Radon transform and several related routines such as
backprojection, inverses, and utilities.

This main module contains the basic functions of the package, designed
to be simple to use. The module :mod:`adrt.core` contains lower-level
routines which provide more control over the computation. Other
utility routines can be found in :mod:`adrt.utils`.
"""


import typing
import collections.abc
import itertools
import numpy as np
import numpy.typing as npt
from ._wrappers import adrt, iadrt, bdrt
from . import utils, core


__all__: typing.Final[collections.abc.Sequence[str]] = [
    "adrt",
    "iadrt",
    "bdrt",
    "iadrt_fmg",
    "utils",
    "core",
]
__version__: typing.Final[str] = "1.2.0"


_F = typing.TypeVar("_F", np.float32, np.float64)


def iadrt_fmg(
    a: npt.NDArray[_F], /, *, max_iters: int | None = None
) -> npt.NDArray[_F]:
    r"""Approximate inverse to the ADRT by the full multigrid method.

    Estimated inverses are computed and iteratively refined until the
    norm of the residual error fails to decrease from one iteration to
    the next. This iteration can also be terminated early if
    `max_iters` is specified.

    Particularly with a limited iteration count this inverse can be
    relatively quick to compute, but may not achieve the best possible
    precision. If your output array `a` is *exact* and floating-point
    precision is sufficient you may consider :func:`iadrt` for an
    exact inverse. Otherwise, for an inverse that may perform more
    reliably for certain inputs, consider the ``iadrt_cg`` recipe
    proposed in the :doc:`examples.cginverse` example.

    See :func:`adrt.core.iadrt_fmg_iter` and
    :func:`adrt.core.iadrt_fmg_step` for more information on the
    iterative refinement applied internally and the full multigrid
    method.

    Parameters
    ----------
    a : numpy.ndarray of float
        The array for which the inverse is to be computed. This array
        must have the shape of an ADRT output. This array *must not*
        have a batch dimension.
    max_iters : int, optional
        If :pycode:`None` (default), the number of internal iterations
        is unbounded. The computation will terminate only when the
        norm of the residual error fails to decrease. Otherwise, this
        must be an integer argument at least :pycode:`1`, and provides
        an upper bound on the number of iterations performed
        internally.

    Returns
    -------
    numpy.ndarray of float
        A square inverse computed by the full multigrid method with
        the lowest residual error observed so long as the decrease was
        monotonic, or the last computation output if iteration was
        terminated by `max_iters`.

    Notes
    -----
    Unlike :func:`iadrt` the output of this routine will be
    square---the same size as the original input to :func:`adrt` would
    have been.

    This function *does not* support batch dimensions.
    """
    if a.ndim > 3:
        raise ValueError(
            f"batch dimension not supported for iadrt_fmg, got {a.ndim} dimensions"
        )
    if max_iters is not None and max_iters < 1:
        raise ValueError(
            f"must allow at least one iteration, but specified {max_iters}"
        )
    _inv1 = a  # Silence linter warning about unbound variable
    for (_inv1, res1), (_, res2) in itertools.pairwise(
        itertools.chain(
            (
                # Pair each estimated inverse x with its residual error
                (x, float(np.linalg.norm(adrt(x) - a)))
                # Use itertools.islice to limit iterations if requested
                for x in itertools.islice(core.iadrt_fmg_iter(a, copy=False), max_iters)
            ),
            # Chain i2 with one extra value so we don't exhaust early
            # Use np.inf so the residual will rise and we won't continue iterating
            [(a, np.inf)],
        )
    ):
        if not res2 < res1:
            # Residual failed to decrease, stop early
            break
    # Create a copy so returned array is writable (we have views from iadrt_fmg_iter)
    return _inv1.copy()
