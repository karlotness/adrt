# Copyright (c) 2022 Karl Otness, Donsub Rim
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


r"""Utility routines for visualization further processing.

The ``adrt.utils`` module contains routines which are useful for
visualization or other basic processing tasks. These routines help to
transform outputs from the core algorithms into forms which may be
easier to process elsewhere. For example, by aligning the quadrants of
the ADRT into a single contiguous image, or interpolating the
irregular ADRT angles into a regular spacing.
"""


import operator
import typing
from collections import namedtuple
import numpy as np
import numpy.typing as npt
from ._wrappers import interp_to_cart


__all__: typing.Final[typing.Sequence[str]] = [
    "stitch_adrt",
    "unstitch_adrt",
    "truncate",
    "coord_adrt_to_cart",
    "coord_cart_to_adrt",
    "interp_to_cart",
]


_A = typing.TypeVar("_A", bound=np.generic)


def stitch_adrt(
    a: npt.NDArray[_A], /, *, remove_repeated: bool = False
) -> npt.NDArray[_A]:
    r"""Reshape and align ADRT channel-wise output into a contiguous image.

    The ADRT routine, :func:`adrt.adrt`, produces an output array
    which is divided into four separate quadrants, each containing the
    Radon transform results for a range of angles. This routine
    stitches these channels together so that they form a contiguous
    output. This may be especially useful in order to visualize the
    output as an image.

    The input array must have shape (..., 4, 2*N-1, N). The optional,
    extra leading dimensions will be treated as batch dimensions and
    will be preserved in the output.

    The parameter ``remove_repeated`` controls whether this output
    should have redundant columns (the last column in each quadrant)
    removed.

    See :ref:`adrt-description` for a description of the ADRT output
    quadrants.

    The function :func:`unstitch_adrt` provides an inverse for this
    operation.

    Parameters
    ----------
    a : numpy.ndarray
        Array of ADRT output data. Shape (..., 4, 2*N-1, N).
    remove_repeated : bool, optional
        Whether redundant columns should be removed. This removes the
        last column from each quadrant.

    Returns
    -------
    numpy.ndarray
        The input data, combined into a contiguous array. This will be
        an array with shape (..., 3*N-2, 4*N) or (..., 3*N-2, 4*N-4)
        if ``remove_repeated`` is ``True``.
    """
    n = a.shape[-1]
    if a.shape[-3:] != (4, 2 * n - 1, n):
        raise ValueError(f"unsuitable shape for ADRT output processing {a.shape}")
    # Compute output shape
    in_rows = 2 * n - 1
    out_rows = 3 * n - 2
    view_cols = n - (1 if remove_repeated else 0)
    output_shape = a.shape[:-3] + (out_rows, 4 * view_cols)
    view_shape = a.shape[:-3] + (out_rows, 4, view_cols)
    # We rely on C-ordered layout to merge the last two dimensions
    # without needing to copy
    ret = np.zeros_like(a, shape=view_shape, order="C")
    # Fill result array
    for i in range(4):
        quadrant = a[..., i, :, :]
        if i % 2:
            quadrant = np.flip(quadrant, axis=(-1, -2))
        if remove_repeated:
            quadrant = quadrant[..., :-1]
        if i < 2:
            ret[..., :in_rows, i, :] = quadrant
        else:
            ret[..., -in_rows:, i, :] = quadrant
    return ret.reshape(output_shape)


def unstitch_adrt(a: npt.NDArray[_A], /) -> npt.NDArray[_A]:
    r"""Slice a stitched ADRT output back into individual quadrants.

    This function provides an inverse for :func:`stitch_adrt` and
    re-slices, flips, and rotates its output into separate quadrants.
    It functions as an inverse regardless of the ``remove_repeated``
    argument that was specified when stitching.

    The input array must have shape (..., 3*N-2, 4*N) or (..., 3*N-2,
    4*N-4). The optional, extra leading dimensions will be treated as
    batch dimensions and will be preserved in the output.

    Parameters
    ----------
    a : numpy.ndarray
        Array of *stitched* ADRT output data.

    Returns
    -------
    numpy.ndarray
        The input data separated into quadrants. This array will have
        shape (..., 4, 2*N-1, N).
    """
    n = (a.shape[-2] + 2) // 3
    if a.shape[-2] != 3 * n - 2 or (a.shape[-1] != 4 * n and a.shape[-1] != 4 * n - 4):
        raise ValueError(f"unsuitable shape for ADRT unstitching {a.shape}")
    removed_repeated = a.shape[-1] == 4 * n - 4
    out_rows = 2 * n - 1
    a = a.reshape(a.shape[:-1] + (4, n - (1 if removed_repeated else 0)))
    ret = []
    for q in range(4):
        quadrant = a[..., :, q, :]
        if removed_repeated:
            # Need to re-add the removed column
            neighbor = a[..., :, (q + 1) % 4, 0, np.newaxis]
            if q == 3:
                # If we've circled the image we need to flip along the rows
                neighbor = np.flip(neighbor, axis=-2)
            quadrant = np.concatenate([quadrant, neighbor], axis=-1)
        # Slice the quadrant to the appropriate size
        if q < 2:
            quadrant = quadrant[..., :out_rows, :]
        else:
            quadrant = quadrant[..., -out_rows:, :]
        # Flip if necessary
        if q % 2:
            quadrant = np.flip(quadrant, axis=(-1, -2))
        ret.append(quadrant)
    # Stack result along a new quadrant dimension
    return np.stack(ret, axis=-3)


def truncate(a: npt.NDArray[_A], /) -> npt.NDArray[_A]:
    r"""Truncate and rotate square domain from iadrt or bdrt output.

    Parameters
    ----------
    a : numpy.ndarray
        array of shape (4,2*N-1,N) or (?,4,2*N-1,N) in which N = 2**n

    Returns
    -------
    out : numpy.ndarray
          array of shape (4,N,N) or (?,4,N,N) in which N = 2**n
    """
    n = a.shape[-1]
    if a.shape[-3:] != (4, 2 * n - 1, n):
        raise ValueError(f"unsuitable shape for ADRT output processing {a.shape}")
    return np.stack(
        [
            np.flip(a[..., 0, :n, :n], axis=-2).swapaxes(-1, -2),
            np.flip(a[..., 1, :n, :n], axis=-2),
            a[..., 2, :n, :n],
            np.flip(a[..., 3, :n, :n], axis=(-1, -2)).swapaxes(-1, -2),
        ],
        axis=-3,
    )


class CartesianCoord(typing.NamedTuple):
    angle: npt.NDArray[np.float64]
    offset: npt.NDArray[np.float64]


def coord_adrt_to_cart(n: typing.SupportsIndex, /) -> CartesianCoord:
    r"""Compute Radon domain coordinates of indices in the ADRT domain

    Parameters
    ----------
    n : int
        n specifies the dimension ADRT domain to be (4, 2*n-1, n)

    Returns
    -------
    angle : numpy.ndarray
        2D array of dimensions (4, n) containing Radon domain theta
        (angle) coordinates of the ADRT domain for all quadrants,
        stacked horizontally.

    offset : numpy.ndarray
        2D array of dimensions (4, 2*n-1, n) containing Radon domain s
        (offset) coordinates of the ADRT domain, stacked horizontally.

    See Also
    --------
    numpy.broadcast_to :
        For broadcasting ``angle`` to the shape of ADRT output
    """
    n = operator.index(n)
    if n < 2:
        raise ValueError(f"invalid Radon domain size {n}, must be at least 2")
    if (n - 1) & n != 0:
        raise ValueError(f"invalid Radon domain size {n}, must a power of two")
    hi, step = np.linspace(
        1, (1 - n) / n, num=2 * n - 1, endpoint=False, retstep=True, dtype=np.float64
    )
    hi += step / 2
    # Compute base angles
    ns = np.linspace(0, 1, num=n, endpoint=True, dtype=np.float64)
    theta = np.arctan(ns)  # [0, pi/4]
    theta_offset = theta - (np.pi / 2)
    h0 = ((np.add.outer(hi, ns) / (1 + ns)) - 0.5) * (np.cos(theta) + np.sin(theta))
    # Build output quadrants
    s_full = np.tile(
        np.concatenate(
            [
                h0[np.newaxis, ...],
                -h0[np.newaxis, ...],
            ],
            axis=0,
        ),
        (2, 1, 1),
    )
    theta_full = np.concatenate(
        [
            theta_offset[np.newaxis, ...],
            -theta[np.newaxis, ...],
            theta[np.newaxis, ...],
            -theta_offset[np.newaxis, ...],
        ],
        axis=0,
    )
    return CartesianCoord(theta_full, s_full)


ADRTIndex = namedtuple("ADRTIndex", ["quadrant", "height", "slope", "factor"])


def coord_cart_to_adrt(theta, t, n):
    r"""Find nearest ADRT entry indices for given point in Radon domain.

    Given a point (theta, s) in Radon domain, find the entry in the ADRT domain
    of dimensions (4, 2*n-1, n).

    Parameters
    ----------
    theta : numpy.ndarray
        1D array containing theta coordinates to convert to ADRT indices
    t : numpy.ndarray
        1D array containing s coordinates to convert to ADRT indices
    n : int
        size n determining the dimensions of the ADRT domain (4, 2*n-1, n)

    Returns
    -------
    quadrant : numpy.ndarray of numpy.uint8
        quadrant index in ADRT domain
    height : numpy.ndarray of numpy.int
        the intercept index in ADRT domain
    slope : numpy.ndarray of numpy.uint
        the slope index in ADRT domain
    factor : numpy.ndarray of numpy.float
        a transformation factor

    """
    th0 = (
        np.abs(theta)
        - np.abs(theta - np.pi / 4)
        - np.abs(theta + np.pi / 4)
        + np.pi / 2
    )
    q = (
        3
        - np.sign(theta).astype(np.uint8)
        - np.sign(theta - np.pi / 4).astype(np.uint8)
        - np.sign(theta + np.pi / 4).astype(np.uint8)
    ) // 2

    sgn = np.sign(theta) - np.sign(theta - np.pi / 4) - np.sign(theta + np.pi / 4)
    t0 = sgn * t

    s = np.tan(th0) * (n - 1)
    si = np.floor(s).astype(np.uint)
    factor = np.sqrt((si / n) ** 2 + (1 - 1 / n) ** 2) / n

    h0 = 0.5 * (1.0 + np.tan(th0)) - t0 / np.cos(th0)

    h = h0 * n + 0.5 * (sgn - 1)
    hi = np.floor(h).astype(int)

    out = ADRTIndex(q, hi, si, factor)
    return out
