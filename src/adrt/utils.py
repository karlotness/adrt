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
import numpy as np
import numpy.typing as npt
from ._wrappers import interp_to_cart


__all__: typing.Final[typing.Sequence[str]] = [
    "stitch_adrt",
    "unstitch_adrt",
    "truncate",
    "coord_adrt",
    "coord_cart_to_adrt",
    "interp_to_cart",
]


_A = typing.TypeVar("_A", bound=np.generic)


def stitch_adrt(
    a: npt.NDArray[_A], /, *, remove_repeated: bool = False
) -> npt.NDArray[_A]:
    r"""Reshape and align ADRT quadrants output into a contiguous image.

    The ADRT routine, :func:`adrt.adrt`, produces an output array
    which is divided into four separate quadrants, each containing the
    Radon transform results for a range of angles. This routine
    stitches these channels together so that they form a contiguous
    output. This may be especially useful in order to visualize the
    output as an image.

    The input array should have the *relative* shape of an ADRT output
    (but the base dimension ``N`` need not be a power of two). Any
    number of optional leading dimensions will be treated as batch
    dimensions and will be preserved in the output.

    The output array will be four times as wide as any one original
    quadrant (with four fewer columns if `remove_repeated` is
    :pycode:`True`), and will be have an additional ``N-1`` rows
    added.

    Parameters
    ----------
    a : numpy.ndarray
        Array of ADRT output data to be stitched.
    remove_repeated : bool, optional
        If :pycode:`False` (default) all columns are preserved in the
        output. If :pycode:`True`, the redundant last column in each
        quadrant is removed.

    Returns
    -------
    numpy.ndarray
        The input data repositioned to form a contiguous array.

    Notes
    -----
    The columns which are removed by `remove_repeated` are only truly
    redundant if `a` has the symmetries of a real ADRT output.

    See :ref:`adrt-description` for a description of the ADRT output
    quadrants.

    The function :func:`unstitch_adrt` provides an inverse for this
    operation.
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
    It functions as an inverse regardless of the `remove_repeated`
    argument that was specified when stitching so long as the ADRT
    output that was stitched respected the symmetries of a real ADRT
    output. In other cases, the removed columns may not have been
    redundant.

    Parameters
    ----------
    a : numpy.ndarray
        Array of *stitched* ADRT output data.

    Returns
    -------
    numpy.ndarray
        The input data re-separated into ADRT quadrants with the
        *relative* shape of an ADRT output.
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
    r"""Truncate and rotate a rectangular ADRT output into a square.

    ADRT output arrays consist of four rectangular quadrants with
    different orientations, such that the image data is not stacked in
    a corresponding position (see the illustration in
    :ref:`adrt-description`).

    This function fixes both aspects. It slices each ADRT quadrant
    into a square, and rotates them so that they are stacked in a
    consistent orientation (in particular, this forms an inverse for
    :func:`adrt.core.adrt_init`).

    For this routine the input array `a`, must have the same
    *relative* shape of an ADRT output, but the base dimension ``N``
    need not be a power of two. The array may also have any number of
    leading batch dimensions.

    Parameters
    ----------
    a : numpy.ndarray
        An ADRT output array with rectangular quadrants.

    Returns
    -------
    numpy.ndarray
        An array with four square quadrants each rotated into a
        consistent orientation.

    Notes
    -----
    This routine can be used to:

    * Slice the output of :func:`adrt.bdrt` before collapsing with
      :func:`numpy.mean` to produce the standard transpose to
      :func:`adrt.adrt`.
    * Slice and rotate the result of :func:`adrt.iadrt` before
      collapsing with :func:`numpy.mean`.
    * Invert :func:`adrt.core.adrt_init`.
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


class ADRTCoord(typing.NamedTuple):
    offset: npt.NDArray[np.float64]
    angle: npt.NDArray[np.float64]


def coord_adrt(n: typing.SupportsIndex, /) -> ADRTCoord:
    r"""Compute Radon domain coordinates of indices in the ADRT domain

    The return value ``angle`` can be broadcast to full size using
    :func:`numpy.broadcast_to`, if desired.

    Parameters
    ----------
    n : int
        n specifies the dimension ADRT domain to be (4, 2*n-1, n)

    Returns
    -------
    offset : numpy.ndarray of numpy.float64
        3D array of dimensions (4, 2*n-1, n) containing Radon domain's
        (offset) coordinates of the ADRT domain for each of 4 quadrants.
    angle : numpy.ndarray of numpy.float64
        3D array of dimensions (4, 1, n) containing Radon domain theta
        (angle) coordinates of the ADRT domain for each of 4 quadrants.
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
    h0 = ((np.add.outer(hi, (1 - 0.5 / n) * ns) / (1 + ns)) - 0.5) * (
        np.cos(theta) + np.sin(theta)
    )
    # Build output quadrants
    s_full = np.tile(np.stack([h0, -h0], axis=0), (2, 1, 1))
    theta_full = np.expand_dims(
        np.stack([theta_offset, -theta, theta, -theta_offset], axis=0), axis=1
    )
    return ADRTCoord(s_full, theta_full)


class ADRTIndex(typing.NamedTuple):
    quadrant: npt.NDArray[np.uint8]
    height: npt.NDArray[np.int64]
    slope: npt.NDArray[np.uint64]
    factor: npt.NDArray[np.float64]


def coord_cart_to_adrt(
    theta: npt.NDArray[typing.Union[np.float32, np.float64]],
    t: npt.NDArray[typing.Union[np.float32, np.float64]],
    n: typing.SupportsIndex,
) -> ADRTIndex:
    r"""Find nearest ADRT entry indices for given point in Radon domain.

    Given a point (theta, t) in Radon domain, find the entry in the ADRT domain
    of dimensions (4, 2*n-1, n).

    Parameters
    ----------
    theta : numpy.ndarray
        array containing theta coordinates to convert to ADRT indices
    t : numpy.ndarray
        array containing t coordinates to convert to ADRT indices
    n : int
        size n determining the dimensions of the ADRT domain (4, 2*n-1, n)

    Returns
    -------
    quadrant : numpy.ndarray of numpy.uint8
        quadrant index in ADRT domain
    height : numpy.ndarray of numpy.int64
        the intercept index in ADRT domain
    slope : numpy.ndarray of numpy.uint64
        the slope index in ADRT domain
    factor : numpy.ndarray of numpy.float64
        a transformation factor
    """
    n = operator.index(n)
    if n < 2:
        raise ValueError(f"invalid Radon domain size {n}, must be at least 2")
    if (n - 1) & n != 0:
        raise ValueError(f"invalid Radon domain size {n}, must a power of two")
    if theta.shape != t.shape:
        raise ValueError(
            f"mismatched shapes for theta and t {theta.shape} vs. {t.shape}"
        )

    # Move theta values into canonical range [-pi/2, pi/2]
    theta = np.where(
        np.abs(theta) <= np.pi / 2,
        theta,
        np.remainder(theta + np.pi / 2, np.pi) - np.pi / 2,
    )

    th0 = (
        np.abs(theta)
        - np.abs(theta - np.pi / 4)
        - np.abs(theta + np.pi / 4)
        + np.pi / 2
    )
    q = (np.floor(np.clip(theta / (np.pi / 4), -2, 1)).astype(np.int8) + 2).astype(
        np.uint8
    )
    sgn = 1 - 2 * (q % 2).astype(int)
    t0 = sgn * t

    s = np.tan(th0) * (n - 1)
    si = np.round(s).astype(np.uint)
    side = si / (n - 1)
    factor = np.sqrt(1.0 + side**2)

    h0 = 0.5 * (1.0 + np.tan(th0)) - t0 * factor
    h = h0 * n
    hi = np.floor(h - 0.1).astype(int)

    return ADRTIndex(q, hi, si, factor)
