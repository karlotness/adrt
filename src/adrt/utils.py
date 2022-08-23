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


__all__ = ["stitch_adrt", "truncate", "interp_to_cart"]


import numpy as np


def stitch_adrt(a, /, *, remove_repeated=False):
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


def truncate(a, /):
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


def interp_to_cart(adrt_out, /):
    r"""Interpolate the ADRT result to a Cartesian angle vs. offset grid.

    Interpolate ADRT result to a uniform Cartesian grid in the Radon domain
    of (theta, s): theta is the normal direction of the line and s is the
    distance of the line to the origin.

    Parameters
    ----------
    a : numpy.ndarray of float
        array of shape (4,2*N-1,N)

    Returns
    -------
    theta_cart_out : numpy.ndarray
          array of shape (N,4*N) containing coordinates theta (angles)
    s_cart_out : numpy.ndarray
          array of shape (N,4*N) containing coordinates s (offsets)
    adrt_cart_out : numpy.ndarray
          array of shape (N,4*N) containing interpolated data

    """

    def _coord_transform(th, s):
        n = th.shape[0]
        th0 = np.abs(th) - np.abs(th - np.pi / 4) - np.abs(th + np.pi / 4) + np.pi / 2
        q = (
            3
            - np.sign(th).astype(int)
            - np.sign(th - np.pi / 4).astype(int)
            - np.sign(th + np.pi / 4).astype(int)
        ) // 2

        sgn = np.sign(th) - np.sign(th - np.pi / 4) - np.sign(th + np.pi / 4)
        s0 = sgn * s

        t = np.tan(th0) * (n - 1)
        ti = np.floor(t).astype(int)
        factor = np.sqrt((ti / n) ** 2 + (1 - 1 / n) ** 2) / n

        h0 = 0.5 * (1.0 + np.tan(th0)) - s0 / np.cos(th0)

        h = h0 * n + 0.5 * (sgn - 1)
        hi = np.floor(h).astype(int)

        return (q, factor, ti, hi)

    def _halfgrid(a, n):
        leftgrid, step = np.linspace(-a, a, num=n, endpoint=False, retstep=True)

        return leftgrid + 0.5 * step

    if adrt_out.ndim == 4:
        n = adrt_out.shape[3]
    elif adrt_out.ndim == 3:
        n = adrt_out.shape[2]

    theta_cart_out = _halfgrid(0.5 * np.pi, 4 * n)
    s_cart_out = _halfgrid(np.sqrt(2) / 2, n)

    angle, offset = np.meshgrid(theta_cart_out, s_cart_out)

    (quadrant, factor, adrt_tindex, adrt_hindex) = _coord_transform(angle, offset)

    ii = np.logical_and(adrt_hindex > -1, adrt_hindex < 2 * n - 1)

    adrt_cart_out = np.zeros((n, 4 * n))

    print(adrt_hindex[ii].min())
    print(adrt_hindex[ii].max())

    print(adrt_tindex[ii].min())
    print(adrt_tindex[ii].max())
    adrt_cart_out[ii] = (
        factor[ii] * adrt_out[quadrant[ii], adrt_hindex[ii], adrt_tindex[ii]]
    )

    return theta_cart_out, s_cart_out, adrt_cart_out
