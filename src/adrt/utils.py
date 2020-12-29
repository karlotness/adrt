# Copyright (C) 2020 Karl Otness, Donsub Rim
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


import numpy as np


def stitch_adrt(a, *, remove_repeated=False):
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
    a : array_like
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
        raise ValueError(f"Unsuitable shape for ADRT output processing: {a.shape}")

    if len(a.shape) > 3:
        had_batch = True
        batch_size = a.shape[:-3]
    else:
        had_batch = False
        batch_size = (1,)
        a = np.expand_dims(a, 0)
    # Compute working array shape
    in_rows = 2 * n - 1
    out_rows = 3 * n - 2
    out_cols = 4 * n - (4 if remove_repeated else 0)
    output_shape = (*batch_size, out_rows, out_cols)

    # Process input array
    ret = np.zeros_like(a, shape=output_shape)
    for i in range(4):

        if (i % 2) == 0:
            quadrant = a[..., i, :, :]
        else:
            quadrant = a[..., i, ::-1, ::-1]
        if remove_repeated:
            quadrant = quadrant[..., :-1]
        if i < 2:
            ret[
                ..., :in_rows, i * (out_cols // 4) : (i + 1) * (out_cols // 4)
            ] = quadrant
        else:
            ret[
                ..., -in_rows:, i * (out_cols // 4) : (i + 1) * (out_cols // 4)
            ] = quadrant

    if had_batch:
        return ret
    else:
        # Remove batch dimension
        return ret[0]


def truncate(a, orient=0):
    r"""
    Truncate and rotate square domain from iadrt or bdrt output

    Parameters
    ----------
    a : array_like
        array of shape (4,2*N-1,N) or (?,4,2*N-1,N) in which N = 2**n

    Returns
    -------
    out : array_like
          array of shape (4,N,N) or (?,4,N,N) in which N = 2**n

    """

    if a.ndim == 3:
        n = a.shape[-1]
        out = np.zeros((4, n, n), dtype=a.dtype)
        if orient % 2 == 0:
            out[0, :, :] = a[0, :n, :n][::-1, :].T
            out[1, :, :] = a[1, :n, :n][::-1, :]
            out[2, :, :] = a[2, :n, :n]
            out[3, :, :] = a[3, :n, :n][::-1, ::-1].T
        elif orient % 2 == 1:
            out[0, :, :] = a[0, :n, :n][::-1, :].T
            out[1, :, :] = a[1, (n - 1) :, :n][:, ::-1]
            out[2, :, :] = a[2, :n, :n]
            out[3, :, :] = a[3, (n - 1) :, :n].T

    elif a.ndim == 4:
        n = a.shape[-1]
        out = np.zeros((a.shape[0], 4, n, n), dtype=a.dtype)
        if orient % 2 == 0:
            out[:, 0, :, :] = a[:, 0, :n, :n][:, ::-1, :].transpose((0, 2, 1))
            out[:, 1, :, :] = a[:, 1, :n, :n][:, ::-1, :]
            out[:, 2, :, :] = a[:, 2, :n, :n]
            out[:, 3, :, :] = a[:, 3, :n, :n][:, ::-1, ::-1].transpose((0, 2, 1))
        elif orient % 2 == 1:
            out[:, 0, :, :] = a[:, 0, :n, :n][:, ::-1, :].transpose((0, 2, 1))
            out[:, 1, :, :] = a[:, 1, (n - 1) :, :n][:, :, ::-1]
            out[:, 2, :, :] = a[:, 2, :n, :n]
            out[:, 3, :, :] = a[:, 3, (n - 1) :, :n].transpose((0, 2, 1))

    return out


def interp_to_cart(adrt_out):
    r"""

    Interpolate ADRT result to a uniform Cartesian grid in the Radon domain
    of (theta, s): theta is the normal direction of the line and s is the
    distance of the line to the origin.

    Parameters
    ----------
    a : array_like
        array of shape (4,2*N-1,N)

    Returns
    -------
    theta_cart_out : array_like
          array of shape (N,4*N) containing coordinates theta (angles)
    s_cart_out : array_like
          array of shape (N,4*N) containing coordinates s (offsets)
    adrt_cart_out : array_like
          array of shape (N,4*N) containing interpolated data

    """

    def _coord_transform(h, t, n):
        """
        Map to coordinates

        """
        theta = np.arctan(t / (n - 1))
        l0 = 0.5 * (np.cos(theta) + np.sin(theta))
        h0 = (n - 1 + t - h) / (n - 1 + t)
        s = -l0 * h0 + (1 - h0) * l0

        return theta, s

    n = adrt_out.shape[-1]

    theta_canon = np.zeros((2 * n - 1, n))
    s_canon = np.zeros((2 * n - 1, n))
    for t in range(n):
        for h in range(2 * n - 1):
            theta, s = _coord_transform(h, t, n)
            theta_canon[h, t] = np.rad2deg(theta)
            s_canon[h, t] = s

    adrt_cart_out = np.zeros((n, 4 * n))
    theta_cart_out = np.zeros((n, 4 * n))
    s_cart_out = np.zeros((n, 4 * n))

    for i in range(4):
        if i == 0:
            s_cart = np.linspace(-np.sqrt(2) / 2, np.sqrt(2) / 2, n)
            th_cart = np.linspace(-90.0, -45.0, n)

            theta_loc = theta_canon - 90.0
            s_loc = s_canon
            quadrant = adrt_out[i, :, :]

            z = np.zeros((n, n))
            w = np.zeros((n, n))
            for j in range(n):
                z[:, j] = np.interp(
                    s_cart, s_loc[:, j], quadrant[:, j], left=0.0, right=0.0
                )
            for j in range(n):
                w[j, :] = np.interp(th_cart, theta_loc[j, :], z[j, :]) / np.cos(
                    np.deg2rad(th_cart[::-1] + 45.0)
                )

            theta_cart, s_cart = np.meshgrid(th_cart, s_cart)

        elif i == 1:
            s_cart = np.linspace(-np.sqrt(2) / 2, np.sqrt(2) / 2, n)
            th_cart = np.linspace(-45.0, 0.0, n)

            theta_loc = -theta_canon
            theta_loc = theta_loc[::-1, :]
            s_loc = -s_canon[::-1, :]
            quadrant = adrt_out[i, ::-1, :]

            z = np.zeros((n, n))
            w = np.zeros((n, n))
            for j in range(n):
                z[:, j] = np.interp(
                    s_cart, s_loc[:, j], quadrant[:, j], left=0.0, right=0.0
                )
            for j in range(n):
                w[j, :] = np.interp(th_cart, theta_loc[j, ::-1], z[j, ::-1]) / np.cos(
                    np.deg2rad(th_cart)
                )

            theta_cart, s_cart = np.meshgrid(th_cart, s_cart)

        elif i == 2:
            s_cart = np.linspace(-np.sqrt(2) / 2, np.sqrt(2) / 2, n)
            th_cart = np.linspace(0.0, 45.0, n)

            z = np.zeros((n, n))
            w = np.zeros((n, n))
            for j in range(n):
                z[:, j] = np.interp(
                    s_cart, s_canon[:, j], adrt_out[i, :, j], left=0.0, right=0.0
                )
            for j in range(n):
                w[j, :] = np.interp(th_cart, theta_canon[j, :], z[j, :]) / np.cos(
                    np.deg2rad(th_cart)
                )

            theta_cart, s_cart = np.meshgrid(th_cart, s_cart)

        elif i == 3:
            s_cart = np.linspace(-np.sqrt(2) / 2, np.sqrt(2) / 2, n)
            th_cart = np.linspace(45.0, 90.0, n)

            theta_loc = (45.0 - theta_canon) + 45.0
            theta_loc = theta_loc[::-1, :]
            s_loc = -s_canon[::-1, :]
            quadrant = adrt_out[i, ::-1, :]

            z = np.zeros((n, n))
            w = np.zeros((n, n))
            for j in range(n):
                z[:, j] = np.interp(
                    s_cart, s_loc[:, j], quadrant[:, j], left=0.0, right=0.0
                )
            for j in range(n):
                w[j, :] = np.interp(th_cart, theta_loc[j, ::-1], z[j, ::-1]) / np.cos(
                    np.deg2rad(th_cart[::-1] - 45.0)
                )

            theta_cart, s_cart = np.meshgrid(th_cart, s_cart)

        adrt_cart_out[:, i * n : (i + 1) * n] = w
        theta_cart_out[:, i * n : (i + 1) * n] = theta_cart
        s_cart_out[:, i * n : (i + 1) * n] = s_cart

    return theta_cart_out, s_cart_out, adrt_cart_out


def cgiadrt(da, **kwargs):
    r"""
    Use the conjugate gradient algorithm to invert the least-squares inverse of
    the ADRT

    Parameters
    ----------
    da : array_like
        an array of ADRT shape (4,2*N-1,N)

    **kwargs
        Additional keyword arguments passed `scipy.sparse.linalg.cg`

    Returns
    -------
    out : array_like
        tuple out put of scipy.sparse.linalg.cg, first entry is a
        (flattened) 1D array of shape (N**2,)

    """

    import adrt
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import cg

    def _matmul(x):
        r"""
        Computes the matrix multiplication R^T R where R is the full ADRT.

        Parameters
        ----------
        x : array_like
            (flattened) 1D array of shape (N**2,) where N = 2**n

        Returns
        -------
        x_out :
            (flattened) 1D array of shape (N**2,) where N = 2**n

        """

        n2 = x.shape[0]
        n = int(np.round(np.sqrt(n2)))
        x2 = x.reshape((n, n))

        da = adrt.adrt(x2)
        ba = adrt.bdrt(da)
        ta = truncate(ba)
        ma = np.mean(ta, axis=0).flatten()

        return ma

    n = da.shape[-1]

    ba = adrt.bdrt(da)
    ta = truncate(ba)
    ta = np.mean(ta, axis=0)

    if "x0" not in kwargs.keys():
        kwargs["x0"] = ta.flatten()

    ta = ta.flatten()
    linop = LinearOperator((n ** 2, n ** 2), matvec=_matmul, dtype=da.dtype)
    out = cg(linop, ta, **kwargs)

    return out
