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

    The input array must have shape (..., 4, 2*N-1, N). Extra leading
    dimensions will be treated as batch dimensions and preserved in
    the output.

    The parameter ``remove_repeated`` controls whether this output
    should have redundant columns (the last column in each quadrant)
    removed.

    See :ref:`adrt-description` for a description of the ADRT output
    quadrants.

    Parameters
    ----------
    a : array_like
        Array of ADRT output data. Shape (..., 4, 2*N-1, N) where
        N is a power of two.
    remove_repeated : bool, optional
        Whether redundant columns should be removed. This removes the
        last column from each quadrant.

    Returns
    -------
    numpy.ndarray
        The input data, combined into a contiguous array. This will be
        an array with shape (..., 2*N-1, 4*N) or (..., 2*N-1, 4*N-4)
        if ``remove_repeated`` is ``True``.
    """

    n = a.shape[-1]
    if a.shape[-3:] != (4, 2 * n - 1, n):
        raise ValueError(f"Unsuitable shape ADRT output processing: {a.shape}")

    quadrants = []
    for i in range(4):
        quadrant = a[..., i, :, :]
        if remove_repeated:
            quadrant = quadrant[..., :-1]
        quadrants.append(quadrant)

    return np.concatenate(quadrants, axis=-1)


def truncate(a, orient=1):
    r"""
    Truncate square domain from iadrt or bdrt output

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
