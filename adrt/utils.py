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


def stitch_adrt(a):
    r"""
    Reshape 4-channel ADRT output to zero-padded 2D continguous array

    Parameters
    ----------
    a : array_like
        array of shape (4,2*N,N) in which N = 2**n

    Returns
    -------
    Z : array_like
        array of shape (3*N-2,4*N) containing a zero-padded continguous array
        with ADRT data

    """

    if (
        not isinstance(a, np.ndarray)
        or (a.shape[0] != 4)
        or ((a.shape[1] + 1) != 2 * a.shape[2])
    ):
        raise ValueError("Passed array is not of the right shape")

    dtype = a.dtype
    m = a.shape[2]

    z = np.zeros((3 * m - 2, 4 * m), dtype=dtype)

    z[: (2 * m - 1), :m] = a[0, :, :]
    z[: (2 * m - 1), m : (2 * m)] = a[1, :, :]
    z[(m - 1) :, (2 * m) : (3 * m)] = a[2, :, :]
    z[(m - 1) :, (3 * m) : (4 * m)] = a[3, :, :]

    return z
