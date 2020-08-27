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
from . import _adrt_cdefs


def _normalize_array(a):
    native_dtype = a.dtype.newbyteorder("=")
    return np.require(a, dtype=native_dtype, requirements=["C_CONTIGUOUS", "ALIGNED"])


def adrt(a):
    r"""The Approximate Discrete Radon Transform (ADRT).

    Computes the ADRT of the provided matrix, `a`. The matrix `a` may
    have either two or three dimensions. If it has three dimensions,
    the first dimension, is treated as a batch and the ADRT is
    computed for each layer independently. The dimensions of the layer
    data must have equal size N, where N is a power of two. The input
    shape is ``(B?, N, N)``.

    The returned array will have a shape of either three or four
    dimensions. The optional fourth dimension has the same size as the
    batch dimension of `a`, if present. The output is divided into
    four quadrants each representing a range of angles. The third and
    fourth axes index into Radon transform displacements and angles,
    respectively. The output has shape: ``(B?, 4, 2 * N - 1, N)``.

    For more information on the construction of the quadrants and the
    contents of this array see: :ref:`adrt-description`.

    Parameters
    ----------
    a : array_like of float
        The array of data for which the ADRT should be computed.

    Returns
    -------
    numpy.ndarray
        The ADRT of the provided data.

    Raises
    ------
    ValueError
        If the array has an invalid shape.
    TypeError
        If the array has an unsupported (i.e. non-float) dtype.

    Notes
    -----
    The transform implemented here is an approximation to the Radon
    transform and *approximates* the sums along lines with carefully
    chosen angles. Each quadrant slices along a range of :math:`\pi/4`
    radians. For a detailed description of the algorithm see
    :ref:`adrt-description` and refer to the source papers [press08]_,
    [brady98]_.

    References
    ----------
    .. [press08] William H. Press, *A Fast Discrete Approximation
       Algorithm for the Radon Transform Related Databases*, SIAM Journal
       on Computing, 27. https://doi.org/10.1073/pnas.0609228103
    .. [brady98] Martin L. Brady, *Discrete Radon transform has an exact,
       fast inverse and generalizes to operations other than sums along
       lines*, Proceedings of the National Academy of Sciences, 103.
       https://doi.org/10.1137/S0097539793256673
    """
    return _adrt_cdefs.adrt(_normalize_array(a))


def iadrt(a):
    return _adrt_cdefs.iadrt(_normalize_array(a))
