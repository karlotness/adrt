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


__all__ = ["num_iters", "adrt_step", "adrt_init", "adrt_iter"]


import numpy as np
from ._wrappers import _format_object_type, num_iters, adrt_step


def adrt_init(a, /):
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
    # Shape is valid, create new output buffer
    n = a.shape[-1]
    output_shape = (a.shape[0], 4, 2 * n - 1, n)
    if a.ndim < 3:
        # No batch dimension
        output_shape = output_shape[1:]
    ret = np.zeros_like(a, shape=output_shape)
    # Quadrant 0
    ret[..., 0, :n, :] = np.flip(a, axis=-1).T
    # Quadrant 1
    ret[..., 1, :n, :] = np.flip(a, axis=-2)
    # Quadrant 2
    ret[..., 2, :n, :] = a
    # Quadrant 3
    ret[..., 3, :n, :] = np.flip(a, axis=(-1, -2)).T
    return ret


def adrt_iter(a, /):
    a = adrt_init(a)
    yield a.copy()
    for i in range(num_iters(a.shape[-1])):
        a = adrt_step(a, i)
        yield a.copy()
