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


import pytest
import numpy as np
import adrt


CONV_KERNEL = np.array(
    [[-1 / 16, -1 / 8, -1 / 16], [-1 / 8, 3 / 4, -1 / 8], [-1 / 16, -1 / 8, -1 / 16]]
)


def press_highpass(arr):
    assert arr.ndim == 2
    conv_kernel = np.expand_dims(CONV_KERNEL, (0, 1))
    arr = np.lib.stride_tricks.sliding_window_view(
        np.pad(arr, 1, mode="reflect"), window_shape=(3, 3), writeable=False
    )
    return np.sum((arr * conv_kernel), axis=(-1, -2)).astype(arr.dtype)


def press_prolongation(arr):
    return np.repeat(np.repeat(arr, 2, axis=-1), 2, axis=-2)


def press_restriction(arr):
    assert arr.ndim == 3
    return (1 / 4) * (arr[:, 0:-1:2, ::2] + arr[:, 1::2, ::2])


def press_inverse(arr):
    # A basic, recursive implementation of the press FMG step
    assert arr.ndim == 3
    assert arr.shape[0] == 4
    # base case
    if arr.shape[-1] == 1:
        return arr[0]
    # recursive case
    r_n_2 = press_restriction(arr)
    f_n_2 = press_inverse(r_n_2)
    f_n_prime = press_prolongation(f_n_2)
    partial_fwd = adrt.adrt(f_n_prime)
    residual_drt = partial_fwd - arr
    bdrt_res = adrt.utils.truncate(adrt.bdrt(residual_drt))
    bdrt_sum = bdrt_res.mean(axis=0) / (arr.shape[-1] - 1)
    image_correction = press_highpass(bdrt_sum)
    return f_n_prime - image_correction


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unique_values(dtype):
    arr = np.arange(4 * 31 * 16).reshape((4, 31, 16)).astype(dtype)
    adrt_result = adrt.core.iadrt_fmg_step(arr)
    reference = press_inverse(arr)
    assert adrt_result.dtype == arr.dtype
    assert adrt_result.shape == (16, 16)
    assert np.allclose(adrt_result, reference, atol=1e-3)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unique_values_batch(dtype):
    arr = np.arange(3 * 4 * 31 * 16).reshape((3, 4, 31, 16)).astype(dtype)
    adrt_result = adrt.core.iadrt_fmg_step(arr)
    reference = np.stack([press_inverse(arr[i]) for i in range(arr.shape[0])])
    assert adrt_result.dtype == arr.dtype
    assert adrt_result.shape == (3, 16, 16)
    assert np.allclose(adrt_result, reference, atol=1e-3)
