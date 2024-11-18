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


def np_highpass(arr):
    assert arr.ndim == 2
    a = -1 / 16
    b = -1 / 8
    c = 3 / 4
    conv_kernel = np.expand_dims(np.array([[a, b, a], [b, c, b], [a, b, a]]), (0, 1))
    arr = np.lib.stride_tricks.sliding_window_view(
        np.pad(arr, 1, mode="reflect"), window_shape=(3, 3), writeable=False
    )
    return np.sum((arr * conv_kernel), axis=(-1, -2)).astype(arr.dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_checkerboard(dtype):
    x, y = np.meshgrid(np.arange(7), np.arange(10))
    arr = ((x % 2) ^ (y % 2)).astype(dtype) * 2 - 1
    adrt_hp = adrt._wrappers._press_fmg_highpass(arr)
    np_hp = np_highpass(arr)
    assert adrt_hp.shape == arr.shape
    assert adrt_hp.dtype == arr.dtype
    assert np.allclose(adrt_hp, np_hp)
    assert np.allclose(adrt_hp, arr)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_vert_stripes(dtype):
    x, y = np.meshgrid(np.arange(7), np.arange(10))
    arr = (y % 2).astype(dtype) * 2 - 1
    adrt_hp = adrt._wrappers._press_fmg_highpass(arr)
    np_hp = np_highpass(arr)
    assert adrt_hp.shape == arr.shape
    assert adrt_hp.dtype == arr.dtype
    assert np.allclose(adrt_hp, np_hp)
    assert np.allclose(adrt_hp, arr)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_horiz_stripes(dtype):
    x, y = np.meshgrid(np.arange(7), np.arange(10))
    arr = (x % 2).astype(dtype) * 2 - 1
    adrt_hp = adrt._wrappers._press_fmg_highpass(arr)
    np_hp = np_highpass(arr)
    assert adrt_hp.shape == arr.shape
    assert adrt_hp.dtype == arr.dtype
    assert np.allclose(adrt_hp, np_hp)
    assert np.allclose(adrt_hp, arr)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_const(dtype):
    arr = np.full((14, 5), 5, dtype=dtype)
    adrt_hp = adrt._wrappers._press_fmg_highpass(arr)
    np_hp = np_highpass(arr)
    assert adrt_hp.shape == arr.shape
    assert adrt_hp.dtype == arr.dtype
    assert np.allclose(adrt_hp, np_hp)
    assert np.allclose(adrt_hp, 0)


def test_unique_values():
    arr = np.arange(56).reshape((7, 8)).astype(np.float32)
    adrt_hp = adrt._wrappers._press_fmg_highpass(arr)
    np_hp = np_highpass(arr)
    assert adrt_hp.shape == arr.shape
    assert adrt_hp.dtype == arr.dtype
    assert np.allclose(adrt_hp, np_hp)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unique_values_batch(dtype):
    arr = np.arange(3 * 256).reshape((3, 16, 16)).astype(np.float32)
    adrt_hp = adrt._wrappers._press_fmg_highpass(arr)
    np_hp = np.stack([np_highpass(a) for a in arr])
    assert adrt_hp.shape == arr.shape
    assert adrt_hp.dtype == arr.dtype
    assert np.allclose(adrt_hp, np_hp)


def test_small():
    arr = np.arange(4).reshape((2, 2)).astype(np.float64)
    adrt_hp = adrt._wrappers._press_fmg_highpass(arr)
    np_hp = np_highpass(arr)
    assert adrt_hp.shape == arr.shape
    assert adrt_hp.dtype == arr.dtype
    assert np.allclose(adrt_hp, np_hp)


def test_rejects_tiny():
    arr = np.ones((1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="array is too small"):
        _ = adrt._wrappers._press_fmg_highpass(arr)
    with pytest.raises(ValueError, match="array is too small"):
        _ = adrt._wrappers._press_fmg_highpass(arr.T)


def test_rejects_non_float():
    arr = np.ones((16, 16), dtype=np.int32)
    with pytest.raises(TypeError, match="int32"):
        _ = adrt._wrappers._press_fmg_highpass(arr)
