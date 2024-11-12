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


def np_restriction(a):
    return (1 / 4) * (a[..., 0:-1:2, ::2] + a[..., 1::2, ::2])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unqiue_values(dtype):
    arr = np.arange(4 * 11 * 6).reshape((4, 11, 6)).astype(dtype)
    result = adrt._wrappers._press_fmg_restriction(arr)
    np_result = np_restriction(arr)
    assert result.dtype == arr.dtype
    assert result.shape == (4, 5, 3)
    assert np.allclose(result, np_result)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unique_values_batch(dtype):
    arr = np.arange(2 * 4 * 11 * 6).reshape((2, 4, 11, 6)).astype(dtype)
    result = adrt._wrappers._press_fmg_restriction(arr)
    np_result = np_restriction(arr)
    assert result.dtype == arr.dtype
    assert result.shape == (2, 4, 5, 3)
    assert np.allclose(result, np_result)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_tiny(dtype):
    arr = np.arange(4 * 3 * 2).reshape((4, 3, 2)).astype(dtype)
    result = adrt._wrappers._press_fmg_restriction(arr)
    np_result = np_restriction(arr)
    assert result.dtype == arr.dtype
    assert result.shape == (4, 1, 1)
    assert np.allclose(result, np_result)


def test_reject_not_even():
    arr = np.arange(4 * 5 * 3).reshape((4, 5, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="must have a valid ADRT output shape"):
        _ = adrt._wrappers._press_fmg_restriction(arr)


def test_reject_mismatch_shape():
    arr = np.arange(4 * 10 * 6).reshape((4, 10, 6)).astype(np.float32)
    with pytest.raises(ValueError, match="must have a valid ADRT output shape"):
        _ = adrt._wrappers._press_fmg_restriction(arr)


def test_reject_mismatch_missing_quadrants():
    arr = np.arange(2 * 11 * 6).reshape((2, 11, 6)).astype(np.float32)
    with pytest.raises(ValueError, match="must have a valid ADRT output shape"):
        _ = adrt._wrappers._press_fmg_restriction(arr)


def test_reject_mismatch_non_float():
    arr = np.arange(4 * 11 * 6).reshape((4, 11, 6)).astype(np.int32)
    with pytest.raises(TypeError, match="int32"):
        _ = adrt._wrappers._press_fmg_restriction(arr)
