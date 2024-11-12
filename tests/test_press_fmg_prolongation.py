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


def np_prolongation(a):
    return np.repeat(np.repeat(a, 2, axis=-1), 2, axis=-2)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unique_values(dtype):
    arr = np.arange(30).reshape((5, 6)).astype(dtype)
    result = adrt._wrappers._press_fmg_prolongation(arr)
    np_result = np_prolongation(arr)
    assert arr.dtype == result.dtype
    assert result.shape == np_result.shape
    assert np.all(result == np_result)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unique_values_batch(dtype):
    arr = np.arange(90).reshape((3, 5, 6)).astype(dtype)
    result = adrt._wrappers._press_fmg_prolongation(arr)
    np_result = np_prolongation(arr)
    assert arr.dtype == result.dtype
    assert result.shape == np_result.shape
    assert np.all(result == np_result)


def test_tiny():
    arr = np.array([[3.0]])
    result = adrt._wrappers._press_fmg_prolongation(arr)
    np_result = np_prolongation(arr)
    assert arr.dtype == result.dtype
    assert result.shape == np_result.shape
    assert np.all(result == np_result)


def test_reject_too_few_dims():
    arr = np.arange(7).astype("float32")
    with pytest.raises(ValueError, match="between 2 and 3 dimensions, but had 1"):
        _ = adrt._wrappers._press_fmg_prolongation(arr)


def test_reject_too_many_dims():
    arr = np.arange(180).reshape((2, 3, 5, 6)).astype("float32")
    with pytest.raises(ValueError, match="between 2 and 3 dimensions, but had 4"):
        _ = adrt._wrappers._press_fmg_prolongation(arr)


def test_reject_non_float():
    arr = np.arange(30).reshape((5, 6)).astype("int32")
    with pytest.raises(TypeError, match="int32"):
        _ = adrt._wrappers._press_fmg_prolongation(arr)
