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


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_single_all_ones(dtype):
    size = 16
    in_arr = np.ones((size, size)).astype(dtype)
    out_arr = adrt.core.adrt_init(in_arr)
    assert in_arr.dtype == out_arr.dtype
    assert out_arr.shape == (4, 2 * size - 1, size)
    assert np.all(out_arr[:, :size, :] == 1)
    assert np.all(out_arr[:, size:, :] == 0)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_single_unique_values(dtype):
    size = 16
    in_arr = np.arange(size**2).reshape((size, size)).astype(dtype)
    out_arr = adrt.core.adrt_init(in_arr)
    values = set(in_arr.astype(np.int32).ravel())
    assert in_arr.dtype == out_arr.dtype
    assert out_arr.shape == (4, 2 * size - 1, size)
    assert set(out_arr.astype(np.int32).ravel()) == values
    assert np.all(out_arr[:, size:, :] == 0)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_batch_unique_values(dtype):
    size = 16
    batches = 3
    in_arr = np.arange(batches * size**2).reshape((batches, size, size)).astype(dtype)
    batch_out_arr = adrt.core.adrt_init(in_arr)
    single_out_arr = np.stack([adrt.core.adrt_init(in_arr[i]) for i in range(batches)])
    assert batch_out_arr.shape[0] == batches
    assert batch_out_arr.ndim == 4
    assert batch_out_arr.shape == single_out_arr.shape
    assert batch_out_arr.dtype == single_out_arr.dtype
    assert np.allclose(batch_out_arr, single_out_arr)


def test_refuses_non_array():
    with pytest.raises(TypeError, match="must be numpy.ndarray"):
        adrt.core.adrt_init(None)
    with pytest.raises(TypeError, match="must be numpy.ndarray"):
        adrt.core.adrt_init(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )


def test_refuses_too_many_dims():
    in_arr = np.ones((2, 3, 16, 16)).astype("float32")
    with pytest.raises(ValueError, match="between 2 and 3 dimensions, but had 4"):
        adrt.core.adrt_init(in_arr)


def test_refuses_too_few_dims():
    in_arr = np.ones(16).astype("float32")
    with pytest.raises(ValueError, match="between 2 and 3 dimensions, but had 1"):
        adrt.core.adrt_init(in_arr)


def test_refuses_non_square():
    in_arr = np.ones((3, 16, 15)).astype("float32")
    with pytest.raises(ValueError, match="must be square"):
        adrt.core.adrt_init(in_arr)
    in_arr = np.ones((15, 16)).astype("float32")
    with pytest.raises(ValueError, match="must be square"):
        adrt.core.adrt_init(in_arr)


def test_refuses_non_power_of_two():
    in_arr = np.ones((7, 7)).astype("float32")
    with pytest.raises(ValueError, match="power of two shape"):
        adrt.core.adrt_init(in_arr)
    in_arr = np.ones((2, 7, 7)).astype("float32")
    with pytest.raises(ValueError, match="power of two shape"):
        adrt.core.adrt_init(in_arr)


def test_refuses_zero_axis_array():
    inarr = np.zeros((0, 32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="found zero in dimension 0"):
        adrt.core.adrt_init(inarr)
