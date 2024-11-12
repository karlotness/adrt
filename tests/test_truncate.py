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
def test_inverts_adrt_init(dtype):
    size = 16
    in_arr = np.arange(5 * size**2).reshape((5, size, size)).astype(dtype)
    out_arr = adrt.utils.truncate(adrt.core.adrt_init(in_arr))
    assert out_arr.shape == (5, 4, size, size)
    assert np.allclose(np.expand_dims(in_arr, 1), out_arr)
    assert out_arr.dtype == in_arr.dtype


def test_accepts_bdrt_output():
    size = 16
    in_arr = adrt.bdrt(np.ones((4, 2 * size - 1, size)))
    out_arr = adrt.utils.truncate(in_arr)
    assert out_arr.dtype == in_arr.dtype
    assert out_arr.shape == (4, size, size)


def test_accepts_iadrt_output():
    size = 16
    in_arr = adrt.iadrt(np.ones((4, 2 * size - 1, size)))
    out_arr = adrt.utils.truncate(in_arr)
    assert out_arr.dtype == in_arr.dtype
    assert out_arr.shape == (4, size, size)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((4, 0, 0), id="zero_dims"),
        pytest.param((4, 4, 2), id="mismatched_sizes"),
        pytest.param((7, 4), id="too_few_dims"),
    ],
)
def test_rejects_invalid_sizes(shape):
    inarr = np.ones(shape).astype("float32")
    with pytest.raises(ValueError, match="unsuitable shape .*ADRT output"):
        _ = adrt.utils.truncate(inarr)
