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
@pytest.mark.parametrize("remove_repeated", [True, False])
def test_unstitch_adrt(dtype, remove_repeated):
    n = 16
    adrt_in = np.arange(n * n, dtype=np.float64).reshape((n, n))
    adrt_out = adrt.adrt(adrt_in).astype(dtype)
    stitched = adrt.utils.stitch_adrt(adrt_out, remove_repeated=remove_repeated)
    unstitched = adrt.utils.unstitch_adrt(stitched)
    assert stitched.dtype == unstitched.dtype
    assert unstitched.dtype == adrt_out.dtype
    assert unstitched.shape == adrt_out.shape
    assert np.all(unstitched == adrt_out)


@pytest.mark.parametrize("remove_repeated", [True, False])
def test_unstitch_adrt_batch(remove_repeated):
    n = 16
    adrt_in = np.arange(6 * n * n, dtype=np.float32).reshape((6, n, n))
    adrt_out = adrt.adrt(adrt_in)
    # Split the batch dimension
    adrt_out = adrt_out.reshape((2, 3) + adrt_out.shape[1:])
    stitched = adrt.utils.stitch_adrt(adrt_out, remove_repeated=remove_repeated)
    unstitched = adrt.utils.unstitch_adrt(stitched)
    assert stitched.dtype == unstitched.dtype
    assert unstitched.dtype == adrt_out.dtype
    assert unstitched.shape == adrt_out.shape
    assert np.all(unstitched == adrt_out)


def test_refuses_wrong_last_dimension():
    n = 16
    stitched = np.zeros((5, 3 * n - 2, 4 * n - 3), dtype=np.float32)
    with pytest.raises(ValueError, match="unsuitable shape .*ADRT unstitch"):
        _ = adrt.utils.unstitch_adrt(stitched)


def test_refuses_wrong_second_to_last_dimension():
    n = 16
    stitched = np.zeros((5, 3 * n - 1, 4 * n), dtype=np.float32)
    with pytest.raises(ValueError, match="unsuitable shape .*ADRT unstitch"):
        _ = adrt.utils.unstitch_adrt(stitched)
