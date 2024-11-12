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


class TestBdrtStepCdefs:
    def test_refuses_too_few_args(self):
        arr = np.zeros((4, 31, 16), dtype=np.float32)
        with pytest.raises(TypeError, match="bdrt_step.* expected 2 arguments, got 1"):
            adrt._adrt_cdefs.bdrt_step(arr)

    def test_refuses_too_many_args(self):
        arr = np.zeros((4, 31, 16), dtype=np.float64)
        with pytest.raises(TypeError, match="bdrt_step.* expected 2 arguments, got 3"):
            adrt._adrt_cdefs.bdrt_step(arr, 0, 0)

    def test_refuses_non_array(self):
        arr = np.zeros((4, 31, 16), dtype=np.float64).tolist()
        with pytest.raises(
            TypeError, match="must be a NumPy array or compatible subclass"
        ):
            adrt._adrt_cdefs.bdrt_step(arr, 0)

    def test_refuses_non_integer(self):
        arr = np.zeros((4, 31, 16), dtype=np.float64)
        with pytest.raises(TypeError, match="int"):
            adrt._adrt_cdefs.bdrt_step(arr, [])

    def test_refuses_out_of_range(self):
        arr = np.zeros((4, 31, 16), dtype=np.float32)
        with pytest.raises(
            ValueError, match=r"step \-1 is out of range.*adrt\.core\.num_iters"
        ):
            adrt._adrt_cdefs.bdrt_step(arr, -1)
        with pytest.raises(
            ValueError, match=r"step 4 is out of range.*adrt\.core\.num_iters"
        ):
            adrt._adrt_cdefs.bdrt_step(arr, 4)


class TestBdrtStep:
    @pytest.mark.parametrize("size", [2, 4, 8])
    def test_materialize_array(self, size):
        n = size
        full_size = 4 * n * (2 * n - 1)
        basis_arr = np.eye(full_size, dtype=np.float32).reshape(
            (full_size, 4, 2 * n - 1, n)
        )
        num_iters = adrt.core.num_iters(n)
        for bdrt_step, adrt_step in zip(
            range(num_iters), reversed(range(num_iters)), strict=True
        ):
            bdrt_out = adrt.core.bdrt_step(basis_arr, bdrt_step)
            adrt_out = adrt.core.adrt_step(basis_arr, adrt_step)
            bdrt_mat = bdrt_out.reshape((full_size, full_size))
            adrt_mat = adrt_out.reshape((full_size, full_size))
            assert np.allclose(adrt_mat, bdrt_mat.T)
            assert adrt_mat.shape == bdrt_mat.T.shape

    def test_refuses_non_integer(self):
        arr = np.zeros((4, 31, 16), dtype=np.float64)
        with pytest.raises((TypeError, AttributeError)):
            adrt.core.bdrt_step(arr, 0.0)
