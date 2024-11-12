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


class TestIAdrtCdefs:
    def test_accepts_float32(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float32)
        _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_accepts_float32_four_dim(self):
        size = 16
        inarr = np.zeros((5, 4, 2 * size - 1, size), dtype=np.float32)
        _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_accepts_float64(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float64)
        _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_accepts_float64_four_dim(self):
        size = 16
        inarr = np.zeros((5, 4, 2 * size - 1, size), dtype=np.float64)
        _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_int32(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.int32)
        with pytest.raises(TypeError, match="int32"):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_mismatched_shape(self):
        size = 16
        inarr_a = np.zeros((4, 2 * size - 1, size - 1), dtype=np.float32)
        inarr_b = np.zeros((4, 2 * size - 2, size), dtype=np.float32)
        with pytest.raises(ValueError, match="must have a valid ADRT output shape"):
            _ = adrt._adrt_cdefs.iadrt(inarr_a)
        with pytest.raises(ValueError, match="must have a valid ADRT output shape"):
            _ = adrt._adrt_cdefs.iadrt(inarr_b)

    def test_refuses_five_dim(self):
        size = 16
        inarr = np.zeros((6, 5, 4, 2 * size - 1, size), dtype=np.float32)
        with pytest.raises(ValueError, match="between 3 and 4 dimensions, but had 5"):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_non_power_of_two(self):
        size = 17
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float32)
        with pytest.raises(ValueError, match="must have a valid ADRT output shape"):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_non_array(self):
        with pytest.raises(
            TypeError, match="must be a NumPy array or compatible subclass"
        ):
            _ = adrt._adrt_cdefs.iadrt(None)
        base_list = [[1.0, 2.0, 3.0, 4.0]] * 7
        arr_list = [base_list for _ in range(4)]
        with pytest.raises(
            TypeError, match="must be a NumPy array or compatible subclass"
        ):
            _ = adrt._adrt_cdefs.iadrt(arr_list)

    def test_refuses_fortran_order(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float32, order="F")
        with pytest.raises(ValueError, match="must be .*C-order"):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_c_non_contiguous(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, 2 * size), dtype=np.float32, order="F")
        inarr = inarr[:, :, ::2]
        assert inarr.shape == (4, 31, 16)
        assert not inarr.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError, match="must be .*contiguous"):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_byteswapped(self):
        size = 16
        inarr = np.ones(
            (4, 2 * size - 1, size), dtype=np.dtype(np.float32).newbyteorder()
        )
        with pytest.raises(ValueError, match="must be .*native byte order"):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_zero_axis_array(self):
        size = 16
        inarr = np.zeros((0, 4, 2 * size - 1, size), dtype=np.float32)
        with pytest.raises(ValueError, match="found zero in dimension 0"):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_zero_size_planes(self):
        inarr = np.zeros((4, 0, 0), dtype=np.float32)
        with pytest.raises(ValueError, match="found zero in dimension (1|2)"):
            _ = adrt._adrt_cdefs.iadrt(inarr)


class TestIAdrt:
    def test_accepts_float32(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float32)
        _ = adrt.iadrt(inarr)

    def test_accepts_float64(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float64)
        _ = adrt.iadrt(inarr)

    def test_accepts_float32_returned_dtype(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float32)
        c_out = adrt.iadrt(inarr)
        assert c_out.dtype == np.float32

    def test_accepts_float64_returned_dtype(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float64)
        c_out = adrt.iadrt(inarr)
        assert c_out.dtype == np.float64

    def test_refuses_int32(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.int32)
        with pytest.raises(TypeError, match="int32"):
            _ = adrt.iadrt(inarr)

    def test_accepts_fortran_order(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float32, order="F")
        _ = adrt.iadrt(inarr)

    def test_accepts_c_non_contiguous(self):
        size = 16
        inarr = np.zeros((4, 2 * (2 * size - 1), size), dtype=np.float32, order="F")
        inarr = inarr[:, ::2]
        assert inarr.shape == (4, 2 * size - 1, size)
        assert not inarr.flags["C_CONTIGUOUS"]
        _ = adrt.iadrt(inarr)

    def test_accepts_byteswapped(self):
        size = 16
        inarr_native = np.ones((4, 2 * size - 1, size), dtype=np.float32)
        inarr_swapped = inarr_native.astype(inarr_native.dtype.newbyteorder("S"))
        out_native = adrt.iadrt(inarr_native)
        out_swapped = adrt.iadrt(inarr_swapped)
        assert np.all(out_native == out_swapped)

    def test_all_zeros_square(self):
        size = 16
        inarr = np.zeros((size, size), dtype=np.float32)
        adrt_out = adrt.adrt(inarr)
        inv = adrt.iadrt(adrt_out)
        inv = np.mean(adrt.utils.truncate(inv), axis=0)
        assert inv.shape == inarr.shape
        assert inv.dtype == inarr.dtype
        assert np.allclose(inv, inarr)

    def test_all_ones_square(self):
        size = 16
        inarr = np.ones((size, size), dtype=np.float32)
        adrt_out = adrt.adrt(inarr)
        inv = adrt.iadrt(adrt_out)
        inv = np.mean(adrt.utils.truncate(inv), axis=0)
        assert inv.shape == inarr.shape
        assert inv.dtype == inarr.dtype
        assert np.allclose(inv, inarr)

    def test_unique_values(self):
        size = 32
        inarr = np.arange(size**2).reshape((size, size)).astype("float32")
        adrt_out = adrt.adrt(inarr)
        inv = adrt.iadrt(adrt_out)
        inv = np.mean(adrt.utils.truncate(inv), axis=0)
        assert inv.shape == inarr.shape
        assert inv.dtype == inarr.dtype
        assert np.allclose(inv, inarr)

    def test_batch_dimension_unique_values(self):
        size = 32
        inarr = np.arange(4 * (size**2)).reshape((4, size, size)).astype("float32")
        adrt_out = adrt.adrt(inarr)
        inv = adrt.iadrt(adrt_out)
        inv = np.mean(adrt.utils.truncate(inv), axis=1)
        assert inv.shape == inarr.shape
        assert inv.dtype == inarr.dtype
        assert np.allclose(inv, inarr)

    def test_small_1x1(self):
        inarr = np.ones((4, 1, 1), dtype="float64")
        expected_out = np.ones((1, 1), dtype="float64")
        c_out = adrt.iadrt(inarr)
        c_out = np.mean(adrt.utils.truncate(c_out), axis=0)
        assert c_out.shape == expected_out.shape
        assert np.allclose(c_out, expected_out)

    def test_small_1x1_batch(self):
        expected_out = np.arange(5, dtype="float64").reshape((5, 1, 1)) + 1
        inarr = np.stack([expected_out] * 4, axis=1)
        c_out = adrt.iadrt(inarr)
        c_out = np.mean(adrt.utils.truncate(c_out), axis=1)
        assert c_out.shape == expected_out.shape
        assert np.allclose(c_out, expected_out)
