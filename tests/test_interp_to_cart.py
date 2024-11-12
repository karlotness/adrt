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


def _cellcenters(a, b, n):
    leftgrid, step = np.linspace(a, b, num=n, endpoint=False, retstep=True)
    centers = leftgrid + 0.5 * step
    return centers


def py_interp_to_cart(adrt_out, /):
    n = adrt_out.shape[-1]
    adrt_cart_out = np.zeros(adrt_out.shape[:-3] + (n, 4 * n))
    theta_cart_out = _cellcenters(-0.5 * np.pi, 0.5 * np.pi, 4 * n)
    s_cart_out = _cellcenters(np.sqrt(2) / 2, -np.sqrt(2) / 2, n)
    angle, offset = np.meshgrid(theta_cart_out, s_cart_out)
    adrt_index = adrt.utils.coord_cart_to_adrt(angle, offset, n)
    quadrant, adrt_hindex, adrt_tindex, factor = adrt_index
    ii = np.logical_and(adrt_hindex > -1, adrt_hindex < 2 * n - 1)
    adrt_cart_out[..., ii] = (
        factor[ii] * adrt_out[..., quadrant[ii], adrt_hindex[ii], adrt_tindex[ii]]
    )
    return adrt_cart_out


class TestInterpToCartCdefs:
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_accepts_float(self, dtype):
        _ = adrt._adrt_cdefs.interp_to_cart(np.zeros((4, 15, 8), dtype=dtype))

    def test_refuses_int32(self):
        arr = np.zeros((4, 15, 8), dtype=np.int32)
        with pytest.raises(TypeError, match="int32"):
            _ = adrt._adrt_cdefs.interp_to_cart(arr)

    def test_refuses_non_array(self):
        with pytest.raises(
            TypeError, match="must be a NumPy array or compatible subclass"
        ):
            _ = adrt._adrt_cdefs.interp_to_cart(None)
        base_list = np.arange(4, 7, 4, dtype=np.float32).tolist()
        with pytest.raises(
            TypeError, match="must be a NumPy array or compatible subclass"
        ):
            _ = adrt._adrt_cdefs.interp_to_cart(base_list)

    def test_refuses_fortran_order(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float32, order="F")
        with pytest.raises(ValueError, match="must be .*C-order"):
            _ = adrt._adrt_cdefs.interp_to_cart(inarr)

    def test_refuses_c_non_contiguous(self):
        size = 16
        inarr = np.zeros((4, 2 * (2 * size - 1), size), dtype=np.float32, order="F")
        inarr = inarr[:, ::2]
        assert inarr.shape == (4, 2 * size - 1, size)
        assert not inarr.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError, match="must be .*contiguous"):
            _ = adrt._adrt_cdefs.interp_to_cart(inarr)

    def test_refuses_byteswapped(self):
        size = 16
        inarr_native = np.ones((4, 2 * size - 1, size), dtype=np.float32)
        inarr_swapped = inarr_native.astype(inarr_native.dtype.newbyteorder("S"))
        with pytest.raises(ValueError, match="must be .*native byte order"):
            _ = adrt._adrt_cdefs.interp_to_cart(inarr_swapped)


class TestInterpToCart:
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_interp_matches_small_input(self, dtype):
        a = np.arange(16, dtype=dtype).reshape((4, 4))
        adrt_out = adrt.adrt(a)
        py_out = py_interp_to_cart(adrt_out)
        native_out = adrt.utils.interp_to_cart(adrt_out)
        assert np.allclose(py_out, native_out)
        assert native_out.dtype == a.dtype
        assert native_out.shape == (4, 16)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_interp_matches_small_input_batch(self, dtype):
        a = np.arange(48, dtype=dtype).reshape((3, 4, 4))
        adrt_out = adrt.adrt(a)
        batched_out = adrt.utils.interp_to_cart(adrt_out)
        stacked_out = np.stack([adrt.utils.interp_to_cart(b) for b in adrt_out])
        assert np.all(batched_out == stacked_out)
        assert stacked_out.dtype == batched_out.dtype
        assert stacked_out.shape == batched_out.shape

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_interp_preserves_zeros(self, dtype):
        a = np.zeros((16, 16), dtype=dtype)
        adrt_out = adrt.adrt(a)
        native_out = adrt.utils.interp_to_cart(adrt_out)
        assert np.all(native_out == 0)
        assert native_out.dtype == a.dtype
        assert native_out.shape == (16, 64)

    def test_refuses_small_1x1(self):
        inarr = np.ones((4, 1, 1), dtype="float64")
        with pytest.raises(ValueError, match="must have a valid ADRT output shape"):
            adrt.utils.interp_to_cart(inarr)

    def test_small_2x2(self):
        inarr = np.ones((4, 3, 2), dtype="float64")
        py_out = py_interp_to_cart(inarr)
        native_out = adrt.utils.interp_to_cart(inarr)
        assert np.allclose(py_out, native_out)
        assert native_out.shape == (2, 8)

    def test_accepts_fortran_order(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float32, order="F")
        _ = adrt.utils.interp_to_cart(inarr)

    def test_accepts_c_non_contiguous(self):
        size = 16
        inarr = np.zeros((4, 2 * (2 * size - 1), size), dtype=np.float32, order="F")
        inarr = inarr[:, ::2]
        assert inarr.shape == (4, 2 * size - 1, size)
        assert not inarr.flags["C_CONTIGUOUS"]
        _ = adrt.utils.interp_to_cart(inarr)

    def test_accepts_byteswapped(self):
        size = 16
        inarr_native = np.ones((4, 2 * size - 1, size), dtype=np.float32)
        inarr_swapped = inarr_native.astype(inarr_native.dtype.newbyteorder("S"))
        out_native = adrt.utils.interp_to_cart(inarr_native)
        out_swapped = adrt.utils.interp_to_cart(inarr_swapped)
        assert np.all(out_native == out_swapped)

    def test_refuses_int32(self):
        with pytest.raises(TypeError, match="int32"):
            adrt.utils.interp_to_cart(np.zeros((4, 2 * 8 - 1, 8), dtype=np.int32))

    def test_refuses_zero_shape(self):
        with pytest.raises(ValueError, match="found zero in dimension 2"):
            adrt.utils.interp_to_cart(np.zeros((4, 2 * 8 - 1, 0), dtype=np.float32))

    def test_refuses_zero_batch(self):
        with pytest.raises(ValueError, match="found zero in dimension 0"):
            adrt.utils.interp_to_cart(np.zeros((0, 4, 2 * 8 - 1, 8), dtype=np.float32))

    def test_refuses_mismatched_shape(self):
        with pytest.raises(ValueError, match="must have a valid ADRT output shape"):
            adrt.utils.interp_to_cart(np.zeros((4, 2 * 8 - 2, 8), dtype=np.float32))

    def test_refuses_five_dim(self):
        with pytest.raises(ValueError, match="between 3 and 4 dimensions, but had 5"):
            adrt.utils.interp_to_cart(
                np.zeros((2, 3, 4, 2 * 8 - 1, 8), dtype=np.float32)
            )

    def test_refuses_non_power_of_two(self):
        with pytest.raises(ValueError, match="must have a valid ADRT output shape"):
            adrt.utils.interp_to_cart(np.zeros((4, 2 * 7 - 1, 7), dtype=np.int32))
