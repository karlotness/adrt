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


def _gen_dline(n, h, ds):
    if n == 1:
        return {(0, h)}
    if ds % 2 == 0:
        s = ds // 2
        a = _gen_dline(n // 2, h, s)
        b = _gen_dline(n // 2, h + s, s)
    else:
        s = (ds - 1) // 2
        a = _gen_dline(n // 2, h, s)
        b = _gen_dline(n // 2, h + s + 1, s)
    b = {(i + (n // 2), j) for i, j in b}
    return a.union(b)


class TestBdrtCdefs:
    def test_accepts_float32(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float32)
        _ = adrt._adrt_cdefs.bdrt(inarr)

    def test_accepts_float32_four_dim(self):
        size = 16
        inarr = np.zeros((5, 4, 2 * size - 1, size), dtype=np.float32)
        _ = adrt._adrt_cdefs.bdrt(inarr)

    def test_accepts_float64(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float64)
        _ = adrt._adrt_cdefs.bdrt(inarr)

    def test_accepts_float64_four_dim(self):
        size = 16
        inarr = np.zeros((5, 4, 2 * size - 1, size), dtype=np.float64)
        _ = adrt._adrt_cdefs.bdrt(inarr)

    def test_refuses_int32(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.int32)
        with pytest.raises(TypeError, match="int32"):
            _ = adrt._adrt_cdefs.bdrt(inarr)

    def test_refuses_mismatched_shape(self):
        size = 16
        inarr_a = np.zeros((4, 2 * size - 1, size - 1), dtype=np.float32)
        inarr_b = np.zeros((4, 2 * size - 2, size), dtype=np.float32)
        with pytest.raises(ValueError, match="must have a valid ADRT output shape"):
            _ = adrt._adrt_cdefs.bdrt(inarr_a)
        with pytest.raises(ValueError, match="must have a valid ADRT output shape"):
            _ = adrt._adrt_cdefs.bdrt(inarr_b)

    def test_refuses_five_dim(self):
        size = 16
        inarr = np.zeros((6, 5, 4, 2 * size - 1, size), dtype=np.float32)
        with pytest.raises(ValueError, match="between 3 and 4 dimensions, but had 5"):
            _ = adrt._adrt_cdefs.bdrt(inarr)

    def test_refuses_non_power_of_two(self):
        size = 17
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float32)
        with pytest.raises(ValueError, match="must have a valid ADRT output shape"):
            _ = adrt._adrt_cdefs.bdrt(inarr)

    def test_refuses_non_array(self):
        with pytest.raises(
            TypeError, match="must be a NumPy array or compatible subclass"
        ):
            _ = adrt._adrt_cdefs.bdrt(None)
        base_list = [[1.0, 2.0, 3.0, 4.0]] * 7
        arr_list = [base_list for _ in range(4)]
        with pytest.raises(
            TypeError, match="must be a NumPy array or compatible subclass"
        ):
            _ = adrt._adrt_cdefs.bdrt(arr_list)

    def test_refuses_fortran_order(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float32, order="F")
        with pytest.raises(ValueError, match="must be .*C-order"):
            _ = adrt._adrt_cdefs.bdrt(inarr)

    def test_refuses_c_non_contiguous(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, 2 * size), dtype=np.float32, order="F")
        inarr = inarr[:, :, ::2]
        assert inarr.shape == (4, 31, 16)
        assert not inarr.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError, match="must be .*contiguous"):
            _ = adrt._adrt_cdefs.bdrt(inarr)

    def test_refuses_byteswapped(self):
        size = 16
        inarr = np.ones(
            (4, 2 * size - 1, size), dtype=np.dtype(np.float32).newbyteorder()
        )
        with pytest.raises(ValueError, match="must be .*native byte order"):
            _ = adrt._adrt_cdefs.bdrt(inarr)

    def test_refuses_zero_axis_array(self):
        size = 16
        inarr = np.zeros((0, 4, 2 * size - 1, size), dtype=np.float32)
        with pytest.raises(ValueError, match="found zero in dimension 0"):
            _ = adrt._adrt_cdefs.bdrt(inarr)

    def test_refuses_zero_size_planes(self):
        inarr = np.zeros((4, 0, 0), dtype=np.float32)
        with pytest.raises(ValueError, match="found zero in dimension (1|2)"):
            _ = adrt._adrt_cdefs.bdrt(inarr)


class TestBdrt:
    def test_accepts_float32(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float32)
        _ = adrt.bdrt(inarr)

    def test_accepts_float64(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float64)
        _ = adrt.bdrt(inarr)

    def test_accepts_float32_returned_dtype(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float32)
        c_out = adrt.bdrt(inarr)
        assert c_out.dtype == np.float32

    def test_accepts_float64_returned_dtype(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float64)
        c_out = adrt.bdrt(inarr)
        assert c_out.dtype == np.float64

    def test_refuses_int32(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.int32)
        with pytest.raises(TypeError, match="int32"):
            _ = adrt.bdrt(inarr)

    def test_accepts_fortran_order(self):
        size = 16
        inarr = np.zeros((4, 2 * size - 1, size), dtype=np.float32, order="F")
        _ = adrt.bdrt(inarr)

    def test_accepts_c_non_contiguous(self):
        size = 16
        inarr = np.zeros((4, 2 * (2 * size - 1), size), dtype=np.float32, order="F")
        inarr = inarr[:, ::2]
        assert inarr.shape == (4, 2 * size - 1, size)
        assert not inarr.flags["C_CONTIGUOUS"]
        _ = adrt.bdrt(inarr)

    def test_accepts_byteswapped(self):
        size = 16
        inarr_native = np.ones((4, 2 * size - 1, size), dtype=np.float32)
        inarr_swapped = inarr_native.astype(inarr_native.dtype.newbyteorder("S"))
        out_native = adrt.bdrt(inarr_native)
        out_swapped = adrt.bdrt(inarr_swapped)
        assert np.all(out_native == out_swapped)

    def test_all_zeros_square(self):
        size = 16
        inarr = np.zeros((size, size), dtype=np.float32)
        adrt_out = adrt.adrt(inarr)
        bdrt_out = adrt.bdrt(adrt_out)
        bdrt_sq = np.mean(adrt.utils.truncate(bdrt_out), axis=0)
        assert bdrt_out.shape == adrt_out.shape
        assert bdrt_sq.shape == inarr.shape
        assert bdrt_sq.dtype == inarr.dtype
        assert np.all(bdrt_out == 0)

    def test_digital_lines(self):
        size = 16
        # Quadrant 0
        sh = _gen_dline(size, size - 5, 3)
        dline = np.zeros((size, size))
        for sh0 in sh:
            dline[sh0[0], sh0[1]] = 0.25
        bdrt_in = np.zeros((4, 2 * size - 1, size))
        bdrt_in[0, 4, 3] = 1.0
        bdrt_out = adrt.bdrt(bdrt_in)
        bdrt_sq = np.mean(adrt.utils.truncate(bdrt_out), axis=0)
        assert np.allclose(bdrt_sq, dline)
        # Quadrant 1
        sh = _gen_dline(size, size - 5, 3)
        dline = np.zeros((size, size))
        for sh0 in sh:
            dline[sh0[1], sh0[0]] = 0.25
        bdrt_in = np.zeros((4, 2 * size - 1, size))
        bdrt_in[1, 4, 3] = 1.0
        bdrt_out = adrt.bdrt(bdrt_in)
        bdrt_sq = np.mean(adrt.utils.truncate(bdrt_out), axis=0)
        assert np.allclose(bdrt_sq, dline)
        # Quadrant 2
        sh = _gen_dline(size, size - 4, 3)
        dline = np.zeros((size, size))
        for sh0 in sh:
            dline[size - sh0[1], sh0[0]] = 0.25
        bdrt_in = np.zeros((4, 2 * size - 1, size))
        bdrt_in[2, 4, 3] = 1.0
        bdrt_out = adrt.bdrt(bdrt_in)
        bdrt_sq = np.mean(adrt.utils.truncate(bdrt_out), axis=0)
        assert np.allclose(bdrt_sq, dline)
        # Quadrant 3
        sh = _gen_dline(size, 2, 3)
        dline = np.zeros((size, size))
        for sh0 in sh:
            dline[sh0[0], size - sh0[1]] = 0.25
        bdrt_in = np.zeros((4, 2 * size - 1, size))
        bdrt_in[3, 4, 3] = 1.0
        bdrt_out = adrt.bdrt(bdrt_in)
        bdrt_sq = np.mean(adrt.utils.truncate(bdrt_out), axis=0)
        assert np.allclose(bdrt_sq, dline)

    def test_digital_line_rotations(self):
        size = 16
        bdrt_in = np.zeros((4, 4, 2 * size - 1, size))
        h, s = 3, 6
        for i in range(4):
            bdrt_in[i, i, h, s] = 1.0
        bdrt_out = adrt.bdrt(bdrt_in)
        bdrt_sq = np.mean(adrt.utils.truncate(bdrt_out), axis=0)
        assert np.allclose(np.rot90(bdrt_sq[0, ...], k=1), bdrt_sq[2, ...])
        assert np.allclose(np.fliplr(np.rot90(bdrt_sq[1, ...], k=2)), bdrt_sq[2, ...])
        assert np.allclose(np.fliplr(np.rot90(bdrt_sq[3, ...], k=1)), bdrt_sq[2, ...])

    def test_all_ones(self):
        size = 16
        bdrt_in = np.ones((4, 4, 2 * size - 1, size)) / size
        bdrt_out = adrt.bdrt(bdrt_in)
        bdrt_sq = np.mean(adrt.utils.truncate(bdrt_out), axis=0)
        assert np.allclose(bdrt_sq.min(), 1.0)
        assert np.allclose(bdrt_sq.max(), 1.0)

    def test_backprojected_delta_levels(self):
        size = 16
        adrt_in = np.zeros((size, size))
        adrt_in[size // 2, size // 2] = 1.0
        adrt_out = adrt.adrt(adrt_in)
        bdrt_out = adrt.bdrt(adrt_out)
        bdrt_sq = np.mean(adrt.utils.truncate(bdrt_out), axis=0)
        assert np.allclose(1.0 * (bdrt_sq == 16.0), adrt_in)
        assert np.sum(bdrt_sq == 4) == 8
        assert np.sum(bdrt_sq == 2) == 19

    def test_bdrt_zeros(self):
        size = 16
        adrt_in = np.ones((size, size))
        adrt_out = adrt.adrt(adrt_in)
        bdrt_out = adrt.bdrt(adrt_out)
        for i in range(4):
            zero_part = np.tril(bdrt_out[i, size:, ::-1])
            assert np.linalg.norm(zero_part, ord="fro") == 0.0

    def test_small_1x1(self):
        inarr = np.ones((4, 1, 1), dtype="float64")
        expected_out = np.ones((1, 1), dtype="float64")
        c_out = adrt.bdrt(inarr)
        c_out = np.mean(adrt.utils.truncate(c_out), axis=0)
        assert c_out.shape == expected_out.shape
        assert np.allclose(c_out, expected_out)

    def test_small_1x1_batch(self):
        expected_out = np.arange(5, dtype="float64").reshape((5, 1, 1)) + 1
        inarr = np.stack([expected_out] * 4, axis=1)
        c_out = adrt.bdrt(inarr)
        c_out = np.mean(adrt.utils.truncate(c_out), axis=1)
        assert c_out.shape == expected_out.shape
        assert np.allclose(c_out, expected_out)

    @pytest.mark.parametrize("size", [1, 2, 4, 8])
    def test_materialize_array(self, size):
        n = size
        full_size = n**2
        adrt_size = 4 * n * (2 * n - 1)
        adrt_in = np.eye(full_size, dtype=np.float32).reshape((full_size, n, n))
        bdrt_in = np.eye(adrt_size, dtype=np.float32).reshape(
            (adrt_size, 4, 2 * n - 1, n)
        )
        adrt_arr = adrt.adrt(adrt_in).reshape((full_size, -1)).T
        bdrt_out = adrt.bdrt(bdrt_in)
        bdrt_arr = np.sum(adrt.utils.truncate(bdrt_out), axis=1).reshape(
            (adrt_size, full_size)
        )
        assert adrt_arr.shape == bdrt_arr.shape
        assert np.allclose(adrt_arr, bdrt_arr)
