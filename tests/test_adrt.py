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


def _naive_adrt(a):
    # This only works for a single image, square power of two
    assert len(a.shape) == 2  # nosec: B101
    assert a.shape[0] == a.shape[1]  # nosec: B101
    n = a.shape[0]
    res = np.zeros((4, 2 * a.shape[0] - 1, a.shape[1]), dtype=a.dtype)
    imgs = [a, a.T, np.flipud(a).T, np.flipud(a)]
    for quad, img in enumerate(imgs):
        for hi, h in enumerate(reversed(range(-n + 1, n))):
            for si, s in enumerate(range(n)):
                v = 0
                for i, j in _gen_dline(n, h, s):
                    if 0 <= i < n and 0 <= j < n:
                        # Add values that are in bounds
                        v += img[i, j]
                # Store result
                res[quad, hi, si] = v
    return res


def make_unaligned(a):
    assert a.dtype.alignment >= 2, "One-byte dtypes cannot be made unaligned"
    b = np.frombuffer(b"\x00" + a.tobytes("C"), dtype=a.dtype, offset=1).reshape(
        a.shape
    )
    assert not b.flags.aligned
    assert np.all(b == a)
    return b


class TestAdrtCdefs:
    def test_accepts_float32(self):
        inarr = np.zeros((16, 16), dtype=np.float32)
        _ = adrt._adrt_cdefs.adrt(inarr)

    def test_accepts_float32_three_dim(self):
        inarr = np.zeros((5, 16, 16), dtype=np.float32)
        _ = adrt._adrt_cdefs.adrt(inarr)

    def test_accepts_float64(self):
        inarr = np.zeros((16, 16), dtype=np.float64)
        _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_int32(self):
        inarr = np.zeros((16, 16), dtype=np.int32)
        with pytest.raises(TypeError, match="int32"):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_non_square(self):
        inarr = np.zeros((16, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="must be square"):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_four_dim(self):
        inarr = np.zeros((5, 3, 16, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="between 2 and 3 dimensions, but had 4"):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_non_power_of_two(self):
        inarr = np.zeros((31, 31), dtype=np.float32)
        with pytest.raises(ValueError, match="power of two shape"):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_non_array(self):
        with pytest.raises(
            TypeError, match="must be a NumPy array or compatible subclass"
        ):
            _ = adrt._adrt_cdefs.adrt(None)
        with pytest.raises(
            TypeError, match="must be a NumPy array or compatible subclass"
        ):
            _ = adrt._adrt_cdefs.adrt(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [1.0, 2.0, 3.0, 4.0],
                ]
            )

    def test_refuses_fortran_order(self):
        inarr = np.zeros((32, 32), dtype=np.float32, order="F")
        with pytest.raises(ValueError, match="must be .*C-order"):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_c_non_contiguous(self):
        inarr = np.zeros((64, 32), dtype=np.float32, order="F")[::2]
        assert inarr.shape == (32, 32)
        assert not inarr.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError, match="must be .*contiguous"):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_byteswapped(self):
        inarr = np.ones((16, 16), dtype=np.dtype(np.float32).newbyteorder())
        with pytest.raises(ValueError, match="must be .*native byte order"):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_zero_axis_array(self):
        inarr = np.zeros((0, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="found zero in dimension 0"):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_zero_size_planes(self):
        inarr = np.zeros((0, 0), dtype=np.float32)
        with pytest.raises(ValueError, match="found zero in dimension (0|1)"):
            _ = adrt._adrt_cdefs.adrt(inarr)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_refuses_unaligned(self, dtype):
        inarr = make_unaligned(np.zeros((16, 16), dtype=dtype))
        with pytest.raises(ValueError, match="must be .*aligned"):
            _ = adrt._adrt_cdefs.adrt(inarr)


class TestAdrt:
    def _check_zero_stencil(self, a):
        n = a.shape[-1]
        assert np.all(
            np.tril(a[0, -n:], k=-1) == 0
        ), "Non-zero value in quadrant 0 triangle"
        assert np.all(
            np.tril(a[1, -n:], k=-1) == 0
        ), "Non-zero value in quadrant 1 triangle"
        assert np.all(
            np.tril(a[2, -n:], k=-1) == 0
        ), "Non-zero value in quadrant 2 triangle"
        assert np.all(
            np.tril(a[3, -n:], k=-1) == 0
        ), "Non-zero value in quadrant 3 triangle"

    def test_accepts_float32(self):
        inarr = np.zeros((16, 16), dtype=np.float32)
        _ = adrt.adrt(inarr)

    def test_accepts_float64(self):
        inarr = np.zeros((16, 16), dtype=np.float64)
        _ = adrt.adrt(inarr)

    def test_float32_returned_dtype(self):
        inarr = np.zeros((16, 16), dtype=np.float32)
        c_out = adrt.adrt(inarr)
        assert c_out.dtype == np.float32

    def test_float64_returned_dtype(self):
        inarr = np.zeros((16, 16), dtype=np.float64)
        c_out = adrt.adrt(inarr)
        assert c_out.dtype == np.float64

    def test_refuses_int32(self):
        inarr = np.zeros((16, 16), dtype=np.int32)
        with pytest.raises(TypeError, match="int32"):
            _ = adrt.adrt(inarr)

    def test_accepts_fortran_order(self):
        inarr = np.zeros((16, 16), dtype=np.float32, order="F")
        _ = adrt.adrt(inarr)

    def test_accepts_c_non_contiguous(self):
        inarr = np.zeros((64, 32), dtype=np.float32, order="F")[::2]
        assert inarr.shape == (32, 32)
        assert not inarr.flags["C_CONTIGUOUS"]
        _ = adrt.adrt(inarr)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_accepts_unaligned(self, dtype):
        base_arr = np.arange(16 * 16, dtype=dtype).reshape((16, 16))
        unaligned_arr = make_unaligned(base_arr)
        base_out = adrt.adrt(base_arr)
        unaligned_out = adrt.adrt(unaligned_arr)
        assert np.all(base_out == unaligned_out)

    def test_accepts_byteswapped(self):
        inarr_native = np.ones((16, 16), dtype=np.float32)
        inarr_swapped = inarr_native.astype(inarr_native.dtype.newbyteorder("S"))
        out_native = adrt.adrt(inarr_native)
        out_swapped = adrt.adrt(inarr_swapped)
        assert np.all(out_native == out_swapped)

    def test_all_ones_square(self):
        inarr = np.ones((16, 16))
        c_out = adrt.adrt(inarr)
        naive_out = _naive_adrt(inarr)
        assert c_out.shape == naive_out.shape
        assert np.allclose(c_out, naive_out)
        self._check_zero_stencil(c_out)

    def test_vertical_line(self):
        inarr = np.zeros((16, 16))
        inarr[:, 8] = 1
        c_out = adrt.adrt(inarr)
        naive_out = _naive_adrt(inarr)
        assert c_out.shape == naive_out.shape
        assert np.allclose(c_out, naive_out)
        self._check_zero_stencil(c_out)

    def test_all_zeros_square(self):
        inarr = np.zeros((32, 32))
        c_out = adrt.adrt(inarr)
        assert c_out.shape == (4, 2 * 32 - 1, 32)
        assert np.all(c_out == 0)

    def test_unique_values(self):
        size = 16
        inarr = np.arange(size**2).reshape((size, size)).astype("float32")
        c_out = adrt.adrt(inarr)
        naive_out = _naive_adrt(inarr)
        assert c_out.shape == naive_out.shape
        assert np.allclose(c_out, naive_out)
        self._check_zero_stencil(c_out)

    def test_batch_dimension_unique_values(self):
        inarr = np.arange(4 * 8 * 8).reshape((4, 8, 8)).astype("float32")
        c_out = adrt.adrt(inarr)
        naive_out = np.stack([_naive_adrt(inarr[i]) for i in range(inarr.shape[0])])
        assert c_out.shape == naive_out.shape
        assert np.allclose(c_out, naive_out)

    def test_small_1x1(self):
        inarr = np.ones((1, 1), dtype="float64")
        expected_out = np.ones((4, 1, 1), dtype="float64")
        c_out = adrt.adrt(inarr)
        assert c_out.shape == expected_out.shape
        assert np.allclose(c_out, expected_out)

    def test_small_1x1_batch(self):
        inarr = np.arange(5, dtype="float64").reshape((5, 1, 1)) + 1
        expected_out = np.stack([inarr] * 4, axis=1)
        c_out = adrt.adrt(inarr)
        assert c_out.shape == expected_out.shape
        assert np.allclose(c_out, expected_out)

    def test_spot_check_vertical_line_left(self):
        inarr = np.zeros((8, 8))
        inarr[:, 0] = 1
        c_out = adrt.adrt(inarr)
        # Check shape & stencil outline
        assert c_out.shape == (4, 2 * 8 - 1, 8)
        self._check_zero_stencil(c_out)
        # Quadrant 0
        assert c_out[0, 7, 0] == 8
        assert np.count_nonzero(c_out[0, :, 0]) == 1
        assert np.all(c_out[0, :7, -1] == 0)
        assert np.all(c_out[0, -8:, -1] == 1)
        # Quadrant 1
        assert np.all(c_out[1, 8:, 0] == 0)
        assert np.all(c_out[1, :8, -1] == 1)
        assert np.all(c_out[1, 8:, 0] == 0)
        assert np.all(c_out[1, :8, 0] == 1)
        # Quadrant 2
        assert np.all(c_out[2, :8, 0] == 1)
        assert np.all(c_out[2, -7:, 0] == 0)
        assert np.all(c_out[2, :8, -1] == 1)
        assert np.all(c_out[2, -7:, -1] == 0)
        # Quadrant 3
        assert np.all(c_out[3, 8:, -1] == 1)
        assert np.all(c_out[3, :7, -1] == 0)
        assert c_out[3, 7, 0] == 8
        assert np.count_nonzero(c_out[3, :, 0]) == 1

    def test_spot_check_vertical_line_right(self):
        inarr = np.zeros((8, 8))
        inarr[:, -1] = 1
        c_out = adrt.adrt(inarr)
        # Check shape & stencil outline
        assert c_out.shape == (4, 2 * 8 - 1, 8)
        self._check_zero_stencil(c_out)
        # Quadrant 0
        assert c_out[0, 0, 0] == 8
        assert np.count_nonzero(c_out[0, :, 0]) == 1
        assert np.all(c_out[0, :8, -1] == 1)
        assert np.all(c_out[0, -7:, -1] == 0)
        # Quadrant 1
        assert np.all(c_out[1, :8, 0] == 1)
        assert np.all(c_out[1, -7:, 0] == 0)
        assert np.all(c_out[1, :7, -1] == 0)
        assert np.all(c_out[1, -8:, -1] == 1)
        # Quadrant 2
        assert np.all(c_out[2, :8, 0] == 1)
        assert np.all(c_out[2, -7:, 0] == 0)
        assert np.all(c_out[2, :7, -1] == 0)
        assert np.all(c_out[2, -8:, -1] == 1)
        # Quadrant 3
        assert np.all(c_out[3, 8:, -1] == 0)
        assert np.all(c_out[3, :8, -1] == 1)
        assert c_out[3, 0, 0] == 8
        assert np.count_nonzero(c_out[3, :, 0]) == 1

    def test_spot_check_horizontal_line_top(self):
        inarr = np.zeros((8, 8))
        inarr[0, :] = 1
        c_out = adrt.adrt(inarr)
        # Check shape & stencil outline
        assert c_out.shape == (4, 2 * 8 - 1, 8)
        self._check_zero_stencil(c_out)
        # Quadrant 0
        assert np.all(c_out[0, :8, 0] == 1)
        assert np.all(c_out[0, -7:, 0] == 0)
        assert np.all(c_out[0, :8, -1] == 1)
        assert np.all(c_out[0, -7:, -1] == 0)
        # Quadrant 1
        assert np.all(c_out[1, 7:, 7] == 1)
        assert np.all(c_out[1, :7, 0] == 0)
        assert c_out[1, 7, 0] == 8
        assert np.count_nonzero(c_out[1, :, 0]) == 1
        # Quadrant 2
        assert c_out[2, 0, 0] == 8
        assert np.count_nonzero(c_out[2, :, 0]) == 1
        assert np.all(c_out[2, :8, -1] == 1)
        assert np.all(c_out[2, -7:, -1] == 0)
        # Quadrant 3
        assert np.all(c_out[3, :8, 0] == 1)
        assert np.all(c_out[3, -7:, 0] == 0)
        assert np.all(c_out[3, -7:, 0] == 0)
        assert np.all(c_out[3, :8, 0] == 1)

    def test_spot_check_horizontal_line_bottom(self):
        inarr = np.zeros((8, 8))
        inarr[-1, :] = 1
        c_out = adrt.adrt(inarr)
        # Check shape & stencil outline
        assert c_out.shape == (4, 2 * 8 - 1, 8)
        self._check_zero_stencil(c_out)
        # Quadrant 0
        assert np.all(c_out[0, :8, 0] == 1)
        assert np.all(c_out[0, -7:, 0] == 0)
        assert np.all(c_out[0, :7, -1] == 0)
        assert np.all(c_out[0, -8:, -1] == 1)
        # Quadrant 1
        assert np.all(c_out[1, 8:, 0] == 0)
        assert np.all(c_out[1, :8, 7] == 1)
        assert c_out[1, 0, 0] == 8
        assert np.count_nonzero(c_out[1, :, 0]) == 1
        # Quadrant 2
        assert c_out[2, 7, 0] == 8
        assert np.count_nonzero(c_out[2, :, 0]) == 1
        assert np.all(c_out[2, :7, -1] == 0)
        assert np.all(c_out[2, -8:, -1] == 1)
        # Quadrant 3
        assert np.all(c_out[3, 8:, 0] == 0)
        assert np.all(c_out[3, :8, 0] == 1)
        assert np.all(c_out[3, 8:, -1] == 0)
        assert np.all(c_out[3, :8, -1] == 1)
