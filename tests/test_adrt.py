# Copyright (C) 2020 Karl Otness, Donsub Rim
# All rights reserved
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


import unittest
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
    b = set((i + (n // 2), j) for i, j in b)
    return a.union(b)


def _naive_adrt(a):
    # This only works for a single image, square power of two
    assert len(a.shape) == 2  # nosec: B101
    assert a.shape[0] == a.shape[1]  # nosec: B101
    n = a.shape[0]
    res = np.zeros((4, 2 * a.shape[0] - 1, a.shape[1]), dtype=a.dtype)

    for quad in range(4):
        if quad == 0:
            img = a
        elif quad == 1:
            img = a.T
        elif quad == 2:
            img = np.flipud(a).T
        elif quad == 3:
            img = np.flipud(a)
        for hi, h in enumerate(reversed(range(-n + 1, n))):
            for si, s in enumerate(range(n)):
                v = 0
                for (i, j) in _gen_dline(n, h, s):
                    if i >= n or i < 0 or j >= n or j < 0:
                        # Skip values out of range
                        continue
                    v += img[i, j]
                # Store result
                res[quad, hi, si] = v
    res[1] = np.rot90(res[1], k=2)
    res[3] = np.rot90(res[3], k=2)
    return res


class TestAdrtCdefs(unittest.TestCase):
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
        with self.assertRaises(TypeError):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_non_square(self):
        inarr = np.zeros((16, 32), dtype=np.float32)
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_four_dim(self):
        inarr = np.zeros((5, 3, 16, 16), dtype=np.float32)
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_non_power_of_two(self):
        inarr = np.zeros((31, 31), dtype=np.float32)
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_non_array(self):
        with self.assertRaises(TypeError):
            _ = adrt._adrt_cdefs.adrt(None)
        with self.assertRaises(TypeError):
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
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_c_non_contiguous(self):
        inarr = np.zeros((64, 32), dtype=np.float32, order="F")[::2]
        self.assertEqual(inarr.shape, (32, 32))
        self.assertFalse(inarr.flags["C_CONTIGUOUS"])
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_byteswapped(self):
        inarr = np.ones((16, 16), dtype=np.float32).newbyteorder()
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_zero_axis_array(self):
        inarr = np.zeros((0, 32, 32), dtype=np.float32)
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.adrt(inarr)

    def test_refuses_zero_size_planes(self):
        inarr = np.zeros((0, 0), dtype=np.float32)
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.adrt(inarr)


class TestAdrt(unittest.TestCase):
    def _check_zero_stencil(self, a):
        n = a.shape[-1]
        return (
            np.all(np.tril(a[0, -n:], k=-1) == 0)
            and np.all(np.triu(a[1, :n], k=1) == 0)
            and np.all(np.tril(a[2, -n:], k=-1) == 0)
            and np.all(np.triu(a[3, :n], k=1) == 0)
        )

    def test_accepts_float32(self):
        inarr = np.zeros((16, 16), dtype=np.float32)
        _ = adrt.adrt(inarr)

    def test_accepts_float64(self):
        inarr = np.zeros((16, 16), dtype=np.float64)
        _ = adrt.adrt(inarr)

    def test_float32_returned_dtype(self):
        inarr = np.zeros((16, 16), dtype=np.float32)
        c_out = adrt.adrt(inarr)
        self.assertEqual(c_out.dtype, np.float32)

    def test_float64_returned_dtype(self):
        inarr = np.zeros((16, 16), dtype=np.float64)
        c_out = adrt.adrt(inarr)
        self.assertEqual(c_out.dtype, np.float64)

    def test_refuses_int32(self):
        inarr = np.zeros((16, 16), dtype=np.int32)
        with self.assertRaises(TypeError):
            _ = adrt.adrt(inarr)

    def test_accepts_fortran_order(self):
        inarr = np.zeros((16, 16), dtype=np.float32, order="F")
        _ = adrt.adrt(inarr)

    def test_accepts_c_non_contiguous(self):
        inarr = np.zeros((64, 32), dtype=np.float32, order="F")[::2]
        self.assertEqual(inarr.shape, (32, 32))
        self.assertFalse(inarr.flags["C_CONTIGUOUS"])
        _ = adrt.adrt(inarr)

    def test_all_ones_square(self):
        inarr = np.ones((32, 32))
        c_out = adrt.adrt(inarr)
        naive_out = _naive_adrt(inarr)
        self.assertEqual(c_out.shape, naive_out.shape)
        self.assertTrue(np.allclose(c_out, naive_out))
        self.assertTrue(self._check_zero_stencil(c_out))

    def test_vertial_line(self):
        inarr = np.zeros((32, 32))
        inarr[:, 16] = 1
        c_out = adrt.adrt(inarr)
        naive_out = _naive_adrt(inarr)
        self.assertEqual(c_out.shape, naive_out.shape)
        self.assertTrue(np.allclose(c_out, naive_out))
        self.assertTrue(self._check_zero_stencil(c_out))

    def test_all_zeros_square(self):
        inarr = np.zeros((32, 32))
        c_out = adrt.adrt(inarr)
        naive_out = _naive_adrt(inarr)
        self.assertEqual(c_out.shape, naive_out.shape)
        self.assertTrue(np.allclose(c_out, naive_out))

    def test_unique_values(self):
        size = 32
        inarr = np.arange(size ** 2).reshape((size, size)).astype("float32")
        c_out = adrt.adrt(inarr)
        naive_out = _naive_adrt(inarr)
        self.assertEqual(c_out.shape, naive_out.shape)
        self.assertTrue(np.allclose(c_out, naive_out))
        self.assertTrue(self._check_zero_stencil(c_out))

    def test_batch_dimension_unique_values(self):
        inarr = np.arange(4 * 32 * 32).reshape((4, 32, 32)).astype("float32")
        c_out = adrt.adrt(inarr)
        naive_out = np.stack([_naive_adrt(inarr[i]) for i in range(inarr.shape[0])])
        self.assertEqual(c_out.shape, naive_out.shape)
        self.assertTrue(np.allclose(c_out, naive_out))

    def test_spot_check_vertical_line_left(self):
        inarr = np.zeros((8, 8))
        inarr[:, 0] = 1
        c_out = adrt.adrt(inarr)
        # Check shape & stencil outline
        self.assertEqual(c_out.shape, (4, 2 * 8 - 1, 8))
        self.assertTrue(self._check_zero_stencil(c_out))
        # Quadrant 0
        self.assertEqual(c_out[0, 7, 0], 8)
        self.assertEqual(np.count_nonzero(c_out[0, :, 0]), 1)
        self.assertTrue(np.all(c_out[0, :7, -1] == 0))
        self.assertTrue(np.all(c_out[0, -8:, -1] == 1))
        # Quadrant 1
        self.assertTrue(np.all(c_out[1, :7, 0] == 0))
        self.assertTrue(np.all(c_out[1, -8:, 0] == 1))
        self.assertTrue(np.all(c_out[1, :7, -1] == 0))
        self.assertTrue(np.all(c_out[1, -8:, -1] == 1))
        # Quadrant 2
        self.assertTrue(np.all(c_out[2, :8, 0] == 1))
        self.assertTrue(np.all(c_out[2, -7:, 0] == 0))
        self.assertTrue(np.all(c_out[2, :8, -1] == 1))
        self.assertTrue(np.all(c_out[2, -7:, -1] == 0))
        # Quadrant 3
        self.assertTrue(np.all(c_out[3, :8, 0] == 1))
        self.assertTrue(np.all(c_out[3, -7:, 0] == 0))
        self.assertEqual(c_out[3, 7, -1], 8)
        self.assertEqual(np.count_nonzero(c_out[3, :, -1]), 1)

    def test_spot_check_vertical_line_right(self):
        inarr = np.zeros((8, 8))
        inarr[:, -1] = 1
        c_out = adrt.adrt(inarr)
        # Check shape & stencil outline
        self.assertEqual(c_out.shape, (4, 2 * 8 - 1, 8))
        self.assertTrue(self._check_zero_stencil(c_out))
        # Quadrant 0
        self.assertEqual(c_out[0, 0, 0], 8)
        self.assertEqual(np.count_nonzero(c_out[0, :, 0]), 1)
        self.assertTrue(np.all(c_out[0, :8, -1] == 1))
        self.assertTrue(np.all(c_out[0, -7:, -1] == 0))
        # Quadrant 1
        self.assertTrue(np.all(c_out[1, :8, 0] == 1))
        self.assertTrue(np.all(c_out[1, -7:, 0] == 0))
        self.assertTrue(np.all(c_out[1, :7, -1] == 0))
        self.assertTrue(np.all(c_out[1, -8:, -1] == 1))
        # Quadrant 2
        self.assertTrue(np.all(c_out[2, :8, 0] == 1))
        self.assertTrue(np.all(c_out[2, -7:, 0] == 0))
        self.assertTrue(np.all(c_out[2, :7, -1] == 0))
        self.assertTrue(np.all(c_out[2, -8:, -1] == 1))
        # Quadrant 3
        self.assertTrue(np.all(c_out[3, :7, 0] == 0))
        self.assertTrue(np.all(c_out[3, -8:, 0] == 1))
        self.assertEqual(c_out[3, -1, -1], 8)
        self.assertEqual(np.count_nonzero(c_out[3, :, -1]), 1)

    def test_spot_check_horizontal_line_top(self):
        inarr = np.zeros((8, 8))
        inarr[0, :] = 1
        c_out = adrt.adrt(inarr)
        # Check shape & stencil outline
        self.assertEqual(c_out.shape, (4, 2 * 8 - 1, 8))
        self.assertTrue(self._check_zero_stencil(c_out))
        # Quadrant 0
        self.assertTrue(np.all(c_out[0, :8, 0] == 1))
        self.assertTrue(np.all(c_out[0, -7:, 0] == 0))
        self.assertTrue(np.all(c_out[0, :8, -1] == 1))
        self.assertTrue(np.all(c_out[0, -7:, -1] == 0))
        # Quadrant 1
        self.assertTrue(np.all(c_out[1, :8, 0] == 1))
        self.assertTrue(np.all(c_out[1, -7:, 0] == 0))
        self.assertEqual(c_out[1, 7, -1], 8)
        self.assertEqual(np.count_nonzero(c_out[1, :, -1]), 1)
        # Quadrant 2
        self.assertEqual(c_out[2, 0, 0], 8)
        self.assertEqual(np.count_nonzero(c_out[2, :, 0]), 1)
        self.assertTrue(np.all(c_out[2, :8, -1] == 1))
        self.assertTrue(np.all(c_out[2, -7:, -1] == 0))
        # Quadrant 3
        self.assertTrue(np.all(c_out[3, :8, 0] == 1))
        self.assertTrue(np.all(c_out[3, -7:, 0] == 0))
        self.assertTrue(np.all(c_out[3, :7, -1] == 0))
        self.assertTrue(np.all(c_out[3, -8:, -1] == 1))

    def test_spot_check_horizontal_line_bottom(self):
        inarr = np.zeros((8, 8))
        inarr[-1, :] = 1
        c_out = adrt.adrt(inarr)
        # Check shape & stencil outline
        self.assertEqual(c_out.shape, (4, 2 * 8 - 1, 8))
        self.assertTrue(self._check_zero_stencil(c_out))
        # Quadrant 0
        self.assertTrue(np.all(c_out[0, :8, 0] == 1))
        self.assertTrue(np.all(c_out[0, -7:, 0] == 0))
        self.assertTrue(np.all(c_out[0, :7, -1] == 0))
        self.assertTrue(np.all(c_out[0, -8:, -1] == 1))
        # Quadrant 1
        self.assertTrue(np.all(c_out[1, :7, 0] == 0))
        self.assertTrue(np.all(c_out[1, -8:, 0] == 1))
        self.assertEqual(c_out[1, -1, -1], 8)
        self.assertEqual(np.count_nonzero(c_out[1, :, -1]), 1)
        # Quadrant 2
        self.assertEqual(c_out[2, 7, 0], 8)
        self.assertEqual(np.count_nonzero(c_out[2, :, 0]), 1)
        self.assertTrue(np.all(c_out[2, :7, -1] == 0))
        self.assertTrue(np.all(c_out[2, -8:, -1] == 1))
        # Quadrant 3
        self.assertTrue(np.all(c_out[3, :7, 0] == 0))
        self.assertTrue(np.all(c_out[3, -8:, 0] == 1))
        self.assertTrue(np.all(c_out[3, :7, -1] == 0))
        self.assertTrue(np.all(c_out[3, -8:, -1] == 1))


if __name__ == "__main__":
    unittest.main()
