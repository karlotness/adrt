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


class TestIAdrtCdefs(unittest.TestCase):
    def test_accepts_float32(self):
        inarr = np.zeros((16, 4 * 16), dtype=np.float32)
        _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_accepts_float32_three_dim(self):
        inarr = np.zeros((5, 16, 4 * 16), dtype=np.float32)
        _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_accepts_float64(self):
        inarr = np.zeros((16, 4 * 16), dtype=np.float64)
        _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_int32(self):
        inarr = np.zeros((16, 4 * 16), dtype=np.int32)
        with self.assertRaises(TypeError):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_mismatched_shape(self):
        inarr = np.zeros((16, 4 * 32), dtype=np.float32)
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_four_dim(self):
        inarr = np.zeros((5, 3, 16, 4 * 16), dtype=np.float32)
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_non_power_of_two(self):
        inarr = np.zeros((31, 4 * 31), dtype=np.float32)
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_non_array(self):
        with self.assertRaises(TypeError):
            _ = adrt._adrt_cdefs.iadrt(None)
        with self.assertRaises(TypeError):
            _ = adrt._adrt_cdefs.iadrt([[1., 2., 3., 4.],
                                        [1., 2., 3., 4.],
                                        [1., 2., 3., 4.],
                                        [1., 2., 3., 4.]])

    def test_refuses_fortran_order(self):
        inarr = np.zeros((32, 4 * 32), dtype=np.float32, order='F')
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_c_non_contiguous(self):
        inarr = np.zeros((2 * 32, 4 * 32), dtype=np.float32, order='F')[::2]
        self.assertEqual(inarr.shape, (32, 4 * 32))
        self.assertFalse(inarr.flags['C_CONTIGUOUS'])
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_byteswapped(self):
        inarr = np.ones((16, 4 * 16), dtype=np.float32).newbyteorder()
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_zero_axis_array(self):
        inarr = np.zeros((0, 32, 4 * 32), dtype=np.float32)
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.iadrt(inarr)

    def test_refuses_zero_size_planes(self):
        inarr = np.zeros((0, 0), dtype=np.float32)
        with self.assertRaises(ValueError):
            _ = adrt._adrt_cdefs.iadrt(inarr)


if __name__ == '__main__':
    unittest.main()
