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
from math import ceil, floor
import numpy as np
import adrtc


def _naive_adrt(a):
    # This only works for a single image, square power of two
    assert len(a.shape) == 2  # nosec: B101
    assert a.shape[0] == a.shape[1]  # nosec: B101
    n = a.shape[0]
    niter = int(np.log2(n))
    r = np.zeros((niter+1, 4, a.shape[0], 2 * a.shape[1], a.shape[1]))

    # Copy in the image
    r[0, 0, :, :n, 0] = a
    r[0, 1, :, :n, 0] = a.T
    r[0, 2, :, :n, 0] = a[::-1].T
    r[0, 3, :, :n, 0] = a[::-1]

    # Perform the recurrence
    for i in range(1, niter + 1):
        for quad in range(4):
            for a in range(2**i):
                for y in range(0, n-2**i+1, 2**i):
                    for x in range(2*n):
                        r[i, quad, y, x, a] = \
                            r[i-1, quad, y, x, floor(a/2)] + \
                            r[i-1, quad, y + 2**(i-1), x - ceil(a/2), floor(a/2)]  # noqa

    # Copy out the result
    return np.hstack([
        r[-1, 0, 0, :n, :],
        r[-1, 1, 0, :n, ::-1][:, 1:],
        r[-1, 2, 0, :n, :][::-1, 1:],
        r[-1, 3, 0, :n, ::-1][::-1, 1:],
    ])


class TestAdrt(unittest.TestCase):
    def test_accepts_float32(self):
        inarr = np.zeros((16, 16), dtype=np.float32)
        _ = adrtc.adrt(inarr)

    def test_accepts_float64(self):
        inarr = np.zeros((16, 16), dtype=np.float64)
        _ = adrtc.adrt(inarr)

    def test_refuses_int32(self):
        inarr = np.zeros((16, 16), dtype=np.int32)
        with self.assertRaises(TypeError):
            _ = adrtc.adrt(inarr)

    def test_accepts_fortran_order(self):
        inarr = np.zeros((16, 16), dtype=np.float32, order='F')
        _ = adrtc.adrt(inarr)

    def test_all_ones_square(self):
        inarr = np.ones((64, 64))
        c_out = adrtc.adrt(inarr)
        naive_out = _naive_adrt(inarr)
        self.assertEqual(c_out.shape, naive_out.shape)
        self.assertTrue(np.allclose(c_out, naive_out))

    def test_all_zeros_square(self):
        inarr = np.zeros((64, 64))
        c_out = adrtc.adrt(inarr)
        naive_out = _naive_adrt(inarr)
        self.assertEqual(c_out.shape, naive_out.shape)
        self.assertTrue(np.allclose(c_out, naive_out))


if __name__ == '__main__':
    unittest.main()
