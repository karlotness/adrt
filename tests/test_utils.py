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


class TestStitchAdrt(unittest.TestCase):
    def _check_column_continuous(self, stitched):
        n = stitched.shape[-1] // 4
        self.assertTrue(np.allclose(stitched[..., :, n - 1], stitched[..., :, n]))
        # Middle seam requires special treatment
        self.assertTrue(
            np.allclose(
                stitched[..., :, 2 * n - 1],
                np.roll(stitched[..., :, 2 * n], -1 * (n - 1), axis=-1),
            )
        )
        self.assertTrue(
            np.allclose(stitched[..., :, 3 * n - 1], stitched[..., :, 3 * n])
        )
        self.assertTrue(
            np.allclose(stitched[..., :, -1], np.flip(stitched[..., :, 0], axis=-1))
        )

    def test_accepts_adrt_output(self):
        n = 16
        inarr = np.arange(n ** 2).reshape((n, n)).astype("float32")
        out = adrt.adrt(inarr)
        stitched = adrt.utils.stitch_adrt(out)
        self.assertEqual(stitched.shape, (4 * n - 3, 4 * n))
        # Check output columns are contiguous
        self._check_column_continuous(stitched)
        # Check quadrant ordering
        self.assertTrue(np.allclose(stitched[:2 * n - 1, :n], out[0]))
        self.assertTrue(np.allclose(stitched[:2 * n - 1, n : 2 * n], out[1]))
        self.assertTrue(np.allclose(stitched[-2 * n + 1:, 2 * n : 3 * n], out[2]))
        self.assertTrue(np.allclose(stitched[-2 * n + 1:, 3 * n:], out[3]))

    def test_accepts_adrt_output_batched(self):
        n = 16
        inarr = np.arange(3 * (n ** 2)).reshape((3, n, n)).astype("float32")
        out = adrt.adrt(inarr)
        stitched = adrt.utils.stitch_adrt(out)
        self.assertEqual(stitched.shape, (3, 4 * n - 3, 4 * n))
        # Check output columns are contiguous
        self._check_column_continuous(stitched)
        # Check quadrant ordering
        self.assertTrue(np.allclose(stitched[:, :2 * n - 1, :n], out[:, 0]))
        self.assertTrue(np.allclose(stitched[:, :2 * n - 1, n : 2 * n], out[:, 1]))
        self.assertTrue(np.allclose(stitched[:, -2 * n + 1:, 2 * n : 3 * n], out[:, 2]))
        self.assertTrue(np.allclose(stitched[:, -2 * n + 1:, 3 * n:], out[:, 3]))

    def test_accepts_adrt_output_remove_repeated(self):
        n = 16
        inarr = np.arange(n ** 2).reshape((n, n)).astype("float32")
        out = adrt.adrt(inarr)
        stitched = adrt.utils.stitch_adrt(out, remove_repeated=True)
        self.assertEqual(stitched.shape, (4 * n - 3, 4 * n - 4))
        # Check deleting repeated columns
        stitch_repeat = adrt.utils.stitch_adrt(out, remove_repeated=False)
        stitch_repeat = np.delete(stitch_repeat, [i * n - 1 for i in range(4)], axis=-1)
        self.assertEqual(stitched.shape, stitch_repeat.shape)
        self.assertTrue(np.allclose(stitched, stitch_repeat))

    def test_accepts_adrt_output_remove_repeated_batched(self):
        n = 16
        inarr = np.arange(3 * (n ** 2)).reshape((3, n, n)).astype("float32")
        out = adrt.adrt(inarr)
        stitched = adrt.utils.stitch_adrt(out, remove_repeated=True)
        self.assertEqual(stitched.shape, (3, 4 * n - 3, 4 * n - 4))
        # Check deleting repeated columns
        stitch_repeat = adrt.utils.stitch_adrt(out, remove_repeated=False)
        stitch_repeat = np.delete(stitch_repeat, [i * n - 1 for i in range(4)], axis=-1)
        self.assertEqual(stitched.shape, stitch_repeat.shape)
        self.assertTrue(np.allclose(stitched, stitch_repeat))


if __name__ == "__main__":
    unittest.main()
