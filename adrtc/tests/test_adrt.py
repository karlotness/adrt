import unittest
import numpy as np
from adrtc.tests import naive_adrt
import adrtc


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
        naive_out = naive_adrt._naive_adrt(inarr)
        self.assertEqual(c_out.shape, naive_out.shape)
        self.assertTrue(np.allclose(c_out, naive_out))


if __name__ == '__main__':
    unittest.main()
