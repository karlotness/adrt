import numpy as np
import adrt


def iter_last(iterable):
    for _a in iterable:
        pass
    return _a


def test_match_adrt_all_ones():
    inarr = np.ones((16, 16))
    c_out = adrt.adrt(inarr)
    last = iter_last(adrt.core.adrt_iter(inarr))
    assert np.allclose(last, c_out)
    assert last.shape == c_out.shape
    assert last.dtype == c_out.dtype


def test_match_adrt_unique_values():
    size = 16
    inarr = np.arange(size ** 2).reshape((size, size))
    c_out = adrt.adrt(inarr)
    last = iter_last(adrt.core.adrt_iter(inarr))
    assert np.allclose(last, c_out)
    assert last.shape == c_out.shape
    assert last.dtype == c_out.dtype
