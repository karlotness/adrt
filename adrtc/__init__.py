import numpy as np
from . import _adrtc_cdefs


def adrt(a):
    native_dtype = a.dtype.newbyteorder('=')
    a = np.require(a, dtype=native_dtype,
                   requirements=['C_CONTIGUOUS', 'ALIGNED'])
    return _adrtc_cdefs.adrt(a)
