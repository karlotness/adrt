# Utilities for ADRT

import numpy as np


def contiguous(a):
    r"""
    Reshape 4-channel ADRT output to zero-padded 2D continguous array

    Parameters
    ----------
    a : array_like
        array of shape (4,2*N,N) in which N = 2**n

    Returns
    -------
    Z : array_like
        array of shape (3*N-2,4*N) containing a zero-padded continguous array
        with ADRT data

    """

    if (
        not isinstance(a, np.ndarray)
        or (a.shape[0] != 4)
        or ((a.shape[1] + 1) != 2 * a.shape[2])
    ):
        raise ValueError("Passed array is not of the right shape")

    dtype = a.dtype
    m = a.shape[2]

    z = np.zeros((3 * m - 2, 4 * m), dtype=dtype)

    z[: (2 * m - 1), :m] = a[0, :, :]
    z[: (2 * m - 1), m : (2 * m)] = a[1, :, :]
    z[(m - 1) :, (2 * m) : (3 * m)] = a[2, :, :]
    z[(m - 1) :, (3 * m) : (4 * m)] = a[3, :, :]

    return z
