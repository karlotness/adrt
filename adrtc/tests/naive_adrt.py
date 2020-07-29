import numpy as np
from math import ceil, floor


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
