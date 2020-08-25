# Utilities for ADRT

def contiguous(A):
    r"""
    Reshape 4-channel ADRT output to zero-padded 2D continguous array 

    Parameters
    ----------
    A : array_like
        array of shape (4,2*N,N) in which N = 2**n
    
    Returns
    -------
    Z : array_like
        array of shape (3*N-2,4*N) containing a zero-padded continguous array
        with ADRT data

    """

    if not isinstance(A, np.ndarray) or (A.shape[0] != 4) \
                    or ((A.shape[1] +1) != 2*A.shape[2]):
         raise ValueError("Passed array is not of the right shape")

    dtype = A.dtype
    N = A.shape[1]
    M = A.shape[2]

    Z = np.zeros((3*M-2,4*M))

    Z[:(2*M-1), :M] = A[0,:,:]
    Z[:(2*M-1),M:(2*M)] = A[1,:,:]
    Z[(M-1):,(2*M):(3*M)] = A[2,:,:]
    Z[(M-1):,(3*M):(4*M)] = A[3,:,:]

    return Z

