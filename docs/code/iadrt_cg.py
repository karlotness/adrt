import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
import adrt


class ADRTNormalOperator(LinearOperator):
    def __init__(self, img_size, dtype=None):
        super().__init__(dtype=dtype, shape=(img_size**2, img_size**2))
        self._img_size = img_size

    def _matmat(self, x):
        # Use batch dimensions to handle columns of matrix x
        n_batch = x.shape[-1]
        batch_img = np.moveaxis(x, -1, 0).reshape(
            (n_batch, self._img_size, self._img_size)
        )
        ret = adrt.utils.truncate(adrt.bdrt(adrt.adrt(batch_img))).mean(axis=1)
        return np.moveaxis(ret, 0, -1).reshape((self._img_size**2, n_batch))

    def _adjoint(self):
        return self


def iadrt_cg(b, /, *, op_cls=ADRTNormalOperator, **kwargs):
    if b.ndim > 3:
        raise ValueError("batch dimension not supported for iadrt_cg")
    img_size = b.shape[-1]
    linop = op_cls(img_size=img_size, dtype=b.dtype)
    tb = adrt.utils.truncate(adrt.bdrt(b)).mean(axis=0).ravel()
    x, info = cg(linop, tb, **kwargs)
    if info != 0:
        raise ValueError(f"convergence failed (cg status {info})")
    return x.reshape((img_size, img_size))
