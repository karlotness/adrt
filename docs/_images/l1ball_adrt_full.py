import numpy as np
import matplotlib.pyplot as plt
import adrt

n = 4
N = 2 ** n
xx = np.linspace(-1.0, 1.0, N)
X, Y = np.meshgrid(xx, xx)
Z = 1.0 * ((np.abs(X - 0.25) + np.abs(Y)) < 0.7).astype(np.float)

dZ = adrt.adrt(Z)
dZ_stitched = adrt.utils.stitch_adrt(dZ)

h1 = np.arange(2 * N - 1, -N, -1)
s1 = np.arange(-2 * N, 2 * N + 1)
H1, S1 = np.meshgrid(s1, h1)

sc = 0.8
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(sc * 12, sc * 6))
im = ax.pcolormesh(H1, S1, dZ_stitched, cmap="Blues")
fig.colorbar(im, ax=ax)
ax.set_aspect("equal")
ax.set_ylabel("$h$")
ax.set_xlabel("$s$")
fig.tight_layout()

fig.show()
