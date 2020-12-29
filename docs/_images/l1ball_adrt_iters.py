import numpy as np
import matplotlib.pyplot as plt
import adrt

n = 4
N = 2 ** n

xx = np.linspace(-1.0, 1.0, N)
X, Y = np.meshgrid(xx, xx)
Z = 1.0 * ((np.abs(X - 0.25) + np.abs(Y)) < 0.7).astype(np.float)

dZ = adrt.adrt(Z)

h1 = np.arange(2 * N - 1, 0, -1)
s1 = np.arange(N + 1)
H1, S1 = np.meshgrid(s1, h1)

sc = 0.9
fig, ax = plt.subplots(ncols=n, nrows=1, figsize=(sc * 14, sc * 5), sharey=True)
for k in range(0, n):
    dZ = adrt.adrt(Z, start=0, end=k + 1)
    im0 = ax[k].pcolormesh(H1, S1, dZ[2, :, :], cmap="Blues", vmin=0.0, vmax=1.0)
    ax[k].set_aspect("equal")
    ax[k].set_title("$\ell = {:d}$".format(k + 1))
    ax[k].set_xlabel("$s$")
    fig.colorbar(im0, ax=ax[k])
ax[0].set_ylabel("$h$")
fig.tight_layout()

fig.show()
