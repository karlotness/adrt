import numpy as np
import matplotlib.pyplot as plt

n = 4
N = 2 ** n
xx = np.linspace(-1.0, 1.0, N)
X, Y = np.meshgrid(xx, xx)
Z = ((np.abs(X - 0.25) + np.abs(Y)) < 0.7).astype(np.float64)

xx1 = np.arange(N + 1)
X1, Y1 = np.meshgrid(xx1, xx1)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 4))
im = ax.pcolormesh(X1, Y1, Z, cmap="Blues")
ax.set_aspect("equal")
fig.colorbar(im, ax=ax)

fig.show()
