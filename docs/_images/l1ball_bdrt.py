import numpy as np
import matplotlib.pyplot as plt
import adrt

n = 4
N = 2**n 
xx = np.linspace(-1.0,1.0,N)
X,Y = np.meshgrid(xx,xx)
Z = 1.0*((np.abs(X-0.25) + np.abs(Y)) < 0.7).astype(np.float)

xx1 = np.arange(N+1)
X1,Y1 = np.meshgrid(xx1,xx1)

dZ = adrt.adrt(Z)
bZ = adrt.bdrt(dZ)
tZ = adrt.utils.truncate(bZ)
mZ = np.mean(tZ,axis=0)

xx1 = np.arange(N+1)
X1,Y1 = np.meshgrid(xx1,xx1)

fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(5,4))
im = ax.pcolormesh(X1,Y1,mZ,cmap='Blues')
fig.colorbar(im,ax=ax)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_title('Back-projected image')
ax.set_aspect('equal');

fig.show()

