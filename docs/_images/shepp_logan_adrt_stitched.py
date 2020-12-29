import numpy as np
import matplotlib.pyplot as plt
import adrt

img = np.load('shepp-logan.npy')

N = img.shape[0]

dimg = adrt.adrt(img)
dimg_stitched = adrt.utils.stitch_adrt(dimg)

h1 = np.arange(2*N-1,-N,-1) 
s1 = np.arange(-2*N,2*N+1)
H1,S1 = np.meshgrid(s1,h1)

sc = 0.8
fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(sc*12,sc*6))
im = ax.pcolormesh(H1,S1,dimg_stitched, cmap='bone')
fig.colorbar(im, ax=ax)
ax.set_aspect('equal')
ax.set_ylabel('$h$')
ax.set_xlabel('$s$')
fig.tight_layout()

fig.show()
