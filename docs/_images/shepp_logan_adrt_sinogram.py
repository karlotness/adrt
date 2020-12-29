import numpy as np
import matplotlib.pyplot as plt
import adrt

img = np.load('shepp-logan.npy')

N = img.shape[0]

dimg = adrt.adrt(img)
theta_cart, s_cart, dimg_cart = adrt.utils.interp_to_cart(dimg)

sc = 0.8
fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(sc*12,sc*6))
im = ax.pcolormesh(theta_cart, s_cart, dimg_cart, cmap='bone')
fig.colorbar(im, ax=ax)
ax.set_ylabel('$t$')
ax.set_xlabel(r'$\theta$')
fig.tight_layout()
fig.show()
