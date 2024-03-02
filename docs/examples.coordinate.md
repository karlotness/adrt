---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  remove_code_source: true
---

# The Coordinate Transform

The ADRT slope-height coordinates $(s, h) \in [0, N - 1] \times [0, 2N]$
and Cartesian angle-offset coordinates
$(\theta, t) \in [-\pi/2, \pi/2] \times [-1/\sqrt{2}, 1/\sqrt{2}]$ are
related by a simple geometric relation, depicted in the following diagram.
The diagram shows the correspondence for quadrant 3, and the transforms for the
other quadrants are derived by flipping and transposing the image. In the
Cartesian domain, the origin is taken to be the center of the image and the
image is scaled to be unit square, so the image is viewed as supported on
$[-1/2, 1/2]^2$.

```{code-cell} ipython3
:tags: [remove-cell]
import numpy as np
import matplotlib.pyplot as plt
import adrt
```

```{code-cell} ipython3
fig, ax = plt.subplots()

theta, t = (0.2*np.pi, -0.25)

ax.set_aspect(1)
xoffset, yoffset = (0.075, 0.025)

# add rectangle
ax.fill(np.array([0, 1, 1, 0, 0]) - 0.5,
        np.array([0, 0, 1, 1, 0]) - 0.5,
        color = (0, 0, 0, 0.15))

# draw auxiliary line normal to the hyperplane
z = np.array([-1.0, 1.0])
ax.plot([0, - t*np.sin(theta)],
        [0, + t*np.cos(theta)], "r")

# draw t-coordinate
ax.annotate("$t$",
        xy=(-0.5*t*np.sin(theta)+0.02, 0.5*t*np.cos(theta)))

# draw line
ax.plot(z*np.cos(theta) - t*np.sin(theta),
        z*np.sin(theta) + t*np.cos(theta), "k")

# mark origin
ax.plot(0, 0, "k.", markersize=10)

# left-intercept
x0, y0 = (-0.5, -0.5*np.tan(theta) + t/np.cos(theta))
# right-intercept
x1, y1 = \
    (0.5, (0.5 + t*np.sin(theta))/np.cos(theta)*np.sin(theta) + t*np.cos(theta))

# mark intercept
ax.plot(x0, y0, "k+")

# draw legs
ax.hlines(y0, -0.5, 0.5, "k", linestyles="--")
ax.vlines(0.5, y0, y1, "k", linestyles="--")

# display s calculation
ax.annotate(r"$\frac{s}{N-1} = \arctan(\theta)$",
            xy=(0.5 + xoffset, 0.5*(y0 + y1)),
            color="b")
ax.vlines(0.5 + 0.5*xoffset, y0, y1, "b")

ax.annotate(r"$\theta$", xy=(-0.5 + 1.5*xoffset, y0 + 0.02))
ax.annotate(r"$\frac{h}{N} = \frac{t}{\cos(\theta)} + \frac{1 + \tan(\theta)}{2}$",
            xy=(-0.5 - 8*xoffset, y0),
            color="b")

# show h-coordinates for bottom and top edges
ax.annotate("$h=0$", xy=(-0.5 - 2.5*xoffset,  0.5 - yoffset), alpha=0.5)
ax.annotate("$h=N$", xy=(-0.5 - 2.5*xoffset, -0.5 - yoffset), alpha=0.5)

ax.hlines(-0.5, -0.5 - xoffset/3, -0.5 + xoffset/3, "k")
ax.hlines( 0.5, -0.5 - xoffset/3, -0.5 + xoffset/3, "k")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
ax.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
ax.set_xlim([-1.2, 1.2])
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.grid(True);
```

To calculate the exact height (intercept) $h$ for each digital line, we
draw a line that passes through the cell centers of the left-most and right-most
entries of the digital line. We then use the line's point of intersection with the left
boundary of the image as the $h$ coordinate of that line. The
slope of that line yields the angle $\theta$ upon taking the
$\arctan$. Note that the integer $\lceil h \rceil - 1$ agrees with
the discrete height index of the digital line.

We remark here that the discrete height index is incremented in the direction
that is opposite from that used in Press[^press06] and starts from 0 at the top of
the image. This choice was made to make the height indexing agree with the 2D
array indexing adopted for this implementation.

We next illustrate the coordinate transform for all quadrants. Let us color each
entry in the four quadrants as follows.

```{code-cell} ipython3
n = 2**2
out = adrt.utils.coord_adrt(n)
angles = np.broadcast_to(out.angle, out.offset.shape)
offsets = out.offset

m = 4*(2*n - 1)*n
z = np.arange(1, m+1) / m
z_adrtshape = z.reshape(4, (2*n - 1), n)
z_stitched = adrt.utils.stitch_adrt(z_adrtshape)

fig, axs = plt.subplots(ncols=4, sharey=True)
axs[0].set_ylabel("h")
for i in range(4):
   ax = axs[i]
   ax.imshow(z_adrtshape[i, ...],
             vmin=0.0,
             vmax=1.0,
             extent=(0, n-1, 2*n-1.5, -0.5))
   ax.set_title(f"Quadrant {i + 1:d}")
   ax.set_xlabel("s");
```


In the stitched view, these would be assembled as follows.

```{code-cell} ipython3
plt.imshow(np.ma.masked_array(z_stitched, z_stitched == 0.0));
```

These entries would be mapped to the points on the Cartesian Radon domain with
the same color.

```{code-cell} ipython3
from matplotlib import cm

cmap = cm.get_cmap()
for i in range(m):
   plt.plot(angles.flatten()[i],
            offsets.flatten()[i],
            marker=".",
            color=cmap(z[i]))

plt.yticks([-0.5*np.sqrt(2), 0, 0.5*np.sqrt(2)],
           [r"-$1/\sqrt{2}$", "0", r"$1/\sqrt{2}$"])
plt.ylabel("$t$")

plt.xticks([-0.5*np.pi, -0.25*np.pi, 0, 0.25*np.pi, 0.5*np.pi],
           [r"$-\pi/2$", r"$-\pi/4$", "0", r"$\pi/4$", r"$\pi/2$"])
plt.xlabel(r"$\theta$");
```

[^press06]: William Press, *Discrete Radon transform has an exact,
    fast inverse and generalizes to operations other than sums along
    lines*, Proceedings of the National Academy of Sciences, 103.
    [doi:10.1073/pnas.0609228103](https://doi.org/10.1073/pnas.0609228103)
