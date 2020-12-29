import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

url = 'https://upload.wikimedia.org/wikipedia/commons/e/e5/Shepp_logan.png'
response = requests.get(url)
image_data = Image.open(BytesIO(response.content))

img = np.array(image_data)
img = np.array(img)
img = np.flipud(img)
img = img.astype(np.float)

np.save('shepp-logan.npy', img)

fig, ax = plt.subplots(ncols=1,nrows=1)
im = ax.pcolormesh(img,cmap="bone")
fig.colorbar(im, ax=ax)
ax.set_aspect("equal")
fig.show()

