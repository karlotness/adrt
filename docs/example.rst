Examples
========

.. plot::
   :include-source: true

   data = np.ones((32, 32))
   raw_adrt = adrt.adrt(data)
   stitched_output = adrt.utils.stitch_adrt(raw_adrt)
   plt.imshow(stitched_output)
   plt.xlabel("Angle")
   plt.ylabel("Displacement")
   plt.colorbar()
