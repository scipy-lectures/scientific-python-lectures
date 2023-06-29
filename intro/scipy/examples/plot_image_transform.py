"""
============================================
Plot geometrical transformations on images
============================================

Demo geometrical transformations of images.
"""

# Load some data
import scipy as sp

face = sp.datasets.face(gray=True)

# Apply a variety of transformations
import matplotlib.pyplot as plt

shifted_face = sp.ndimage.shift(face, (50, 50))
shifted_face2 = sp.ndimage.shift(face, (50, 50), mode="nearest")
rotated_face = sp.ndimage.rotate(face, 30)
cropped_face = face[50:-50, 50:-50]
zoomed_face = sp.ndimage.zoom(face, 2)
zoomed_face.shape

plt.figure(figsize=(15, 3))
plt.subplot(151)
plt.imshow(shifted_face, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(152)
plt.imshow(shifted_face2, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(153)
plt.imshow(rotated_face, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(154)
plt.imshow(cropped_face, cmap=plt.cm.gray)
plt.axis("off")

plt.subplot(155)
plt.imshow(zoomed_face, cmap=plt.cm.gray)
plt.axis("off")

plt.subplots_adjust(wspace=0.05, left=0.01, bottom=0.01, right=0.99, top=0.99)

plt.show()
