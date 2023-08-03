"""
Geometrical transformations
==============================

This examples demos some simple geometrical transformations on a Raccoon face.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

face = sp.datasets.face(gray=True)
lx, ly = face.shape
# Cropping
crop_face = face[lx // 4 : -lx // 4, ly // 4 : -ly // 4]
# up <-> down flip
flip_ud_face = np.flipud(face)
# rotation
rotate_face = sp.ndimage.rotate(face, 45)
rotate_face_noreshape = sp.ndimage.rotate(face, 45, reshape=False)

plt.figure(figsize=(12.5, 2.5))


plt.subplot(151)
plt.imshow(face, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(152)
plt.imshow(crop_face, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(153)
plt.imshow(flip_ud_face, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(154)
plt.imshow(rotate_face, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(155)
plt.imshow(rotate_face_noreshape, cmap=plt.cm.gray)
plt.axis("off")

plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)

plt.show()
