import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import filter
from skimage import exposure

camera = data.camera()


plt.figure(figsize=(4, 4))
plt.imshow(camera, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.tight_layout()
plt.show()
