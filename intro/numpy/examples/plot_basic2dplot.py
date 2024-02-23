"""
2D plotting
===========

Plot a basic 2D figure

"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
image = rng.random((30, 30))
plt.imshow(image, cmap=plt.cm.hot)
plt.colorbar()
plt.show()
