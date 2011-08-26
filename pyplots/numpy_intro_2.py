import numpy as np
import matplotlib.pyplot as plt

image = np.random.rand(30, 30)
plt.imshow(image)
plt.gray()
plt.show()

plt.pcolor(image)
plt.hot()
plt.colorbar()
plt.show()
