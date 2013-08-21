import numpy as np
import matplotlib.pyplot as plt

check = np.zeros((9, 9))
check[::2, 1::2] = 1
check[1::2, ::2] = 1
plt.matshow(check, cmap='gray')
plt.show()
