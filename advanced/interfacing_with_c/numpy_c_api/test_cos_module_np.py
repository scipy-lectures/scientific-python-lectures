import cos_module_np
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 2 * np.pi, 0.1)
y = cos_module_np.cos_func_np(x)
plt.plot(x, y)
plt.show()
