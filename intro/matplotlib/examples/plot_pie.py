"""
Pie chart
=========

A simple pie chart example with matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

n = 20
Z = np.ones(n)
Z[-1] *= 2

plt.axes([0.025, 0.025, 0.95, 0.95])

plt.pie(Z, explode=Z * 0.05, colors=[f"{i / float(n):f}" for i in range(n)])
plt.axis("equal")
plt.xticks([])
plt.yticks()

plt.show()
