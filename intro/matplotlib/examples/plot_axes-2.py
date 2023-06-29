"""
Axes
====

This example shows various axes command to position matplotlib axes.

"""

import matplotlib.pyplot as plt

plt.axes([0.1, 0.1, 0.5, 0.5])
plt.xticks([])
plt.yticks([])
plt.text(
    0.1, 0.1, "axes([0.1, 0.1, 0.5, 0.5])", ha="left", va="center", size=16, alpha=0.5
)

plt.axes([0.2, 0.2, 0.5, 0.5])
plt.xticks([])
plt.yticks([])
plt.text(
    0.1, 0.1, "axes([0.2, 0.2, 0.5, 0.5])", ha="left", va="center", size=16, alpha=0.5
)

plt.axes([0.3, 0.3, 0.5, 0.5])
plt.xticks([])
plt.yticks([])
plt.text(
    0.1, 0.1, "axes([0.3, 0.3, 0.5, 0.5])", ha="left", va="center", size=16, alpha=0.5
)

plt.axes([0.4, 0.4, 0.5, 0.5])
plt.xticks([])
plt.yticks([])
plt.text(
    0.1, 0.1, "axes([0.4, 0.4, 0.5, 0.5])", ha="left", va="center", size=16, alpha=0.5
)

plt.show()
