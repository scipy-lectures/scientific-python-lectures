"""
Horizontal arrangement of subplots
==================================

An example showing horizontal arrangement of subplots with matplotlib.
"""

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.subplot(2, 1, 1)
plt.xticks([])
plt.yticks([])
plt.text(0.5, 0.5, "subplot(2,1,1)", ha="center", va="center", size=24, alpha=0.5)

plt.subplot(2, 1, 2)
plt.xticks([])
plt.yticks([])
plt.text(0.5, 0.5, "subplot(2,1,2)", ha="center", va="center", size=24, alpha=0.5)

plt.tight_layout()
plt.show()
