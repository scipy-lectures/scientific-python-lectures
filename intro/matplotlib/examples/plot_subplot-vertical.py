"""
Subplot plot arrangement vertical
==================================

An example showing vertical arrangement of subplots with matplotlib.
"""

import matplotlib.pyplot as plt


plt.figure(figsize=(6, 4))
plt.subplot(1, 2, 1)
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'subplot(1,2,1)', ha='center', va='center',
        size=24, alpha=.5)

plt.subplot(1, 2, 2)
plt.xticks(())
plt.yticks(())
plt.text(0.5, 0.5, 'subplot(1,2,2)', ha='center', va='center',
        size=24, alpha=.5)

plt.tight_layout()
plt.show()
