"""
The lidar system, data (2 of 2 datasets)
========================================

Generate a chart of more complex data recorded by the lidar system
"""

import numpy as np
import matplotlib.pyplot as plt

waveform_2 = np.load("waveform_2.npy")

t = np.arange(len(waveform_2))

fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(t, waveform_2)
plt.xlabel("Time [ns]")
plt.ylabel("Amplitude [bins]")
plt.show()
