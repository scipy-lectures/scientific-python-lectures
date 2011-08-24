"""
Discover the periods in ../../../data/populations.txt
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('../../../data/populations.txt')
years = data[:,0]
populations = data[:,1:]

ft_populations = np.fft.fft(populations, axis=0)
frequencies = np.fft.fftfreq(populations.shape[0], years[1] - years[0])

plt.figure()
plt.plot(years, populations)
plt.figure()
plt.plot(1/frequencies, abs(ft_populations), 'o')
plt.xlim(0, 20)
plt.show()

# There's probably a period of around 10 years (obvious from the
# plot), but for this crude a method, there's not enough data to say
# much more.
