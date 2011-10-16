import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('../data/populations.txt')
populations = np.ma.masked_array(data[:,1:])
year = data[:,0]

bad_years = (((year >= 1903) & (year <= 1910))
           | ((year >= 1917) & (year <= 1918)))
populations[bad_years,0] = np.ma.masked
populations[bad_years,1] = np.ma.masked

print populations.mean(axis=0)
# [40472.7272727 18627.2727273 42400.0]

print populations.std(axis=0)
# [21087.656489 15625.7998142 3322.50622558]

# Note that Matplotlib knows about masked arrays:

plt.plot(year, populations, 'o-')
plt.show()
