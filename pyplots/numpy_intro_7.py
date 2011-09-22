import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('../data/populations.txt')
year, hares, lynxes, carrots = data.T  # trick: columns to variables

plt.axes([0.2, 0.1, 0.5, 0.8])
plt.plot(year, hares, year, lynxes, year, carrots)
plt.legend(('Hare', 'Lynx', 'Carrot'), loc=(1.05, 0.5))
plt.show()
