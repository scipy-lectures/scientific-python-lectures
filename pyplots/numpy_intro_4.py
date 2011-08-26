import numpy as np
import matplotlib.pyplot as plt

# We can first plot the data:

data = np.loadtxt('../data/populations.txt')
year, hares, lynxes, carrots = data.T  # trick: columns to variables

plt.axes([0.2, 0.1, 0.5, 0.8])
plt.plot(year, hares, year, lynxes, year, carrots)
plt.legend(('Hare', 'Lynx', 'Carrot'), loc=(1.05, 0.5))
plt.show()

# The mean populations over time:
populations = data[:,1:]
print populations.mean(axis=0)
# [ 34080.95238095,  20166.66666667,  42400.        ]

# The sample standard deviations:
print populations.std(axis=0, ddof=1)
# [ 21413.98185877,  16655.99991995,   3404.55577132]

# Which species has the highest population each year?
print np.argmax(populations, axis=1)
# [2, 2, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 2, 2, 2, 2, 2]
