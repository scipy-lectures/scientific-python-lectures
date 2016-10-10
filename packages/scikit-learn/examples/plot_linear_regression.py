"""
A simple linear regression
===========================

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


a = 0.5
b = 1.0

# x from 0 to 10
x = 30 * np.random.random(20)

# y = a*x + b with noise
y = a * x + b + np.random.normal(size=x.shape)

# create a linear regression classifier
clf = LinearRegression()
clf.fit(x[:, np.newaxis], y)

# predict y from the data
x_new = np.linspace(0, 30, 100)
y_new = clf.predict(x_new[:, np.newaxis])

# plot the results
plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.scatter(x, y)
ax.plot(x_new, y_new)

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.axis('tight')


plt.show()
