import numpy as np
import pylab
import cos_doubles

x = np.arange(0, 2 * np.pi, 0.1)
y = np.zeros(len(x))

cos_doubles.cos_doubles_func(x, y, len(x))
pylab.plot(x, y)
pylab.show()
