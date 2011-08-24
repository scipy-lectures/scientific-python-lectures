import numpy as np
from numpy import newaxis

def f(a, b, c):
    return a**b - c

a = np.linspace(0, 1, 24)
b = np.linspace(0, 1, 12)
c = np.linspace(0, 1, 6)

samples = f(a[:,newaxis,newaxis],
            b[newaxis,:,newaxis],
            c[newaxis,newaxis,:])

# or,
#
# a, b, c = np.ogrid[0:1:24j, 0:1:12j, 0:1:6j]
# samples = f(a, b, c)

integral = samples.mean()

print "Approximation:", integral
print "Exact:", np.log(2) - 0.5
