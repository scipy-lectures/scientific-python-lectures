"""
The correction for the calculation of pi using the Wallis formula.
"""
from __future__ import division

pi = 3.14159265358979312

my_pi = 1.

for i in range(1, 100000):
    my_pi *= 4 * i ** 2 / (4 * i ** 2 - 1.)

my_pi *= 2

print pi
print my_pi
print abs(pi - my_pi)

###############################################################################
num = 1
den = 1
for i in range(1, 100000):
    tmp = 4 * i * i
    num *= tmp
    den *= tmp - 1

better_pi = 2 * (num / den)

print pi
print better_pi
print abs(pi - better_pi)
print abs(my_pi - better_pi)

###############################################################################
# Solution in a single line using more adcanved constructs (reduce, lambda,
# list comprehensions
print 2 * reduce(lambda x, y: x * y,
                 [float((4 * (i ** 2))) / ((4 * (i ** 2)) - 1)
                 for i in range(1, 100000)])
