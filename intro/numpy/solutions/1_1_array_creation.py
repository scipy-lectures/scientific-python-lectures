import numpy as np

a = np.ones((4, 4), dtype=int)
a[3,1] = 6
a[2,3] = 2

b = np.zeros((6, 5))
b[1:] = np.diag(np.arange(2, 7))

print a
print b
