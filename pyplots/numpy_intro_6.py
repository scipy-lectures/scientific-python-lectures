import numpy as np
import matplotlib.pyplot as plt

x, y = np.arange(5), np.arange(5)
distance = np.sqrt(x**2 + y[:, np.newaxis]**2)
print distance
# [[ 0.        ,  1.        ,  2.        ,  3.        ,  4.        ],
#  [ 1.        ,  1.41421356,  2.23606798,  3.16227766,  4.12310563],
#  [ 2.        ,  2.23606798,  2.82842712,  3.60555128,  4.47213595],
#  [ 3.        ,  3.16227766,  3.60555128,  4.24264069,  5.        ],
#  [ 4.        ,  4.12310563,  4.47213595,  5.        ,  5.65685425]]


# Or in color:

plt.pcolor(distance)
plt.colorbar()
plt.axis('equal')
plt.show()            # <-- again, not needed in interactive Python
