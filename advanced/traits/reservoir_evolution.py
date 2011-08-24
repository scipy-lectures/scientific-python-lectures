import numpy as np

initial_stock = 10.
inflows = np.array([6., 6, 4, 4, 1, 2, 0, 0, 3, 1, 5, 3])
releases = np.array([4., 5, 3, 5, 3, 5, 5, 3, 2, 1, 3, 3])

stock = initial_stock + (inflows - releases).cumsum()

print stock


