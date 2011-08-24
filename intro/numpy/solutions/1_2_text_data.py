import numpy as np

data = np.loadtxt('../../../data/populations.txt')
reduced_data = data[5:,:-1]
np.savetxt('pop2.txt', reduced_data)
