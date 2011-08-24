import numpy as np

data = np.loadtxt('../../../data/populations.txt')
year, hares, lynxes, carrots = data.T
populations = data[:,1:]

print "       Hares, Lynxes, Carrots"
print "Mean:", populations.mean(axis=0)
print "Std:", populations.std(axis=0)

j_max_years = np.argmax(populations, axis=0)
print "Max. year:", year[j_max_years]

max_species = np.argmax(populations, axis=1)
species = np.array(['Hare', 'Lynx', 'Carrot'])
print "Max species:"
print year
print species[max_species]

above_50000 = np.any(populations > 50000, axis=1)
print "Any above 50000:", year[above_50000]

j_top_2 = np.argsort(populations, axis=0)[:2]
print "Top 2 years with lowest populations for each:"
print year[j_top_2]

hare_grad = np.gradient(hares, 1.0)
print "diff(Hares) vs. Lynxes correlation", np.corrcoef(hare_grad, lynxes)[0,1]

import matplotlib.pyplot as plt
plt.plot(year, hare_grad, year, -lynxes)
plt.savefig('plot.png')
