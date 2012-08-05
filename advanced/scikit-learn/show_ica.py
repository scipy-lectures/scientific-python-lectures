from sklearn import datasets, decomposition
import pylab as pl
import numpy as np

digits = datasets.load_digits()

digits.data += .2 * np.random.normal(size=digits.data.shape)
ica = decomposition.FastICA(n_components=10)
tt = ica.fit(digits.data.T).transform(digits.data.T).T

for i in range(8):
    pl.subplot(2, 4, 1 + i)
    pl.imshow(tt[i].reshape(8, 8), cmap=pl.cm.gray_r, interpolation='nearest')
#    pl.axis('off')
pl.show()
