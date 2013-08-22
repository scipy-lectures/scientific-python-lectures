from sklearn import datasets
import pylab as pl

digits = datasets.load_digits()

for i in range(8):
    pl.subplot(2, 4, 1 + i)
    pl.imshow(digits.images[3 * i], cmap=pl.cm.gray_r, interpolation='nearest')
#    pl.axis('off')
pl.show()

pl.imshow(digits.images[8], cmap=pl.cm.gray_r, interpolation='nearest')
pl.show()

pl.imshow(digits.images[8].reshape(1, -1), cmap=pl.cm.gray_r, interpolation='nearest')
pl.axis('off')
pl.show()
