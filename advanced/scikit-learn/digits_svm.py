from scikits.learn import datasets, svm
import pylab as pl

digits = datasets.load_digits()
clf = svm.LinearSVC(fit_intercept=False)
clf.fit(digits.data, digits.target)

for i in range(4):
    pl.subplot(2, 4, 1 + i)
    pl.imshow(clf.coef_[i].reshape(8, 8), cmap=pl.cm.gray_r, interpolation='nearest')
    pl.axis('off')
pl.show()
