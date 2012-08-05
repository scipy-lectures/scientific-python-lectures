from sklearn import datasets, svm

digits = datasets.load_digits()
clf = svm.SVC(kernel='linear')
n_train = int(.9 * digits.target.shape[0])
clf.fit(digits.data[:n_train], digits.target[:n_train])
print clf.score(digits.data[n_train:], digits.target[n_train:])
