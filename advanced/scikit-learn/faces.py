import numpy as np
from scikits.learn import cross_val, datasets, decomposition, svm

lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
faces = np.reshape(lfw_people.data, (lfw_people.target.shape[0], -1))
train, test = iter(cross_val.StratifiedKFold(lfw_people.target, k=4)).next()
X_train, X_test = faces[train], faces[test]
y_train, y_test = lfw_people.target[train], lfw_people.target[test]


pca = decomposition.RandomizedPCA(n_components=150, whiten=True)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

clf = svm.SVC(C=5., gamma=0.001)
clf.fit(X_train_pca, y_train)

print clf.score(X_test_pca, y_test)

## reshape : 50, 37

