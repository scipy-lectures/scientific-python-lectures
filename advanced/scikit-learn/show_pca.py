from sklearn import datasets, decomposition
import pylab as pl

iris = datasets.load_iris()

pca = decomposition.PCA(n_components=2)
iris_2D = pca.fit(iris.data).transform(iris.data)

pl.scatter(iris_2D[:, 0], iris_2D[:, 1], c=iris.target)
pl.show()
