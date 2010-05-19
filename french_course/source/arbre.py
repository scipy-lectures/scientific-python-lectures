import scipy, scipy.spatial
import numpy as np

np.random.seed(4)
points = np.random.random((2, 100))
plot(points[0], points[1], 'ok')
tree = scipy.spatial.KDTree(points.T)
indices = tree.query_ball_point((0.5, 0.5), 0.1)
plot([0.5], [0.5], 'or', ms=10)
plot(points[0,indices], points[1, indices], 'oc')
results = tree.query(tree.data, k=4)
[plot(points[0, l], points[1, l], 'oy') for l in results[1][::40]]
