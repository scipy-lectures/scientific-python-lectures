# For this example to run, you also need the 'ica.py' file

import numpy as np
import scipy as sp

from ica import fastica


def test():
    rng = np.random.default_rng()
    data = rng.random((5000, 100))
    u, s, v = sp.linalg.svd(data, full_matrices=False)
    pca = u[:, :10].T @ data
    results = fastica(pca.T, whiten=False)


if __name__ == "__main__":
    test()
