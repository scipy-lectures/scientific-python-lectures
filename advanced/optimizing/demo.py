# For this example to run, you also need the 'ica.py' file

from ica import fastica

import numpy as np
import scipy as sp


# @profile  # uncomment this line to run with line_profiler
def test():
    rng = np.random.default_rng()
    data = rng.random((5000, 100))
    u, s, v = sp.linalg.svd(data)
    pca = u[:, :10].T @ data
    results = fastica(pca.T, whiten=False)


if __name__ == "__main__":
    test()
