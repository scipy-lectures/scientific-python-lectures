import pickle

import numpy as np
import pylab as pl

results = pickle.load(file('compare_optimizers.pkl'))
n_methods = len(results.values()[0]['Rosenbrock'])
n_dims = len(results)

symbols = 'o>Ds*'

pl.figure(1, figsize=(8, 4))
pl.clf()

colors = pl.cm.Spectral(np.linspace(0, 1, n_dims))[:, :3]

for n_dim_index, ((n_dim, n_dim_bench), color) in enumerate(
            zip(sorted(results.items()), colors)):
    for (cost_name, cost_bench), symbol in zip(sorted(n_dim_bench.items()),
                    symbols):
        for method_index, (method_name, this_bench) in enumerate(
                                                sorted(cost_bench.items())):
            bench = np.mean(this_bench)
            pl.plot([method_index + .1*n_dim_index, ], [bench, ],
                    marker=symbol, color=color)

pl.xticks(np.arange(n_methods),
          sorted(results.values()[0]['Rosenbrock'].keys()))
pl.xlim(-.5, n_methods - .5)
pl.yticks(())
pl.show()


