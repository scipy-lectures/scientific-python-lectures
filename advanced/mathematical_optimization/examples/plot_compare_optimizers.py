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

method_names = results.values()[0]['Rosenbrock'].keys()
method_names.sort(key=lambda x: x[::-1], reverse=True)

for n_dim_index, ((n_dim, n_dim_bench), color) in enumerate(
            zip(sorted(results.items()), colors)):
    for (cost_name, cost_bench), symbol in zip(sorted(n_dim_bench.items()),
                    symbols):
        for method_index, method_name, in enumerate(method_names):
            this_bench = cost_bench[method_name]
            bench = np.mean(this_bench)
            pl.plot([method_index + .1*n_dim_index, ], [bench, ],
                    marker=symbol, color=color)

for cost_name, symbol in zip(sorted(n_dim_bench.keys()),
            symbols):
    pl.plot([-10, ], [0, ], symbol, color='.5',
            label=cost_name)

pl.xticks(np.arange(n_methods), method_names)
pl.xlim(-.5, n_methods - .5)
pl.yticks(())
pl.legend(loc='best', numpoints=1, handletextpad=0, prop=dict(size=12),
          frameon=False)
pl.ylabel('# function calls (a.u.)')
pl.twinx()

for n_dim, color in zip(sorted(results.keys()), colors):
    pl.plot([-10, ], [0, ], 'o', color=color,
            label='# dim: %i' % n_dim)
pl.legend(loc=(.61, .37), numpoints=1, handletextpad=0, prop=dict(size=12),
          frameon=False, ncol=2)
pl.xlim(-.5, n_methods - .5)

pl.xticks(np.arange(n_methods), method_names)

pl.tight_layout()
pl.show()


