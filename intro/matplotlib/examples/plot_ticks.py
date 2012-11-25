import pylab as pl
import numpy as np


def tickline():
    pl.xlim(0, 10), pl.ylim(-1, 1), pl.yticks([])
    ax = pl.gca()
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_minor_locator(pl.MultipleLocator(0.1))
    ax.plot(np.arange(11), np.zeros(11), color='none')
    return ax

locators = [
                'pl.NullLocator()',
                'pl.MultipleLocator(1.0)',
                'pl.FixedLocator([0, 2, 8, 9, 10])',
                'pl.IndexLocator(3, 1)',
                'pl.LinearLocator(5)',
                'pl.LogLocator(2, [1.0])',
                'pl.AutoLocator()',
            ]

n_locators = len(locators)

size = 512, 40 * n_locators
dpi = 72.0
figsize = size[0] / float(dpi), size[1] / float(dpi)
fig = pl.figure(figsize=figsize, dpi=dpi)
fig.patch.set_alpha(0)


for i, locator in enumerate(locators):
    pl.subplot(n_locators, 1, i + 1)
    ax = tickline()
    ax.xaxis.set_major_locator(eval(locator))
    pl.text(5, 0.3, locator[3:], ha='center')

pl.subplots_adjust(bottom=.01, top=.99, left=.01, right=.99)
pl.show()
