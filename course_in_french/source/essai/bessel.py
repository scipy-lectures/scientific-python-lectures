from numpy import *
from pylab import *

#figure(1, figsize=(5, 5))
figure(1, figsize=(5, 3.4))
clf()

##############################################################################
from scipy.special import jn
for index, i in enumerate(range(1, 14)):
    # First plot the filled regions
    x = linspace(0, 10, 200)
    y = jn(i, x)
    color = list(cm.hsv((index-1)/13.))
    fill(r_[0, x, 10], r_[0, y, 0], linewidth=0, facecolor=color, alpha=0.4)
    fill(r_[0, x, 10], r_[0, -y, 0], linewidth=0, facecolor=color, alpha=0.4)

for i in reversed(range(1, 14)):
    # Then plot the outlines.
    index = i - 1
    x = linspace(0, 10, 400)
    y = jn(i, x)
    color = list(cm.hsv((index-1)/13.))
     # Find the intersection with the next function
    yy = jn(i+1, x)
    xmin = argmax(y)
    is_close = abs(y - yy)[xmin:] < 0.02
    if alltrue(~ is_close):
        xmax = len(x)
    else:
        xmax = argmax(is_close) + xmin + 2 + floor(0.7*index)
    x = x[:xmax]
    y = y[:xmax]
    plot(x, y, linewidth=1.5, color=color)
    plot(x, -y, linewidth=1.5, color=color)
    
    if x[xmin] < 10:
        x0 = x[xmin]
        y0 = -y.max()+0.04
        t = linspace(-pi, pi, 100)
        plot([x0, ], [y0, ], 'o',
                    #color=(1 - 0.4*(1-array(color))),
                    alpha = 0.6,
                    color = (1, 1, 1, 1),
                    markersize=18,
                    markeredgecolor=color,
                    markeredgewidth=2,
            )
        text(x0, y0, r"$%i$" % i,
                size=20,
                verticalalignment='center',
                horizontalalignment='center')

text(5.5, 0.75,
    r"""$\bf J_m(x)=\sum^{\infty}_{l=0}\frac{(-1)^l x^{2l+m}}{2^{2l+m}l!(m+l)!}$"""
    ,
    verticalalignment='center',
    horizontalalignment='center',
    size=24,
    )

gca().xaxis.tick_bottom()
gca().yaxis.tick_left()
ylim(-0.7, 1.05)
yticks(arange(-0.6, 0.61, 0.2))

savefig('bessel.pdf')

show()

