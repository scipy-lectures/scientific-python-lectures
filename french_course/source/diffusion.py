import numpy as np
import scipy, scipy.optimize
import plots

nreal = 1000
tmax = 200
walk=2*(np.random.random_integers(0, 1, (nreal,200))-0.5)
cumwalk = np.cumsum(walk, axis=1)
sq_distance = cumwalk**2
mean_distance = np.sqrt(np.mean(sq_distance, axis=0))
axes([0.18,0.18,0.75,0.75])
plot(mean_distance, lw=2, label='distance moyenne')

def f(A, y, x):
    err = y - A*np.sqrt(x)
    return err

t = np.arange(tmax)
coeff = scipy.optimize.leastsq(f, 0.8, args=(mean_distance, t))
plot(t, coeff[0]*np.sqrt(t), lw=2, label='fit en $\sqrt{t}$')
xlabel('$t$', fontsize=30)
ylabel('$d(t)$', fontsize=30)
legend(loc='3')

pylab.rcParams.update({'xtick.labelsize': 20})
pylab.rcParams.update({'ytick.labelsize': 20})
figure(figsize=(12,6))
axes([0.1,0.15,0.38,0.75])
plot(mean_distance**2, lw=2)
xlabel('$t$', fontsize=20)
ylabel('$d^2(t)$', fontsize=20)
ylim(0, 220)
axes([0.57,0.15,0.38,0.75])
plot(mean_distance, lw=2)
xlabel('$t$', fontsize=22)
ylabel('$d(t)$', fontsize=22)
ylim(0, 15)
