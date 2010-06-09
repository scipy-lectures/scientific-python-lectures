"""Generate the exercice results on the Gumbell distribution
"""
import numpy as N
from scipy.interpolate import UnivariateSpline
import pylab as P


def gumbell_dist(arr):
    return -N.log(-N.log(arr))

years_nb = 21
wspeeds = N.load('data/sprog-windspeeds.npy')
max_speeds = N.array([arr.max() for arr in N.array_split(wspeeds, years_nb)])
sorted_max_speeds = N.sort(max_speeds)

cprob = (N.arange(years_nb, dtype=N.float32) + 1)/(years_nb + 1)
gprob = gumbell_dist(cprob)
speed_spline = UnivariateSpline(gprob, sorted_max_speeds, k=1)
nprob = gumbell_dist(N.linspace(1e-3, 1-1e-3, 1e2))
fitted_max_speeds = speed_spline(nprob)

fifty_prob = gumbell_dist(49./50.)
fifty_wind = speed_spline(fifty_prob)

P.figure()
P.bar(N.arange(years_nb) + 1, max_speeds)
P.axis('tight')
P.xlabel('Year')
P.ylabel('Annual wind speed maxima [$m/s$]')
P.savefig('source/summary-exercices/sprog-annual-maxima.png')

P.figure()
P.plot(sorted_max_speeds, gprob, 'o')
P.plot(fitted_max_speeds, nprob, 'g--')
P.plot([fifty_wind], [fifty_prob], 'o', ms=8., mfc='y', mec='y')
P.plot([fifty_wind, fifty_wind], [P.axis()[2], fifty_prob], 'k--')
P.text(35, -1, r'$V_{50} = %.2f \, m/s$' % fifty_wind)
P.xlabel('Annual wind speed maxima [$m/s$]')
P.ylabel('Gumbell cumulative probability')
P.savefig('source/summary-exercices/gumbell-wind-speed-prediction.png')

