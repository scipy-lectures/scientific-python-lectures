"""Generate the image cumulative-wind-speed-prediction.png
for the interpolate section of scipy.rst.
"""

import numpy as N
from scipy.interpolate import UnivariateSpline
import pylab as P

max_speeds = N.load('data/max-speeds.npy')
years_nb = max_speeds.shape[0]

cprob = (N.arange(years_nb, dtype=N.float32) + 1)/(years_nb + 1)
sorted_max_speeds = N.sort(max_speeds)
speed_spline = UnivariateSpline(cprob, sorted_max_speeds)
nprob = N.linspace(0, 1, 1e2)
fitted_max_speeds = speed_spline(nprob)

fifty_prob = 1. - 0.02
fifty_wind = speed_spline(fifty_prob)

P.figure()
P.plot(sorted_max_speeds, cprob, 'o')
P.plot(fitted_max_speeds, nprob, 'g--')
P.plot([fifty_wind], [fifty_prob], 'o', ms=8., mfc='y', mec='y')
P.text(30, 0.05, '$V_{50} = %.2f \, m/s$' % fifty_wind)
P.plot([fifty_wind, fifty_wind], [P.axis()[2], fifty_prob], 'k--')
P.xlabel('Annual wind speed maxima [$m/s$]')
P.ylabel('Cumulative probability')
P.savefig('source/summary-exercices/cumulative-wind-speed-prediction.png')
