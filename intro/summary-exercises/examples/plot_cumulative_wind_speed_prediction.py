"""Generate the image cumulative-wind-speed-prediction.png
for the interpolate section of scipy.rst.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
import pylab as pl

max_speeds = np.load('max-speeds.npy')
years_nb = max_speeds.shape[0]

cprob = (np.arange(years_nb, dtype=np.float32) + 1)/(years_nb + 1)
sorted_max_speeds = np.sort(max_speeds)
speed_spline = UnivariateSpline(cprob, sorted_max_speeds)
nprob = np.linspace(0, 1, 1e2)
fitted_max_speeds = speed_spline(nprob)

fifty_prob = 1. - 0.02
fifty_wind = speed_spline(fifty_prob)

pl.figure()
pl.plot(sorted_max_speeds, cprob, 'o')
pl.plot(fitted_max_speeds, nprob, 'g--')
pl.plot([fifty_wind], [fifty_prob], 'o', ms=8., mfc='y', mec='y')
pl.text(30, 0.05, '$V_{50} = %.2f \, m/s$' % fifty_wind)
pl.plot([fifty_wind, fifty_wind], [pl.axis()[2], fifty_prob], 'k--')
pl.xlabel('Annual wind speed maxima [$m/s$]')
pl.ylabel('Cumulative probability')
