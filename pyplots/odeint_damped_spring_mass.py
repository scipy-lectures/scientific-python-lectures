"""Damped spring-mass oscillator
"""

import numpy as np
from scipy.integrate import odeint
import pylab as pl

mass = 0.5
kspring = 4
cviscous = 0.4

nu_coef = cviscous / mass
om_coef = kspring / mass

def calc_deri(yvec, time, nuc, omc):
    return (yvec[1], -nuc * yvec[1] - omc * yvec[0])

time_vec = np.linspace(0, 10, 100)
yarr = odeint(calc_deri, (1, 0), time_vec, args=(nu_coef, om_coef))

pl.plot(time_vec, yarr[:, 0], label='y')
pl.plot(time_vec, yarr[:, 1], label="y'")
pl.legend()

