"""Solve the ODE dy/dt = -2y between t = 0..4, with the
initial condition y(t=0) = 1.
"""

import numpy as np
from scipy.integrate import odeint
import pylab as pl

def calc_derivative(ypos, time):
    return -2*ypos

time_vec = np.linspace(0, 4, 40)
yvec = odeint(calc_derivative, 1, time_vec)

pl.plot(time_vec, yvec)
pl.xlabel('Time [s]')
pl.ylabel('y position [m]')

