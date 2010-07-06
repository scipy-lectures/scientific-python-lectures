"""Solve the ODE dy/dt = -2y between t = 0..4, with the
initial condition y(t=0) = 1.
"""

import numpy as np
from scipy.integrate import odeint

def calc_derivative(ypos, time):
    return -2*ypos

time_vec = np.linspace(0, 4, 40)
yvec = odeint(calc_derivative, 1, time_vec)

import pylab as P
P.plot(time_vec, yvec)
P.xlabel('Time [s]')
P.ylabel('y position [m]')
P.savefig('source/odeint-introduction.png')

