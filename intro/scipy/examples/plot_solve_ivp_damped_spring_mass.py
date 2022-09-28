"""
============================================
Integrate the Damped spring-mass oscillator
============================================


"""

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

mass = 0.5  # kg
kspring = 4  # N/m
cviscous = 0.4  # N s/m


eps = cviscous / (2 * mass * np.sqrt(kspring/mass))
omega = np.sqrt(kspring / mass)


def calc_deri(time, yvec, eps, omega):
    return (yvec[1], -eps * omega * yvec[1] - omega **2 * yvec[0])

time_span = (0, 10)
yinit = (1, 0)
solution = solve_ivp(calc_deri, time_span, yinit, args=(eps, omega), method='LSODA')

plt.figure(figsize=(4, 3))
plt.plot(solution.t, solution.y[0,:], label='y')
plt.plot(solution.t, solution.y[1,:], label="y'")
plt.legend(loc='best')
plt.show()

