"""
=========================
Integrating a simple ODE
=========================

Solve the ODE dy/dt = -2y between t = 0..4, with the initial condition
y(t=0) = 1.
"""

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

def calc_derivative(time, ypos):
    return -2*ypos

t_span = (0, 4)
solution = solve_ivp(calc_derivative, t_span, (1,))

plt.figure(figsize=(4, 3))
plt.plot(solution.t, solution.y[0,:])
plt.xlabel('t: Time')
plt.ylabel('y: Position')
plt.tight_layout()

