"""
=========================
Integrating a simple ODE
=========================

Solve the ODE dy/dt = -2y between t = 0..4, with the initial condition
y(t=0) = 1.
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def f(t, y):
    return -2 * y


t_span = (0, 4)  # time interval
t_eval = np.linspace(*t_span)  # times at which to evaluate `y`
y0 = [
    1,
]  # initial state
res = sp.integrate.solve_ivp(f, t_span=t_span, y0=y0, t_eval=t_eval)

plt.figure(figsize=(4, 3))
plt.plot(res.t, res.y[0])
plt.xlabel("t")
plt.ylabel("y")
plt.title("Solution of Initial Value Problem")
plt.tight_layout()
plt.show()
