"""
============================================
Integrate the Damped spring-mass oscillator
============================================


"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

m = 0.5  # kg
k = 4  # N/m
c = 0.4  # N s/m

zeta = c / (2 * m * np.sqrt(k / m))
omega = np.sqrt(k / m)


def f(t, z, zeta, omega):
    return (z[1], -zeta * omega * z[1] - omega**2 * z[0])


t_span = (0, 10)
t_eval = np.linspace(*t_span, 100)
z0 = [1, 0]
res = sp.integrate.solve_ivp(
    f, t_span, z0, t_eval=t_eval, args=(zeta, omega), method="LSODA"
)

plt.figure(figsize=(4, 3))
plt.plot(res.t, res.y[0], label="y")
plt.plot(res.t, res.y[1], label="dy/dt")
plt.legend(loc="best")
plt.show()
