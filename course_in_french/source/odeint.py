from scipy.integrate import odeint
import numpy as np

def rhs(y, t):
    return -2*y
    
t = np.linspace(0, 10, 100)
y = odeint(rhs, 1, t)

figure(figsize=(8, 3.5))
axes([0.1,0.15, 0.38, 0.8])
plot(t, y, lw=2)
xlabel('t')
ylabel('y')
axes([0.58,0.15, 0.38, 0.8])
semilogy(t, y, lw=2)
xlabel('t')
ylabel('y')

"""
>>> plot(t, y[:,0], lw=2, label="y")
[<matplotlib.lines.Line2D object at 0xd56bdcc>]
>>> plot(t, y[:,1], lw=2, label="y'")
[<matplotlib.lines.Line2D object at 0xd5f38ac>]
>>> axhline(0)
<matplotlib.lines.Line2D object at 0xd618d0c>
>>> legend()
<matplotlib.legend.Legend object at 0xd61d66c>
>>> xlabel('t')
<matplotlib.text.Text object at 0xd5ebe4c>
"""
