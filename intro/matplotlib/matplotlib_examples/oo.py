#file matplotlib/oo.py

from matplotlib.figure import Figure          #1

figsize = (8, 5)                              #2
fig = Figure(figsize=figsize)                 #3
ax = fig.add_subplot(111)                     #4
line = ax.plot(range(10))[0]                  #5
ax.set_title('Plotted with OO interface')     #6
ax.set_xlabel('measured')
ax.set_ylabel('calculated')
ax.grid(True)                                 #7
line.set_marker('o')                          #8

from matplotlib.backends.backend_agg import FigureCanvasAgg #9
canvas = FigureCanvasAgg(fig)                 #10
canvas.print_figure("oo.png", dpi=80)         #11


import Tkinter as Tk                          #12
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #13

root = Tk.Tk()                                #14
canvas2 = FigureCanvasTkAgg(fig, master=root) #15
canvas2.show()                                #16
canvas2.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1) #17
Tk.mainloop()                                 #18

from matplotlib import _pylab_helpers         #19
import pylab                                  #20

pylab_fig = pylab.figure(1, figsize=figsize)  #21
figManager = _pylab_helpers.Gcf.get_active()  #22
figManager.canvas.figure = fig                #23
pylab.show()                                  #24
