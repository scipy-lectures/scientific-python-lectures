from numpy import sqrt
import pylab

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth

inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = 0.5*fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [5*fig_width,5*fig_height]

#----------------- Graphics -----------------
params = {'backend': 'ps',
          'axes.labelsize': 20,
          'text.usetex': True,
          'text.fontsize': 20,
          'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'figure.figsize': fig_size,
          #'xtick.major.pad':-20,
          #'ytick.major.pad':-35,
          'xtick.major.size':6,
          'ytick.major.size':6,
          'legend.fontsize':20}

pylab.rcParams.update(params)

