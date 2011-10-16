""" Small example to plot lena."""
import scipy
l = scipy.lena()
from scipy import misc
misc.imsave('lena.png', l) # uses the Image module (PIL)
