""" Small example to plot lena."""
from scipy import misc
l = misc.lena()
misc.imsave('lena.png', l) # uses the Image module (PIL)
