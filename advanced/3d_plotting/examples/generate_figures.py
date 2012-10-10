"""
Example generating the figures for the tutorial.
"""
import numpy as np
from mayavi import mlab

# Seed the random number generator, for reproducibility
np.random.seed(0)

mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
mlab.clf()

### begin points3d example
x, y, z, value = np.random.random((4, 40))
mlab.points3d(x, y, z, value)
### end points3d example

mlab.view(distance='auto')
mlab.text(.02, .9, 'points3d', width=.35)
mlab.savefig('points3d.png')


### begin plot3d example
mlab.clf()  # Clear the figure
t = np.linspace(0, 20, 200)
mlab.plot3d(np.sin(t), np.cos(t), 0.1*t, t)
### end plot3d example

mlab.view(distance='auto')
mlab.text(.02, .9, 'plot3d', width=.25)
mlab.savefig('plot3d.png')


### begin surf example
mlab.clf()
x, y = np.mgrid[-10:10:100j, -10:10:100j]
r = np.sqrt(x**2 + y**2)
z = np.sin(r)/r
mlab.surf(z, warp_scale='auto')
### end surf example

mlab.view(distance='auto')
mlab.text(.02, .9, 'surf', width=.15)
mlab.savefig('surf.png')

### begin mesh example
mlab.clf()
phi, theta = np.mgrid[0:np.pi:11j, 0:2*np.pi:11j]
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)
mlab.mesh(x, y, z)
mlab.mesh(x, y, z, representation='wireframe',
          color=(0, 0, 0))
### end mesh example

mlab.view(distance='auto')
mlab.text(.02, .9, 'mesh', width=.2)
mlab.savefig('mesh.png')

### begin contour3d example
mlab.clf()
x, y, z = np.mgrid[-5:5:64j, -5:5:64j, -5:5:64j]
values = x*x*0.5 + y*y + z*z*2.0
mlab.contour3d(values)
### end contour3d example

mlab.view(distance='auto')
mlab.text(.02, .9, 'contour3d', width=.45)
mlab.savefig('contour3d.png')

