"""
=================
Coil Application
=================

An application to visualize the field create by a list of
coils.

This code is fairly complex, but it is actuallty a very rich application,
and a full-blown example of what you might want to do
"""

import numpy as np
from scipy import linalg, special
from traits.api import HasTraits, Array, CFloat, Str, List, \
   Instance, on_trait_change
from traits.ui.api import Item, View, HGroup, ListEditor, \
        HSplit, VSplit, spring
from mayavi.core.ui.api import EngineView, MlabSceneModel, \
        SceneEditor

##############################################################################
# A current loop

class Loop(HasTraits):
    """ A current loop.
    """
    direction = Array(float, value=(0, 0, 1), cols=3,
                    shape=(3,), desc='directing vector of the loop',
                    enter_set=True, auto_set=False)

    # CFloat tries to convert automatically to floats
    radius    = CFloat(0.1, desc='radius of the loop',
                    enter_set=True, auto_set=False)

    position  = Array(float, value=(0, 0, 0), cols=3,
                    shape=(3,), desc='position of the center of the loop',
                    enter_set=True, auto_set=False)

    plot      = None

    name      = Str

    view = View(HGroup(Item('name', style='readonly', show_label=False),
                       spring, 'radius'),
                'position', 'direction', '_')

    # For a Borg-like pattern
    __shared_state = {'number':0}

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.__shared_state['number'] += 1
        self.name =  'Coil %i' % self.__shared_state['number']

    def base_vectors(self):
        """ Returns 3 orthognal base vectors, the first one colinear to
            the axis of the loop.
        """
        # normalize n
        n = self.direction / (self.direction**2).sum(axis=-1)

        # choose two vectors perpendicular to n 
        # choice is arbitrary since the coil is symetric about n
        if  np.abs(n[0])==1 :
            l = np.r_[n[2], 0, -n[0]]
        else:
            l = np.r_[0, n[2], -n[1]]

        l /= (l**2).sum(axis=-1)
        m = np.cross(n, l)
        return n, l, m

    @on_trait_change('direction,radius,position')
    def redraw(self):
        if hasattr(self, 'app'):
            self.mk_B_field()
            if self.app.scene._renderer is not None:
                self.display()
                self.app.visualize_field()

    def display(self, half=False):
        """
        Display the coil in the 3D view.
        If half is True, display only one half of the coil.
        """
        n, l, m = self.base_vectors()
        theta = np.linspace(0, (2-half)*np.pi, 30)
        theta = theta[..., np.newaxis]
        coil = self.radius*(np.sin(theta)*l + np.cos(theta)*m)
        coil += self.position
        coil_x, coil_y, coil_z = coil.T
        if self.plot is None:
            self.plot = self.app.scene.mlab.plot3d(coil_x, coil_y, coil_z, 
                                    tube_radius=0.007, color=(0, 0, 1),
                                    name=self.name )
        else:
            self.plot.mlab_source.set(x=coil_x, y=coil_y, z=coil_z)

    def mk_B_field(self):
        """
        returns the magnetic field for the current loop calculated 
        from eqns (1) and (2) in Phys Rev A Vol. 35, N 4, pp. 1535-1546; 1987. 

        return: 
            B is a vector for the B field at point r in inverse units of 
        (mu I) / (2 pi d) 
        for I in amps and d in meters and mu = 4 pi * 10^-7 we get Tesla 
        """
        ### Translate the coordinates in the coil's frame
        n, l, m = self.base_vectors()
        R       = self.radius
        r0      = self.position
        r       = np.c_[np.ravel(self.app.X), np.ravel(self.app.Y),
                                                np.ravel(self.app.Z)]

        # transformation matrix coil frame to lab frame
        trans = np.vstack((l, m, n))

        r -= r0	  #point location from center of coil
        r = np.dot(r, linalg.inv(trans) ) 	    #transform vector to coil frame 

        #### calculate field

        # express the coordinates in polar form
        x = r[:, 0]
        y = r[:, 1]
        z = r[:, 2]
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan(x/y)

        E = special.ellipe((4 * R * rho)/( (R + rho)**2 + z**2))
        K = special.ellipk((4 * R * rho)/( (R + rho)**2 + z**2))
        Bz =  1/np.sqrt((R + rho)**2 + z**2) * ( 
                    K 
                  + E * (R**2 - rho**2 - z**2)/((R - rho)**2 + z**2) 
                )
        Brho = z/(rho*np.sqrt((R + rho)**2 + z**2)) * ( 
                -K 
                + E * (R**2 + rho**2 + z**2)/((R - rho)**2 + z**2) 
                )
        # On the axis of the coil we get a divided by zero here. This returns a
        # NaN, where the field is actually zero :
        Brho[np.isnan(Brho)] = 0

        B = np.c_[np.cos(theta)*Brho, np.sin(theta)*Brho, Bz ]

        # Rotate the field back in the lab's frame
        B = np.dot(B, trans)

        Bx, By, Bz = B.T
        Bx = np.reshape(Bx, self.app.X.shape)
        By = np.reshape(By, self.app.X.shape)
        Bz = np.reshape(Bz, self.app.X.shape)

        Bnorm = np.sqrt(Bx**2 + By**2 + Bz**2)

        # We need to threshold ourselves, rather than with VTK, to be able 
        # to use an ImageData
        Bmax = 10 * np.median(Bnorm)

        Bx[Bnorm > Bmax] = np.NAN 
        By[Bnorm > Bmax] = np.NAN
        Bz[Bnorm > Bmax] = np.NAN
        Bnorm[Bnorm > Bmax] = np.NAN

        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.Bnorm = Bnorm


##############################################################################
# The application

class Application(HasTraits):

    scene = Instance(MlabSceneModel, (), editor=SceneEditor())

    # The mayavi engine view.
    engine_view = Instance(EngineView)

    # We use a traits List to be able to add coils to it
    coils = List(Loop,
                    value=( Loop(position=(0, 0, -0.05), ),
                            Loop(position=(0, 0,  0.05), ), ),
                    editor=ListEditor(use_notebook=True, deletable=False,
                                        style='custom'),
                 )

    # The grid of points on which we want to evaluate the field
    X, Y, Z = np.mgrid[-0.15:0.15:20j, -0.15:0.15:20j, -0.15:0.15:20j]

    # Avoid rounding issues:
    f = 1e4  # this gives the precision we are interested by :
    X = np.round(X * f) / f
    Y = np.round(Y * f) / f
    Z = np.round(Z * f) / f

    Bx    = Array(value=np.zeros_like(X))
    By    = Array(value=np.zeros_like(X))
    Bz    = Array(value=np.zeros_like(X))
    Bnorm = Array(value=np.zeros_like(X))

    field = None

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.engine_view = EngineView(engine=self.scene.engine)

    @on_trait_change('scene.activated')
    def init_view(self):
        # This gets fired when the viewer of the scene is created
        self.scene.scene_editor.background = (0, 0, 0)
        for coil in self.coils:
            coil.app = self
            coil.display()
            coil.mk_B_field()

        self.visualize_field()

    def visualize_field(self):
        self.Bx    = np.zeros_like(self.X)
        self.By    = np.zeros_like(self.X)
        self.Bz    = np.zeros_like(self.X)
        self.Bnorm = np.zeros_like(self.X)
        for coil in self.coils:
            if hasattr(coil, 'Bx'):
                self.Bx += coil.Bx
                self.By += coil.By
                self.Bz += coil.Bz
                self.Bnorm += coil.Bnorm

        if self.field is None:
            self.field = self.scene.mlab.pipeline.vector_field(
                            self.X, self.Y, self.Z, self.Bx, self.By, self.Bz, 
                            scalars = self.Bnorm,
                            name='B field')
            vectors = self.scene.mlab.pipeline.vectors(self.field,
                                    mode='arrow', resolution=10,
                                    mask_points=6, colormap='YlOrRd',
                                    scale_factor=2*np.abs(self.X[0,0,0]
                                                          -self.X[1,1,1]) )
            vectors.module_manager.vector_lut_manager.reverse_lut = True
            vectors.glyph.mask_points.random_mode = False
            self.scene.mlab.axes()
            self.scp = self.scene.mlab.pipeline.scalar_cut_plane(self.field,
                                                      colormap='hot')
        else:
            self.field.mlab_source.set(x=self.X,  y=self.Y,  z=self.Z,
                                       u=self.Bx, v=self.By, w=self.Bz,
                                       scalars=self.Bnorm)

    view = View(HSplit(
                    VSplit(Item(name='engine_view',
                                   style='custom',
                                   resizable=True),
                            Item('coils', springy=True),
                        show_labels=False),
                        'scene',
                        show_labels=False),
                    resizable=True,
                    title='Coils...',
                    height=0.8,
                    width=0.8,
                )

##############################################################################
if __name__ == '__main__':
    app = Application()
    app.configure_traits()

