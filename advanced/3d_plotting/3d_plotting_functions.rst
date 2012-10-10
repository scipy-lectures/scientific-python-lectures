3D plotting functions
=======================

The :mod:`mayavi.mlab` module provides simple plotting functions to apply
 to numpy arrays.
 
.. image:: examples/points3d.png
    :align: right
    :scale: 60

Points
-------
 
.. literalinclude:: examples/generate_figures.py
    :start-after: ### begin points3d example
    :end-before: ### end points3d example

|clear-floats|

.. image:: examples/plot3d.png
    :align: right
    :scale: 60

Lines
------

.. literalinclude:: examples/generate_figures.py
    :start-after: ### begin plot3d example
    :end-before: ### end plot3d example

|clear-floats|

.. image:: examples/surf.png
    :align: right
    :scale: 60

Elevation surface
-------------------

.. literalinclude:: examples/generate_figures.py
    :start-after: ### begin surf example
    :end-before: ### end surf example

|clear-floats|

.. image:: examples/mesh.png
    :align: right
    :scale: 60

Arbitrary regular mesh
-----------------------

.. literalinclude:: examples/generate_figures.py
    :start-after: ### begin mesh example
    :end-before: ### end mesh example

.. note:: 

    A surface is defined by points **connected** to form triangles or
    polygones. In `mlab.func` and `mlab.mesh`, the connectivity is
    implicity given by the layout of the arrays. See also
    `mlab.triangular_mesh`.

**Our data is often more than points and values: it needs some
connectivity information**

.. _mayavi-voldata-label: 

.. image:: examples/contour3d.png
    :align: right
    :scale: 60

Volumetric data
----------------

.. literalinclude:: examples/generate_figures.py
    :start-after: ### begin contour3d example
    :end-before: ### end contour3d example

**This function works with a regular orthogonal grid:**

    .. image:: examples/viz_volume_structure.png
	:align: center
	:scale: 70


