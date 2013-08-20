.. _matplotlib:

====================
Matplotlib: plotting
====================

:authors: Nicolas Rougier, Mike Müller, Gaël Varoquaux

.. sidebar:: **Thanks**

    Many thanks to **Bill Wing** and **Christoph Deil** for review and
    corrections.

.. contents:: Chapters contents
   :local:
   :depth: 2

Introduction
============

.. tip::

    `Matplotlib <http://matplotlib.org/>`_ is probably the single most
    used Python package for 2D-graphics. It provides both a very quick
    way to visualize data from Python and publication-quality figures in
    many formats.  We are going to explore matplotlib in interactive mode
    covering most common cases.

IPython and the pylab mode
--------------------------

.. tip::

    `IPython <http://ipython.org/>`_ is an enhanced interactive Python
    shell that has lots of interesting features including named inputs
    and outputs, access to shell commands, improved debugging and many
    more. It is central to the scientific-computing workflow in Python
    for its use in combination with Matplotlib:

We start IPython with the command line argument ``-pylab`` (``--pylab``
since IPython version 0.12), for interactive matplotlib sessions with
Matlab/Mathematica-like functionality.

pylab
-----

.. tip::

    *pylab* provides a procedural interface to the matplotlib
    object-oriented plotting library. It is modeled closely after
    Matlab™. Therefore, the majority of plotting commands in pylab have
    Matlab™ analogs with similar arguments.  Important commands are
    explained with interactive examples.


Simple plot
===========

.. tip::

    In this section, we want to draw the cosine and sine functions on the
    same plot. Starting from the default settings, we'll enrich the
    figure step by step to make it nicer.

    First step is to get the data for the sine and cosine functions:

::

   import numpy as np

   X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
   C, S = np.cos(X), np.sin(X)


``X`` is now a numpy array with 256 values ranging from -π to +π
(included). ``C`` is the cosine (256 values) and ``S`` is the sine (256
values).

To run the example, you can type them in an IPython interactive session::

    $ ipython --pylab

This brings us to the IPython prompt: ::

    IPython 0.13 -- An enhanced Interactive Python.
    ?       -> Introduction to IPython's features.
    %magic  -> Information about IPython's 'magic' % functions.
    help    -> Python's own help system.
    object? -> Details about 'object'. ?object also works, ?? prints more.

    Welcome to pylab, a matplotlib-based Python environment.
    For more information, type 'help(pylab)'.

.. tip::

    You can also download each of the examples and run it using regular
    python, but you will loose interactive data manipulation::

        $ python exercice_1.py

    You can get source for each step by clicking on the corresponding figure.


Plotting with default settings
-------------------------------

.. image:: auto_examples/images/plot_exercice_1_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_1.html

.. hint:: Documentation

   * `plot tutorial <http://matplotlib.sourceforge.net/users/pyplot_tutorial.html>`_
   * `plot() command <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.plot>`_

.. tip::

    Matplotlib comes with a set of default settings that allow
    customizing all kinds of properties. You can control the defaults of
    almost every property in matplotlib: figure size and dpi, line width,
    color and style, axes, axis and grid properties, text and font
    properties and so on.

::

   import pylab as pl
   import numpy as np

   X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
   C, S = np.cos(X), np.sin(X)

   pl.plot(X, C)
   pl.plot(X, S)

   pl.show()


Instantiating defaults
----------------------

.. image:: auto_examples/images/plot_exercice_2_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_2.html

.. hint:: Documentation

   *  `Customizing matplotlib <http://matplotlib.sourceforge.net/users/customizing.html>`_

In the script below, we've instantiated (and commented) all the figure settings
that influence the appearance of the plot.

.. tip::

    The settings have been explicitly set to their default values, but
    now you can interactively play with the values to explore their
    affect (see `Line properties`_ and `Line styles`_ below).

::

   import pylab as pl
   import numpy as np

   # Create a figure of size 8x6 points, 80 dots per inch
   pl.figure(figsize=(8, 6), dpi=80)

   # Create a new subplot from a grid of 1x1
   pl.subplot(1, 1, 1)

   X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
   C, S = np.cos(X), np.sin(X)

   # Plot cosine with a blue continuous line of width 1 (pixels)
   pl.plot(X, C, color="blue", linewidth=1.0, linestyle="-")

   # Plot sine with a green continuous line of width 1 (pixels)
   pl.plot(X, S, color="green", linewidth=1.0, linestyle="-")

   # Set x limits
   pl.xlim(-4.0, 4.0)

   # Set x ticks
   pl.xticks(np.linspace(-4, 4, 9, endpoint=True))

   # Set y limits
   pl.ylim(-1.0, 1.0)

   # Set y ticks
   pl.yticks(np.linspace(-1, 1, 5, endpoint=True))

   # Save figure using 72 dots per inch
   # savefig("exercice_2.png", dpi=72)

   # Show result on screen
   pl.show()


Changing colors and line widths
--------------------------------

.. image:: auto_examples/images/plot_exercice_3_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_3.html

.. hint:: Documentation

   * `Controlling line properties <http://matplotlib.sourceforge.net/users/pyplot_tutorial.html#controlling-line-properties>`_
   * `Line API <http://matplotlib.sourceforge.net/api/artist_api.html#matplotlib.lines.Line2D>`_

.. tip::

    First step, we want to have the cosine in blue and the sine in red
    and a slighty thicker line for both of them. We'll also slightly
    alter the figure size to make it more horizontal.
    
::

   ...
   pl.figure(figsize=(10, 6), dpi=80)
   pl.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
   pl.plot(X, S, color="red",  linewidth=2.5, linestyle="-")
   ...


Setting limits
--------------

.. image:: auto_examples/images/plot_exercice_4_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_4.html

.. hint:: Documentation

   * `xlim() command <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.xlim>`_
   * `ylim() command <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.ylim>`_

.. tip::

    Current limits of the figure are a bit too tight and we want to make
    some space in order to clearly see all data points.

::

   ...
   pl.xlim(X.min() * 1.1, X.max() * 1.1)
   pl.ylim(C.min() * 1.1, C.max() * 1.1)
   ...



Setting ticks
-------------

.. image:: auto_examples/images/plot_exercice_5_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_5.html

.. hint:: Documentation

   * `xticks() command <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.xticks>`_
   * `yticks() command <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.yticks>`_
   * `Tick container <http://matplotlib.sourceforge.net/users/artists.html#axis-container>`_
   * `Tick locating and formatting <http://matplotlib.sourceforge.net/api/ticker_api.html>`_

.. tip::

    Current ticks are not ideal because they do not show the interesting
    values (+/-π,+/-π/2) for sine and cosine. We'll change them such that
    they show only these values.

::

   ...
   pl.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
   pl.yticks([-1, 0, +1])
   ...



Setting tick labels
-------------------

.. image:: auto_examples/images/plot_exercice_6_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_6.html


.. hint:: Documentation

   * `Working with text <http://matplotlib.sourceforge.net/users/index_text.html>`_
   * `xticks() command <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.xticks>`_
   * `yticks() command <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.yticks>`_
   * `set_xticklabels() <http://matplotlib.sourceforge.net/api/axes_api.html?#matplotlib.axes.Axes.set_xticklabels>`_
   * `set_yticklabels() <http://matplotlib.sourceforge.net/api/axes_api.html?#matplotlib.axes.Axes.set_yticklabels>`_


.. tip::

    Ticks are now properly placed but their label is not very explicit.
    We could guess that 3.142 is π but it would be better to make it
    explicit. When we set tick values, we can also provide a
    corresponding label in the second argument list. Note that we'll use
    latex to allow for nice rendering of the label.

::

   ...
   pl.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
             [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

   pl.yticks([-1, 0, +1],
             [r'$-1$', r'$0$', r'$+1$'])
   ...



Moving spines
-------------

.. image:: auto_examples/images/plot_exercice_7_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_7.html


.. hint:: Documentation

   * `Spines <http://matplotlib.sourceforge.net/api/spines_api.html#matplotlib.spines>`_
   * `Axis container <http://matplotlib.sourceforge.net/users/artists.html#axis-container>`_
   * `Transformations tutorial <http://matplotlib.sourceforge.net/users/transforms_tutorial.html>`_

.. tip::

    Spines are the lines connecting the axis tick marks and noting the
    boundaries of the data area. They can be placed at arbitrary
    positions and until now, they were on the border of the axis. We'll
    change that since we want to have them in the middle. Since there are
    four of them (top/bottom/left/right), we'll discard the top and right
    by setting their color to none and we'll move the bottom and left
    ones to coordinate 0 in data space coordinates.

::

   ...
   ax = pl.gca()  # gca stands for 'get current axis'
   ax.spines['right'].set_color('none')
   ax.spines['top'].set_color('none')
   ax.xaxis.set_ticks_position('bottom')
   ax.spines['bottom'].set_position(('data',0))
   ax.yaxis.set_ticks_position('left')
   ax.spines['left'].set_position(('data',0))
   ...



Adding a legend
---------------

.. image:: auto_examples/images/plot_exercice_8_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_8.html


.. hint:: Documentation

   * `Legend guide <http://matplotlib.sourceforge.net/users/legend_guide.html>`_
   * `legend() command <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.legend>`_
   * `Legend API <http://matplotlib.sourceforge.net/api/legend_api.html#matplotlib.legend.Legend>`_

.. tip::

    Let's add a legend in the upper left corner. This only requires
    adding the keyword argument label (that will be used in the legend
    box) to the plot commands.

::

   ...
   pl.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
   pl.plot(X, S, color="red",  linewidth=2.5, linestyle="-", label="sine")

   pl.legend(loc='upper left')
   ...



Annotate some points
--------------------

.. image:: auto_examples/images/plot_exercice_9_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_9.html


.. hint:: Documentation

   * `Annotating axis <http://matplotlib.sourceforge.net/users/annotations_guide.html>`_
   * `annotate() command <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.annotate>`_

.. tip::

    Let's annotate some interesting points using the annotate command. We
    chose the 2π/3 value and we want to annotate both the sine and the
    cosine. We'll first draw a marker on the curve as well as a straight
    dotted line. Then, we'll use the annotate command to display some
    text with an arrow.

::

   ...

   t = 2 * np.pi / 3
   pl.plot([t, t], [0, np.cos(t)], color='blue', linewidth=2.5, linestyle="--")
   pl.scatter([t, ], [np.cos(t), ], 50, color='blue')

   pl.annotate(r'$sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
               xy=(t, np.sin(t)), xycoords='data',
               xytext=(+10, +30), textcoords='offset points', fontsize=16,
               arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

   pl.plot([t, t],[0, np.sin(t)], color='red', linewidth=2.5, linestyle="--")
   pl.scatter([t, ],[np.sin(t), ], 50, color='red')

   pl.annotate(r'$cos(\frac{2\pi}{3})=-\frac{1}{2}$',
               xy=(t, np.cos(t)), xycoords='data',
               xytext=(-90, -50), textcoords='offset points', fontsize=16,
               arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
   ...



Devil is in the details
------------------------

.. image:: auto_examples/images/plot_exercice_10_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_exercice_10.html

.. hint:: Documentation

   * `Artists <http://matplotlib.sourceforge.net/api/artist_api.html>`_
   * `BBox <http://matplotlib.sourceforge.net/api/artist_api.html#matplotlib.text.Text.set_bbox>`_

.. tip::

    The tick labels are now hardly visible because of the blue and red
    lines. We can make them bigger and we can also adjust their
    properties such that they'll be rendered on a semi-transparent white
    background. This will allow us to see both the data and the labels.

::

   ...
   for label in ax.get_xticklabels() + ax.get_yticklabels():
       label.set_fontsize(16)
       label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65))
   ...




Figures, Subplots, Axes and Ticks
=================================

A **"figure"** in matplotlib means the whole window in the user interface.
Within this figure there can be **"subplots"**.

.. tip::

    So far we have used implicit figure and axes creation. This is handy
    for fast plots. We can have more control over the display using
    figure, subplot, and axes explicitly.  While subplot positions the
    plots in a regular grid, axes allows free placement within the
    figure. Both can be useful depending on your intention. We've already
    worked with figures and subplots without explicitly calling them.
    When we call plot, matplotlib calls ``gca()`` to get the current axes
    and gca in turn calls ``gcf()`` to get the current figure. If there
    is none it calls ``figure()`` to make one, strictly speaking, to make
    a ``subplot(111)``. Let's look at the details.

Figures
-------

.. tip::

    A figure is the windows in the GUI that has "Figure #" as title.
    Figures are numbered starting from 1 as opposed to the normal Python
    way starting from 0. This is clearly MATLAB-style.  There are several
    parameters that determine what the figure looks like:

==============  ======================= ============================================
Argument        Default                 Description
==============  ======================= ============================================
``num``         ``1``                   number of figure
``figsize``     ``figure.figsize``      figure size in in inches (width, height)
``dpi``         ``figure.dpi``          resolution in dots per inch
``facecolor``   ``figure.facecolor``    color of the drawing background
``edgecolor``   ``figure.edgecolor``    color of edge around the drawing background
``frameon``     ``True``                draw figure frame or not
==============  ======================= ============================================

.. tip::

    The defaults can be specified in the resource file and will be used
    most of the time. Only the number of the figure is frequently
    changed.

    As with other objects, you can set figure properties also setp or
    with the set_something methods.

    When you work with the GUI you can close a figure by clicking on the
    x in the upper right corner. But you can close a figure
    programmatically by calling close. Depending on the argument it
    closes (1) the current figure (no argument), (2) a specific figure
    (figure number or figure instance as argument), or (3) all figures
    (``"all"`` as argument).

::

    pl.close(1)     # Closes figure 1


Subplots
--------

.. tip::

    With subplot you can arrange plots in a regular grid. You need to
    specify the number of rows and columns and the number of the plot.
    Note that the `gridspec
    <http://matplotlib.sourceforge.net/users/gridspec.html>`_ command is
    a more powerful alternative.

.. avoid an ugly interplay between 'tip' and the images below: we want a
   line-return

|clear-floats|

.. image:: auto_examples/images/plot_subplot-horizontal_1.png
   :scale: 28
   :target: auto_examples/plot_subplot-horizontal.html
.. image:: auto_examples/images/plot_subplot-vertical_1.png
   :scale: 28
   :target: auto_examples/plot_subplot-vertical.html
.. image:: auto_examples/images/plot_subplot-grid_1.png
   :scale: 28
   :target: auto_examples/plot_subplot-grid.html
.. image:: auto_examples/images/plot_gridspec_1.png
   :scale: 28
   :target: auto_examples/plot_gridspec.html


Axes
----

Axes are very similar to subplots but allow placement of plots at any location
in the figure. So if we want to put a smaller plot inside a bigger one we do
so with axes.

.. image:: auto_examples/images/plot_axes_1.png
   :scale: 35
   :target: auto_examples/plot_axes.html
.. image:: auto_examples/images/plot_axes-2_1.png
   :scale: 35
   :target: auto_examples/plot_axes-2.html


Ticks
-----

Well formatted ticks are an important part of publishing-ready
figures. Matplotlib provides a totally configurable system for ticks. There are
tick locators to specify where ticks should appear and tick formatters to give
ticks the appearance you want. Major and minor ticks can be located and
formatted independently from each other. Per default minor ticks are not shown,
i.e. there is only an empty list for them because it is as ``NullLocator`` (see
below).

Tick Locators
.............

Tick locators control the positions of the ticks. They are set as
follows::

    ax = pl.gca()
    ax.xaxis.set_major_locator(eval(locator))

There are several locators for different kind of requirements:

.. image:: auto_examples/images/plot_ticks_1.png
    :scale: 60
    :target: auto_examples/plot_ticks.html


All of these locators derive from the base class :class:`matplotlib.ticker.Locator`.
You can make your own locator deriving from it. Handling dates as ticks can be
especially tricky. Therefore, matplotlib provides special locators in
matplotlib.dates.


Other Types of Plots: examples and exercises
=============================================

.. image:: auto_examples/images/plot_plot_1.png
   :scale: 39
   :target: `Regular Plots`_
.. image:: auto_examples/images/plot_scatter_1.png
   :scale: 39
   :target: `Scatter Plots`_
.. image:: auto_examples/images/plot_bar_1.png
   :scale: 39
   :target: `Bar Plots`_
.. image:: auto_examples/images/plot_contour_1.png
   :scale: 39
   :target: `Contour Plots`_
.. image:: auto_examples/images/plot_imshow_1.png
   :scale: 39
   :target: `Imshow`_
.. image:: auto_examples/images/plot_quiver_1.png
   :scale: 39
   :target: `Quiver Plots`_
.. image:: auto_examples/images/plot_pie_1.png
   :scale: 39
   :target: `Pie Charts`_
.. image:: auto_examples/images/plot_grid_1.png
   :scale: 39
   :target: `Grids`_
.. image:: auto_examples/images/plot_multiplot_1.png
   :scale: 39
   :target: `Multi Plots`_
.. image:: auto_examples/images/plot_polar_1.png
   :scale: 39
   :target: `Polar Axis`_
.. image:: auto_examples/images/plot_plot3d_1.png
   :scale: 39
   :target: `3D Plots`_
.. image:: auto_examples/images/plot_text_1.png
   :scale: 39
   :target: `Text`_


Regular Plots
-------------

.. image:: auto_examples/images/plot_plot_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_plot_ex.html

.. hint::

   You need to use the `fill_between
   <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.fill_between>`_
   command.

Starting from the code below, try to reproduce the graphic on the right taking
care of filled areas::

   n = 256
   X = np.linspace(-np.pi, np.pi, n, endpoint=True)
   Y = np.sin(2 * X)

   pl.plot(X, Y + 1, color='blue', alpha=1.00)
   pl.plot(X, Y - 1, color='blue', alpha=1.00)

Click on the figure for solution.


Scatter Plots
-------------

.. image:: auto_examples/images/plot_scatter_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_scatter_ex.html

.. hint::

   Color is given by angle of (X,Y).


Starting from the code below, try to reproduce the graphic on the right taking
care of marker size, color and transparency.

::

   n = 1024
   X = np.random.normal(0,1,n)
   Y = np.random.normal(0,1,n)

   pl.scatter(X,Y)

Click on figure for solution.


Bar Plots
---------

.. image:: auto_examples/images/plot_bar_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_bar_ex.html

.. hint::

   You need to take care of text alignment.


Starting from the code below, try to reproduce the graphic on the right by
adding labels for red bars.

::

   n = 12
   X = np.arange(n)
   Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
   Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

   pl.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
   pl.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

   for x, y in zip(X, Y1):
       pl.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')

   pl.ylim(-1.25, +1.25)

Click on figure for solution.


Contour Plots
-------------

.. image:: auto_examples/images/plot_contour_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_contour_ex.html


.. hint::

   You need to use the `clabel
   <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.clabel>`_
   command.

Starting from the code below, try to reproduce the graphic on the right taking
care of the colormap (see `Colormaps`_ below).

::

   def f(x, y):
       return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 -y ** 2)

   n = 256
   x = np.linspace(-3, 3, n)
   y = np.linspace(-3, 3, n)
   X, Y = np.meshgrid(x, y)

   pl.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap='jet')
   C = pl.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)

Click on figure for solution.



Imshow
------

.. image:: auto_examples/images/plot_imshow_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_imshow_ex.html


.. hint::

   You need to take care of the ``origin`` of the image in the imshow command and
   use a `colorbar
   <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.colorbar>`_


Starting from the code below, try to reproduce the graphic on the right taking
care of colormap, image interpolation and origin.

::

   def f(x, y):
       return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

   n = 10
   x = np.linspace(-3, 3, 4 * n)
   y = np.linspace(-3, 3, 3 * n)
   X, Y = np.meshgrid(x, y)
   pl.imshow(f(X, Y))

Click on the figure for the solution.


Pie Charts
----------

.. image:: auto_examples/images/plot_pie_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_pie_ex.html


.. hint::

   You need to modify Z.

Starting from the code below, try to reproduce the graphic on the right taking
care of colors and slices size.

::

   Z = np.random.uniform(0, 1, 20)
   pl.pie(Z)

Click on the figure for the solution.



Quiver Plots
------------

.. image:: auto_examples/images/plot_quiver_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_quiver_ex.html


.. hint::

   You need to draw arrows twice.

Starting from the code above, try to reproduce the graphic on the right taking
care of colors and orientations.

::

   n = 8
   X, Y = np.mgrid[0:n, 0:n]
   pl.quiver(X, Y)

Click on figure for solution.


Grids
-----

.. image:: auto_examples/images/plot_grid_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_grid_ex.html


Starting from the code below, try to reproduce the graphic on the right taking
care of line styles.

::

   axes = pl.gca()
   axes.set_xlim(0, 4)
   axes.set_ylim(0, 3)
   axes.set_xticklabels([])
   axes.set_yticklabels([])


Click on figure for solution.


Multi Plots
-----------

.. image:: auto_examples/images/plot_multiplot_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_multiplot_ex.html

.. hint::

   You can use several subplots with different partition.


Starting from the code below, try to reproduce the graphic on the right.

::

   pl.subplot(2, 2, 1)
   pl.subplot(2, 2, 3)
   pl.subplot(2, 2, 4)

Click on figure for solution.


Polar Axis
----------

.. image:: auto_examples/images/plot_polar_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_polar_ex.html


.. hint::

   You only need to modify the ``axes`` line


Starting from the code below, try to reproduce the graphic on the right.

::

   pl.axes([0, 0, 1, 1])

   N = 20
   theta = np.arange(0., 2 * np.pi, 2 * np.pi / N)
   radii = 10 * np.random.rand(N)
   width = np.pi / 4 * np.random.rand(N)
   bars = pl.bar(theta, radii, width=width, bottom=0.0)

   for r, bar in zip(radii, bars):
       bar.set_facecolor(cm.jet(r / 10.))
       bar.set_alpha(0.5)

Click on figure for solution.


3D Plots
--------

.. image:: auto_examples/images/plot_plot3d_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_plot3d_ex.html


.. hint::

   You need to use `contourf
   <http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.contourf>`_


Starting from the code below, try to reproduce the graphic on the right.

::

   from mpl_toolkits.mplot3d import Axes3D

   fig = pl.figure()
   ax = Axes3D(fig)
   X = np.arange(-4, 4, 0.25)
   Y = np.arange(-4, 4, 0.25)
   X, Y = np.meshgrid(X, Y)
   R = np.sqrt(X**2 + Y**2)
   Z = np.sin(R)

   ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

Click on figure for solution.

.. seealso:: :ref:`mayavi-label`

Text
----


.. image:: auto_examples/images/plot_text_ex_1.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_text_ex.html


.. hint::

   Have a look at the `matplotlib logo
   <http://matplotlib.sourceforge.net/examples/api/logo2.html>`_.

Try to do the same from scratch !

Click on figure for solution.


Beyond this tutorial
====================

Matplotlib benefits from extensive documentation as well as a large
community of users and developpers. Here are some links of interest:

Tutorials
---------

.. hlist::

  * `Pyplot tutorial <http://matplotlib.sourceforge.net/users/pyplot_tutorial.html>`_

    - Introduction
    - Controlling line properties
    - Working with multiple figures and axes
    - Working with text

  * `Image tutorial <http://matplotlib.sourceforge.net/users/image_tutorial.html>`_

    - Startup commands
    - Importing image data into Numpy arrays
    - Plotting numpy arrays as images

  * `Text tutorial <http://matplotlib.sourceforge.net/users/index_text.html>`_

    - Text introduction
    - Basic text commands
    - Text properties and layout
    - Writing mathematical expressions
    - Text rendering With LaTeX
    - Annotating text

  * `Artist tutorial <http://matplotlib.sourceforge.net/users/artists.html>`_

    - Introduction
    - Customizing your objects
    - Object containers
    - Figure container
    - Axes container
    - Axis containers
    - Tick containers

  * `Path tutorial <http://matplotlib.sourceforge.net/users/path_tutorial.html>`_

    - Introduction
    - Bézier example
    - Compound paths

  * `Transforms tutorial <http://matplotlib.sourceforge.net/users/transforms_tutorial.html>`_

    - Introduction
    - Data coordinates
    - Axes coordinates
    - Blended transformations
    - Using offset transforms to create a shadow effect
    - The transformation pipeline



Matplotlib documentation
------------------------

* `User guide <http://matplotlib.sourceforge.net/users/index.html>`_

* `FAQ <http://matplotlib.sourceforge.net/faq/index.html>`_

  - Installation
  - Usage
  - How-To
  - Troubleshooting
  - Environment Variables

* `Screenshots <http://matplotlib.sourceforge.net/users/screenshots.html>`_


Code documentation
------------------

The code is well documented and you can quickly access a specific command
from within a python session:

::

   >>> import pylab as pl
   >>> help(pl.plot)
   Help on function plot in module matplotlib.pyplot:

   plot(*args, **kwargs)
      Plot lines and/or markers to the
      :class:`~matplotlib.axes.Axes`.  *args* is a variable length
      argument, allowing for multiple *x*, *y* pairs with an
      optional format string.  For example, each of the following is
      legal::

          plot(x, y)         # plot x and y using default line style and color
          plot(x, y, 'bo')   # plot x and y using blue circle markers
          plot(y)            # plot y using x as index array 0..N-1
          plot(y, 'r+')      # ditto, but with red plusses

      If *x* and/or *y* is 2-dimensional, then the corresponding columns
      will be plotted.
      ...

Galleries
---------

The `matplotlib gallery <http://matplotlib.sourceforge.net/gallery.html>`_ is
also incredibly useful when you search how to render a given graphic. Each
example comes with its source.

A smaller gallery is also available `here <http://www.loria.fr/~rougier/coding/gallery/>`_.


Mailing lists
--------------

Finally, there is a `user mailing list
<https://lists.sourceforge.net/lists/listinfo/matplotlib-users>`_ where you can
ask for help and a `developers mailing list
<https://lists.sourceforge.net/lists/listinfo/matplotlib-devel>`_ that is more
technical.



Quick references
================

Here is a set of tables that show main properties and styles.

Line properties
----------------

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Property
     - Description
     - Appearance

   * - alpha (or a)
     - alpha transparency on 0-1 scale
     - .. image:: auto_examples/images/plot_alpha_1.png

   * - antialiased
     - True or False - use antialised rendering
     - .. image:: auto_examples/images/plot_aliased_1.png
       .. image:: auto_examples/images/plot_antialiased_1.png

   * - color (or c)
     - matplotlib color arg
     - .. image:: auto_examples/images/plot_color_1.png

   * - linestyle (or ls)
     - see `Line properties`_
     -

   * - linewidth (or lw)
     - float, the line width in points
     - .. image:: auto_examples/images/plot_linewidth_1.png

   * - solid_capstyle
     - Cap style for solid lines
     - .. image:: auto_examples/images/plot_solid_capstyle_1.png

   * - solid_joinstyle
     - Join style for solid lines
     - .. image:: auto_examples/images/plot_solid_joinstyle_1.png

   * - dash_capstyle
     - Cap style for dashes
     - .. image:: auto_examples/images/plot_dash_capstyle_1.png

   * - dash_joinstyle
     - Join style for dashes
     - .. image:: auto_examples/images/plot_dash_joinstyle_1.png

   * - marker
     - see `Markers`_
     -

   * - markeredgewidth (mew)
     - line width around the marker symbol
     - .. image:: auto_examples/images/plot_mew_1.png

   * - markeredgecolor (mec)
     - edge color if a marker is used
     - .. image:: auto_examples/images/plot_mec_1.png

   * - markerfacecolor (mfc)
     - face color if a marker is used
     - .. image:: auto_examples/images/plot_mfc_1.png

   * - markersize (ms)
     - size of the marker in points
     - .. image:: auto_examples/images/plot_ms_1.png



Line styles
-----------

.. image:: auto_examples/images/plot_linestyles_1.png

Markers
-------

.. image:: auto_examples/images/plot_markers_1.png
   :scale: 90

Colormaps
---------

All colormaps can be reversed by appending ``_r``. For instance, ``gray_r`` is
the reverse of ``gray``.

If you want to know more about colormaps, checks `Documenting the matplotlib
colormaps <https://gist.github.com/2719900>`_.

.. image:: auto_examples/images/plot_colormaps_1.png
   :scale: 80

