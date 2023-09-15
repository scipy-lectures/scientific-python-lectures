
.. _matplotlib:

.. currentmodule:: matplotlib.pyplot

====================
Matplotlib: plotting
====================

.. sidebar:: **Thanks**

    Many thanks to **Bill Wing** and **Christoph Deil** for review and
    corrections.

**Authors**: *Nicolas Rougier, Mike Müller, Gaël Varoquaux*

.. contents:: Chapter contents
   :local:
   :depth: 1

Introduction
============

.. tip::

    `Matplotlib <https://matplotlib.org/>`__ is probably the most
    used Python package for 2D-graphics. It provides both a quick
    way to visualize data from Python and publication-quality figures in
    many formats.  We are going to explore matplotlib in interactive mode
    covering most common cases.

IPython, Jupyter, and matplotlib modes
---------------------------------------

.. tip::

    The `Jupyter <https://jupyter.org>`_ notebook and the
    `IPython <https://ipython.org/>`_ enhanced interactive Python, are
    tuned for the scientific-computing workflow in Python,
    in combination with Matplotlib:

For interactive matplotlib sessions, turn on the **matplotlib mode**

:IPython console:

  When using the IPython console, use::

    In [1]: %matplotlib

:Jupyter notebook:

  In the notebook, insert, **at the beginning of the
  notebook** the following `magic
  <https://ipython.readthedocs.io/en/stable/interactive/magics.html>`_::

    %matplotlib inline

pyplot
------

.. tip::

    *pyplot* provides a procedural interface to the matplotlib object-oriented
    plotting library. It is modeled closely after Matlab™. Therefore, the
    majority of plotting commands in pyplot have Matlab™ analogs with similar
    arguments.  Important commands are explained with interactive examples.

::

    import matplotlib.pyplot as plt

Simple plot
===========

.. tip::

    In this section, we want to draw the cosine and sine functions on the same
    plot. Starting from the default settings, we'll enrich the figure step by
    step to make it nicer.

    First step is to get the data for the sine and cosine functions:

::

   import numpy as np

   X = np.linspace(-np.pi, np.pi, 256)
   C, S = np.cos(X), np.sin(X)


``X`` is now a numpy array with 256 values ranging from :math:`-\pi` to :math:`+\pi`
(included). ``C`` is the cosine (256 values) and ``S`` is the sine (256
values).

To run the example, you can type them in an IPython interactive session::

    $ ipython --matplotlib

This brings us to the IPython prompt: ::

    IPython 0.13 -- An enhanced Interactive Python.
    ?       -> Introduction to IPython's features.
    %magic  -> Information about IPython's 'magic' % functions.
    help    -> Python's own help system.
    object? -> Details about 'object'. ?object also works, ?? prints more.

.. tip::

    You can also download each of the examples and run it using regular
    python, but you will lose interactive data manipulation::

        $ python plot_exercise_1.py

    You can get source for each step by clicking on the corresponding figure.


Plotting with default settings
-------------------------------

.. image:: auto_examples/exercises/images/sphx_glr_plot_exercise_1_001.png
   :align: right
   :scale: 35
   :target: auto_examples/exercises/plot_exercise_1.html

.. hint:: Documentation

   * `plot tutorial <https://matplotlib.org/users/pyplot_tutorial.html>`_
   * :func:`~plot()` command

.. tip::

    Matplotlib comes with a set of default settings that allow
    customizing all kinds of properties. You can control the defaults of
    almost every property in matplotlib: figure size and dpi, line width,
    color and style, axes, axis and grid properties, text and font
    properties and so on.

|clear-floats|

::

   import numpy as np
   import matplotlib.pyplot as plt

   X = np.linspace(-np.pi, np.pi, 256)
   C, S = np.cos(X), np.sin(X)

   plt.plot(X, C)
   plt.plot(X, S)

   plt.show()


Instantiating defaults
----------------------

.. image:: auto_examples/exercises/images/sphx_glr_plot_exercise_2_001.png
   :align: right
   :scale: 35
   :target: auto_examples/exercises/plot_exercise_2.html

.. hint:: Documentation

   *  `Customizing matplotlib <https://matplotlib.org/users/customizing.html>`_

In the script below, we've instantiated (and commented) all the figure settings
that influence the appearance of the plot.

.. tip::

    The settings have been explicitly set to their default values, but
    now you can interactively play with the values to explore their
    affect (see `Line properties`_ and `Line styles`_ below).

|clear-floats|

::

   import numpy as np
   import matplotlib.pyplot as plt

   # Create a figure of size 8x6 inches, 80 dots per inch
   plt.figure(figsize=(8, 6), dpi=80)

   # Create a new subplot from a grid of 1x1
   plt.subplot(1, 1, 1)

   X = np.linspace(-np.pi, np.pi, 256)
   C, S = np.cos(X), np.sin(X)

   # Plot cosine with a blue continuous line of width 1 (pixels)
   plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-")

   # Plot sine with a green continuous line of width 1 (pixels)
   plt.plot(X, S, color="green", linewidth=1.0, linestyle="-")

   # Set x limits
   plt.xlim(-4.0, 4.0)

   # Set x ticks
   plt.xticks(np.linspace(-4, 4, 9))

   # Set y limits
   plt.ylim(-1.0, 1.0)

   # Set y ticks
   plt.yticks(np.linspace(-1, 1, 5))

   # Save figure using 72 dots per inch
   # plt.savefig("exercise_2.png", dpi=72)

   # Show result on screen
   plt.show()


Changing colors and line widths
--------------------------------

.. image:: auto_examples/exercises/images/sphx_glr_plot_exercise_3_001.png
   :align: right
   :scale: 35
   :target: auto_examples/exercises/plot_exercise_3.html

.. hint:: Documentation

   * `Controlling line properties <https://matplotlib.org/users/pyplot_tutorial.html#controlling-line-properties>`_
   * :class:`~matplotlib.lines.Line2D` API

.. tip::

    First step, we want to have the cosine in blue and the sine in red and a
    slightly thicker line for both of them. We'll also slightly alter the figure
    size to make it more horizontal.

|clear-floats|

::

   ...
   plt.figure(figsize=(10, 6), dpi=80)
   plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
   plt.plot(X, S, color="red",  linewidth=2.5, linestyle="-")
   ...


Setting limits
--------------

.. image:: auto_examples/exercises/images/sphx_glr_plot_exercise_4_001.png
   :align: right
   :scale: 35
   :target: auto_examples/exercises/plot_exercise_4.html

.. hint:: Documentation

   * :func:`xlim()` command
   * :func:`ylim()` command

.. tip::

    Current limits of the figure are a bit too tight and we want to make
    some space in order to clearly see all data points.

|clear-floats|

::

   ...
   plt.xlim(X.min() * 1.1, X.max() * 1.1)
   plt.ylim(C.min() * 1.1, C.max() * 1.1)
   ...



Setting ticks
-------------

.. image:: auto_examples/exercises/images/sphx_glr_plot_exercise_5_001.png
   :align: right
   :scale: 35
   :target: auto_examples/exercises/plot_exercise_5.html

.. hint:: Documentation

   * :func:`xticks()` command
   * :func:`yticks()` command
   * `Tick container <https://matplotlib.org/users/artists.html#axis-container>`_
   * `Tick locating and formatting <https://matplotlib.org/api/ticker_api.html>`_

.. tip::

    Current ticks are not ideal because they do not show the interesting values
    (:math:`\pm \pi`,:math:`\pm \pi`/2) for sine and cosine. We'll change them such that they show
    only these values.

|clear-floats|

::

   ...
   plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
   plt.yticks([-1, 0, +1])
   ...



Setting tick labels
-------------------

.. image:: auto_examples/exercises/images/sphx_glr_plot_exercise_6_001.png
   :align: right
   :scale: 35
   :target: auto_examples/exercises/plot_exercise_6.html


.. hint:: Documentation

   * `Working with text <https://matplotlib.org/users/index_text.html>`_
   * :func:`~xticks()` command
   * :func:`~yticks()` command
   * :meth:`~matplotlib.axes.Axes.set_xticklabels()`
   * :meth:`~matplotlib.axes.Axes.set_yticklabels()`


.. tip::

    Ticks are now properly placed but their label is not very explicit.
    We could guess that 3.142 is :math:`\pi` but it would be better to make it
    explicit. When we set tick values, we can also provide a
    corresponding label in the second argument list. Note that we'll use
    latex to allow for nice rendering of the label.

|clear-floats|

::

   ...
   plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
             [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

   plt.yticks([-1, 0, +1],
             [r'$-1$', r'$0$', r'$+1$'])
   ...



Moving spines
-------------

.. image:: auto_examples/exercises/images/sphx_glr_plot_exercise_7_001.png
   :align: right
   :scale: 35
   :target: auto_examples/exercises/plot_exercise_7.html


.. hint:: Documentation

   * :mod:`~matplotlib.spines` API
   * `Axis container <https://matplotlib.org/users/artists.html#axis-container>`_
   * `Transformations tutorial <https://matplotlib.org/users/transforms_tutorial.html>`_

.. tip::

    Spines are the lines connecting the axis tick marks and noting the
    boundaries of the data area. They can be placed at arbitrary
    positions and until now, they were on the border of the axis. We'll
    change that since we want to have them in the middle. Since there are
    four of them (top/bottom/left/right), we'll discard the top and right
    by setting their color to none and we'll move the bottom and left
    ones to coordinate 0 in data space coordinates.

|clear-floats|

::

   ...
   ax = plt.gca()  # gca stands for 'get current axis'
   ax.spines['right'].set_color('none')
   ax.spines['top'].set_color('none')
   ax.xaxis.set_ticks_position('bottom')
   ax.spines['bottom'].set_position(('data',0))
   ax.yaxis.set_ticks_position('left')
   ax.spines['left'].set_position(('data',0))
   ...



Adding a legend
---------------

.. image:: auto_examples/exercises/images/sphx_glr_plot_exercise_8_001.png
   :align: right
   :scale: 35
   :target: auto_examples/exercises/plot_exercise_8.html


.. hint:: Documentation

   * `Legend guide <https://matplotlib.org/users/legend_guide.html>`_
   * :func:`legend()` command
   * :mod:`~matplotlib.legend` API

.. tip::

    Let's add a legend in the upper left corner. This only requires
    adding the keyword argument label (that will be used in the legend
    box) to the plot commands.

|clear-floats|

::

   ...
   plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
   plt.plot(X, S, color="red",  linewidth=2.5, linestyle="-", label="sine")

   plt.legend(loc='upper left')
   ...



Annotate some points
--------------------

.. image:: auto_examples/exercises/images/sphx_glr_plot_exercise_9_001.png
   :align: right
   :scale: 35
   :target: auto_examples/exercises/plot_exercise_9.html


.. hint:: Documentation

   * `Annotating axis <https://matplotlib.org/users/annotations_guide.html>`_
   * :func:`annotate()` command

.. tip::

    Let's annotate some interesting points using the annotate command. We
    chose the :math:`2\pi / 3` value and we want to annotate both the sine and the
    cosine. We'll first draw a marker on the curve as well as a straight
    dotted line. Then, we'll use the annotate command to display some
    text with an arrow.

|clear-floats|

::

   ...

   t = 2 * np.pi / 3
   plt.plot([t, t], [0, np.cos(t)], color='blue', linewidth=2.5, linestyle="--")
   plt.scatter([t, ], [np.cos(t), ], 50, color='blue')

   plt.annotate(r'$cos(\frac{2\pi}{3})=-\frac{1}{2}$',
                xy=(t, np.cos(t)), xycoords='data',
                xytext=(-90, -50), textcoords='offset points', fontsize=16,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

   plt.plot([t, t],[0, np.sin(t)], color='red', linewidth=2.5, linestyle="--")
   plt.scatter([t, ],[np.sin(t), ], 50, color='red')

   plt.annotate(r'$sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
                xy=(t, np.sin(t)), xycoords='data',
                xytext=(+10, +30), textcoords='offset points', fontsize=16,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
   ...



Devil is in the details
------------------------

.. image:: auto_examples/exercises/images/sphx_glr_plot_exercise_10_001.png
   :align: right
   :scale: 35
   :target: auto_examples/exercises/plot_exercise_10.html

.. hint:: Documentation

   * :mod:`~matplotlib.artist` API
   * :meth:`~matplotlib.text.Text.set_bbox()` method

.. tip::

    The tick labels are now hardly visible because of the blue and red
    lines. We can make them bigger and we can also adjust their
    properties such that they'll be rendered on a semi-transparent white
    background. This will allow us to see both the data and the labels.

|clear-floats|

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

    So far we have used implicit figure and axes creation. This is handy for
    fast plots. We can have more control over the display using figure,
    subplot, and axes explicitly.  While subplot positions the plots in a
    regular grid, axes allows free placement within the figure. Both can be
    useful depending on your intention. We've already worked with figures and
    subplots without explicitly calling them.  When we call plot, matplotlib
    calls :func:`gca` to get the current axes and gca in turn calls :func:`gcf` to
    get the current figure. If there is none it calls :func:`figure` to make one,
    strictly speaking, to make a ``subplot(111)``. Let's look at the details.

Figures
-------

.. tip::

    A figure is the windows in the GUI that has "Figure #" as title.  Figures
    are numbered starting from 1 as opposed to the normal Python way starting
    from 0. This is clearly MATLAB-style.  There are several parameters that
    determine what the figure looks like:

==============  ======================= ============================================
Argument        Default                 Description
==============  ======================= ============================================
``num``         ``1``                   number of figure
``figsize``     ``figure.figsize``      figure size in inches (width, height)
``dpi``         ``figure.dpi``          resolution in dots per inch
``facecolor``   ``figure.facecolor``    color of the drawing background
``edgecolor``   ``figure.edgecolor``    color of edge around the drawing background
``frameon``     ``True``                draw figure frame or not
==============  ======================= ============================================

.. tip::

    The defaults can be specified in the resource file and will be used most of
    the time. Only the number of the figure is frequently changed.

    As with other objects, you can set figure properties also setp or with the
    set_something methods.

    When you work with the GUI you can close a figure by clicking on the x in
    the upper right corner. But you can close a figure programmatically by
    calling close. Depending on the argument it closes (1) the current figure
    (no argument), (2) a specific figure (figure number or figure instance as
    argument), or (3) all figures (``"all"`` as argument).

::

    plt.close(1)     # Closes figure 1


Subplots
--------

.. tip::

    With subplot you can arrange plots in a regular grid. You need to specify
    the number of rows and columns and the number of the plot.  Note that the
    `gridspec <https://matplotlib.org/users/gridspec.html>`_ command
    is a more powerful alternative.

.. avoid an ugly interplay between 'tip' and the images below: we want a
   line-return

|clear-floats|

.. image:: auto_examples/images/sphx_glr_plot_subplot-horizontal_001.png
   :scale: 25
   :target: auto_examples/plot_subplot-horizontal.html
.. image:: auto_examples/images/sphx_glr_plot_subplot-vertical_001.png
   :scale: 25
   :target: auto_examples/plot_subplot-vertical.html
.. image:: auto_examples/images/sphx_glr_plot_subplot-grid_001.png
   :scale: 25
   :target: auto_examples/plot_subplot-grid.html
.. image:: auto_examples/images/sphx_glr_plot_gridspec_001.png
   :scale: 25
   :target: auto_examples/plot_gridspec.html


Axes
----

Axes are very similar to subplots but allow placement of plots at any location
in the figure. So if we want to put a smaller plot inside a bigger one we do
so with axes.

.. image:: auto_examples/images/sphx_glr_plot_axes_001.png
   :scale: 35
   :target: auto_examples/plot_axes.html
.. image:: auto_examples/images/sphx_glr_plot_axes-2_001.png
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

    ax = plt.gca()
    ax.xaxis.set_major_locator(eval(locator))

There are several locators for different kind of requirements:

.. raw:: latex

    ~

.. image:: auto_examples/options/images/sphx_glr_plot_ticks_001.png
    :scale: 60
    :target: auto_examples/options/plot_ticks.html

.. raw:: latex

    ~

All of these locators derive from the base class :class:`matplotlib.ticker.Locator`.
You can make your own locator deriving from it. Handling dates as ticks can be
especially tricky. Therefore, matplotlib provides special locators in
matplotlib.dates.


Other Types of Plots: examples and exercises
=============================================

.. image:: auto_examples/pretty_plots/images/sphx_glr_plot_plot_ext_001.png
   :scale: 39
   :target: `Regular Plots`_
.. image:: auto_examples/pretty_plots/images/sphx_glr_plot_scatter_ext_001.png
   :scale: 39
   :target: `Scatter Plots`_
.. image:: auto_examples/pretty_plots/images/sphx_glr_plot_bar_ext_001.png
   :scale: 39
   :target: `Bar Plots`_
.. image:: auto_examples/pretty_plots/images/sphx_glr_plot_contour_ext_001.png
   :scale: 39
   :target: `Contour Plots`_
.. image:: auto_examples/pretty_plots/images/sphx_glr_plot_imshow_ext_001.png
   :scale: 39
   :target: `Imshow`_
.. image:: auto_examples/pretty_plots/images/sphx_glr_plot_quiver_ext_001.png
   :scale: 39
   :target: `Quiver Plots`_
.. image:: auto_examples/pretty_plots/images/sphx_glr_plot_pie_ext_001.png
   :scale: 39
   :target: `Pie Charts`_
.. image:: auto_examples/pretty_plots/images/sphx_glr_plot_grid_ext_001.png
   :scale: 39
   :target: `Grids`_
.. image:: auto_examples/pretty_plots/images/sphx_glr_plot_multiplot_ext_001.png
   :scale: 39
   :target: `Multi Plots`_
.. image:: auto_examples/pretty_plots/images/sphx_glr_plot_polar_ext_001.png
   :scale: 39
   :target: `Polar Axis`_
.. image:: auto_examples/pretty_plots/images/sphx_glr_plot_plot3d_ext_001.png
   :scale: 39
   :target: `3D Plots`_
.. image:: auto_examples/pretty_plots/images/sphx_glr_plot_text_ext_001.png
   :scale: 39
   :target: `Text`_


Regular Plots
-------------

.. image:: auto_examples/images/sphx_glr_plot_plot_001.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_plot.html

Starting from the code below, try to reproduce the graphic taking
care of filled areas:

.. hint::

   You need to use the :func:`fill_between()` command.


::

   n = 256
   X = np.linspace(-np.pi, np.pi, n)
   Y = np.sin(2 * X)

   plt.plot(X, Y + 1, color='blue', alpha=1.00)
   plt.plot(X, Y - 1, color='blue', alpha=1.00)

Click on the figure for solution.


Scatter Plots
-------------

.. image:: auto_examples/images/sphx_glr_plot_scatter_001.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_scatter.html

Starting from the code below, try to reproduce the graphic taking
care of marker size, color and transparency.

.. hint::

   Color is given by angle of (X,Y).


::

   n = 1024
   rng = np.random.default_rng()
   X = rng.normal(0,1,n)
   Y = rng.normal(0,1,n)

   plt.scatter(X,Y)

Click on figure for solution.


Bar Plots
---------

.. image:: auto_examples/images/sphx_glr_plot_bar_001.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_bar.html

Starting from the code below, try to reproduce the graphic by
adding labels for red bars.

.. hint::

   You need to take care of text alignment.

|clear-floats|

::

   n = 12
   X = np.arange(n)
   rng = np.random.default_rng()
   Y1 = (1 - X / float(n)) * rng.uniform(0.5, 1.0, n)
   Y2 = (1 - X / float(n)) * rng.uniform(0.5, 1.0, n)

   plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
   plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

   for x, y in zip(X, Y1):
       plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')

   plt.ylim(-1.25, +1.25)

Click on figure for solution.


Contour Plots
-------------

.. image:: auto_examples/images/sphx_glr_plot_contour_001.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_contour.html


Starting from the code below, try to reproduce the graphic taking
care of the colormap (see `Colormaps`_ below).

.. hint::

   You need to use the :func:`clabel()` command.

::

   def f(x, y):
       return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 -y ** 2)

   n = 256
   x = np.linspace(-3, 3, n)
   y = np.linspace(-3, 3, n)
   X, Y = np.meshgrid(x, y)

   plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap='jet')
   C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)

Click on figure for solution.



Imshow
------

.. image:: auto_examples/images/sphx_glr_plot_imshow_001.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_imshow.html


Starting from the code below, try to reproduce the graphic taking
care of colormap, image interpolation and origin.

.. hint::

   You need to take care of the ``origin`` of the image in the imshow command and
   use a :func:`colorbar()`


::

   def f(x, y):
       return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)

   n = 10
   x = np.linspace(-3, 3, 4 * n)
   y = np.linspace(-3, 3, 3 * n)
   X, Y = np.meshgrid(x, y)
   plt.imshow(f(X, Y))

Click on the figure for the solution.


Pie Charts
----------

.. image:: auto_examples/images/sphx_glr_plot_pie_001.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_pie.html


Starting from the code below, try to reproduce the graphic taking
care of colors and slices size.

.. hint::

   You need to modify Z.

::

   rng = np.random.default_rng()
   Z = rng.uniform(0, 1, 20)
   plt.pie(Z)

Click on the figure for the solution.



Quiver Plots
------------

.. image:: auto_examples/images/sphx_glr_plot_quiver_001.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_quiver.html


Starting from the code below, try to reproduce the graphic taking
care of colors and orientations.

.. hint::

   You need to draw arrows twice.

::

   n = 8
   X, Y = np.mgrid[0:n, 0:n]
   plt.quiver(X, Y)

Click on figure for solution.


Grids
-----

.. image:: auto_examples/images/sphx_glr_plot_grid_001.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_grid.html


Starting from the code below, try to reproduce the graphic taking
care of line styles.

::

   axes = plt.gca()
   axes.set_xlim(0, 4)
   axes.set_ylim(0, 3)
   axes.set_xticklabels([])
   axes.set_yticklabels([])


Click on figure for solution.


Multi Plots
-----------

.. image:: auto_examples/images/sphx_glr_plot_multiplot_001.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_multiplot.html

Starting from the code below, try to reproduce the graphic.

.. hint::

   You can use several subplots with different partition.


::

   plt.subplot(2, 2, 1)
   plt.subplot(2, 2, 3)
   plt.subplot(2, 2, 4)

Click on figure for solution.


Polar Axis
----------

.. image:: auto_examples/images/sphx_glr_plot_polar_001.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_polar.html


.. hint::

   You only need to modify the ``axes`` line

Starting from the code below, try to reproduce the graphic.


::

   plt.axes([0, 0, 1, 1])

   N = 20
   theta = np.arange(0., 2 * np.pi, 2 * np.pi / N)
   rng = np.random.default_rng()
   radii = 10 * rng.random(N)
   width = np.pi / 4 * rng.random(N)
   bars = plt.bar(theta, radii, width=width, bottom=0.0)

   for r, bar in zip(radii, bars):
       bar.set_facecolor(plt.cm.jet(r / 10.))
       bar.set_alpha(0.5)

Click on figure for solution.


3D Plots
--------

.. image:: auto_examples/images/sphx_glr_plot_plot3d_001.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_plot3d.html

Starting from the code below, try to reproduce the graphic.

.. hint::

   You need to use :func:`contourf()`


::

   from mpl_toolkits.mplot3d import Axes3D

   fig = plt.figure()
   ax = Axes3D(fig)
   X = np.arange(-4, 4, 0.25)
   Y = np.arange(-4, 4, 0.25)
   X, Y = np.meshgrid(X, Y)
   R = np.sqrt(X**2 + Y**2)
   Z = np.sin(R)

   ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

Click on figure for solution.

Text
----


.. image:: auto_examples/images/sphx_glr_plot_text_001.png
   :align: right
   :scale: 35
   :target: auto_examples/plot_text.html


Try to do the same from scratch !

.. hint::

   Have a look at the `matplotlib logo
   <https://matplotlib.org/examples/api/logo2.html>`_.


Click on figure for solution.

|

____

|

.. topic:: **Quick read**

   If you want to do a first quick pass through the Scientific Python Lectures
   to learn the ecosystem, you can directly skip to the next chapter:
   :ref:`scipy`.

   The remainder of this chapter is not necessary to follow the rest of
   the intro part. But be sure to come back and finish this chapter later.

Beyond this tutorial
====================

Matplotlib benefits from extensive documentation as well as a large
community of users and developers. Here are some links of interest:

Tutorials
---------

.. hlist::

  * `Pyplot tutorial <https://matplotlib.org/users/pyplot_tutorial.html>`_

    - Introduction
    - Controlling line properties
    - Working with multiple figures and axes
    - Working with text

  * `Image tutorial <https://matplotlib.org/users/image_tutorial.html>`_

    - Startup commands
    - Importing image data into NumPy arrays
    - Plotting NumPy arrays as images

  * `Text tutorial <https://matplotlib.org/users/index_text.html>`_

    - Text introduction
    - Basic text commands
    - Text properties and layout
    - Writing mathematical expressions
    - Text rendering With LaTeX
    - Annotating text

  * `Artist tutorial <https://matplotlib.org/users/artists.html>`_

    - Introduction
    - Customizing your objects
    - Object containers
    - Figure container
    - Axes container
    - Axis containers
    - Tick containers

  * `Path tutorial <https://matplotlib.org/users/path_tutorial.html>`_

    - Introduction
    - Bézier example
    - Compound paths

  * `Transforms tutorial <https://matplotlib.org/users/transforms_tutorial.html>`_

    - Introduction
    - Data coordinates
    - Axes coordinates
    - Blended transformations
    - Using offset transforms to create a shadow effect
    - The transformation pipeline



Matplotlib documentation
------------------------

.. hlist::

  * `User guide <https://matplotlib.org/users/index.html>`_

  * `FAQ <https://matplotlib.org/faq/index.html>`_

    - Installation
    - Usage
    - How-To
    - Troubleshooting
    - Environment Variables

  * `Screenshots <https://matplotlib.org/users/screenshots.html>`_


Code documentation
------------------

The code is well documented and you can quickly access a specific command
from within a python session:

::

   >>> import matplotlib.pyplot as plt
   >>> help(plt.plot)
   Help on function plot in module matplotlib.pyplot:
   <BLANKLINE>
   plot(*args: ...) -> 'list[Line2D]'
       Plot y versus x as lines and/or markers.
   <BLANKLINE>
       Call signatures::
   <BLANKLINE>
           plot([x], y, [fmt], *, data=None, **kwargs)
           plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)
   ...


Galleries
---------

The `matplotlib gallery <https://matplotlib.org/gallery.html>`_ is
also incredibly useful when you search how to render a given graphic. Each
example comes with its source.


Mailing lists
--------------

Finally, there is a `user mailing list
<https://mail.python.org/mailman/listinfo/matplotlib-users>`_ where you can
ask for help and a `developers mailing list
<https://mail.python.org/mailman/listinfo/matplotlib-devel>`_ that is more
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
     - .. image:: auto_examples/options/images/sphx_glr_plot_alpha_001.png

   * - antialiased
     - True or False - use antialised rendering
     - .. image:: auto_examples/options/images/sphx_glr_plot_aliased_001.png
       .. image:: auto_examples/options/images/sphx_glr_plot_antialiased_001.png

   * - color (or c)
     - matplotlib color arg
     - .. image:: auto_examples/options/images/sphx_glr_plot_color_001.png

   * - linestyle (or ls)
     - see `Line properties`_
     -

   * - linewidth (or lw)
     - float, the line width in points
     - .. image:: auto_examples/options/images/sphx_glr_plot_linewidth_001.png

   * - solid_capstyle
     - Cap style for solid lines
     - .. image:: auto_examples/options/images/sphx_glr_plot_solid_capstyle_001.png

   * - solid_joinstyle
     - Join style for solid lines
     - .. image:: auto_examples/options/images/sphx_glr_plot_solid_joinstyle_001.png

   * - dash_capstyle
     - Cap style for dashes
     - .. image:: auto_examples/options/images/sphx_glr_plot_dash_capstyle_001.png

   * - dash_joinstyle
     - Join style for dashes
     - .. image:: auto_examples/options/images/sphx_glr_plot_dash_joinstyle_001.png

   * - marker
     - see `Markers`_
     -

   * - markeredgewidth (mew)
     - line width around the marker symbol
     - .. image:: auto_examples/options/images/sphx_glr_plot_mew_001.png

   * - markeredgecolor (mec)
     - edge color if a marker is used
     - .. image:: auto_examples/options/images/sphx_glr_plot_mec_001.png

   * - markerfacecolor (mfc)
     - face color if a marker is used
     - .. image:: auto_examples/options/images/sphx_glr_plot_mfc_001.png

   * - markersize (ms)
     - size of the marker in points
     - .. image:: auto_examples/options/images/sphx_glr_plot_ms_001.png



Line styles
-----------

.. image:: auto_examples/options/images/sphx_glr_plot_linestyles_001.png

Markers
-------

.. image:: auto_examples/options/images/sphx_glr_plot_markers_001.png
   :scale: 90

Colormaps
---------

All colormaps can be reversed by appending ``_r``. For instance, ``gray_r`` is
the reverse of ``gray``.

If you want to know more about colormaps, check the `documentation on Colormaps in matplotlib <https://matplotlib.org/tutorials/colors/colormaps.html>`_.

.. image:: auto_examples/options/images/sphx_glr_plot_colormaps_001.png
   :scale: 80


Full code examples
==================

.. include::  auto_examples/index.rst
    :start-line: 1
