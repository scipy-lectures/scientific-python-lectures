"""source code for figure
numpy_fancy_indexing.(png/pdf) used in Sect. 1.3.1.7 Fancy indexing

Requirement: PyX>=0.14  (pip installable)

"""

import os
import sys
from math import cos, radians, sin
from pyx import canvas, color, path, text, unit


def framebox(
    nx, ny, framecolor, nw=1, nh=1, boxsize=1, framefactor=0.1, reducedsize=False
):
    """draw color frame in one or across several boxes

    The drawing is done on the global canvas called c.

    nx, ny:      coordinates of lower left corner in number of boxes
    framecolor:  color
    nw, nh:      width and height of frame in boxes
    framefactor: framewidth as fraction of boxsize
    boxsize:     side length of fox
    reducedsize: if True, the size of the frame is reduced by the framewidth

    """
    x = nx * boxsize
    y = ny * boxsize
    w = nw * boxsize
    h = nh * boxsize
    if reducedsize:
        outer_offset = framefactor * boxsize
        inner_offset = 2 * outer_offset
    else:
        outer_offset = 0
        inner_offset = framefactor * boxsize
    p = (
        path.rect(
            x + outer_offset,
            y + outer_offset,
            w - 2 * outer_offset,
            h - 2 * outer_offset,
        )
        + path.rect(
            x + inner_offset,
            y + inner_offset,
            w - 2 * inner_offset,
            h - 2 * inner_offset,
        ).reversed()
    )
    c.fill(p, [framecolor])


ncols = 6
nrows = ncols
boxsize = 1
angle = radians(40)
reducedboxsize = 0.65 * boxsize

ex1color = color.rgb(0, 0.5, 0)
ex2color = color.rgb(0, 0, 0.7)
ex3color = color.rgb(0.7, 0, 0)

text.set(text.LatexRunner)
preamble = r"""\usepackage[T1]{fontenc}
               \usepackage{bera}
               \renewcommand*\familydefault{\ttdefault}
               \usepackage{color}"""
for nr, elem in enumerate((ex1color, ex2color, ex3color)):
    preamble = (
        preamble
        + rf"\definecolor{{ex{nr + 1}color}}{{rgb}}{{{elem.r}, {elem.g}, {elem.b}}}"
    )
text.preamble(preamble)
unit.set(xscale=1.2)

c = canvas.canvas()
for n in range(5):
    framebox(n + 1, ncols - n - 1, ex1color)
for nx in (0, 2, 5):
    framebox(nx, 0, ex2color, nh=3, reducedsize=True)
for n in (0, 2, 5):
    framebox(2, (ncols - n - 1), ex3color)

for nx in range(ncols + 1):
    p = path.path(
        path.moveto(nx * boxsize, 0),
        path.lineto(nx * boxsize, ncols * boxsize),
        path.rlineto(reducedboxsize * cos(angle), reducedboxsize * sin(angle)),
    )
    c.stroke(p)
for ny in range(nrows + 1):
    p = path.path(
        path.moveto(0, ny * boxsize),
        path.lineto(nrows * boxsize, ny * boxsize),
        path.rlineto(reducedboxsize * cos(angle), reducedboxsize * sin(angle)),
    )
    c.stroke(p)
p = path.path(
    path.moveto(
        ncols * boxsize + reducedboxsize * cos(angle), reducedboxsize * sin(angle)
    ),
    path.rlineto(0, ncols * boxsize),
    path.rlineto(-nrows * boxsize, 0),
    path.rlineto(-reducedboxsize * cos(angle), -reducedboxsize * sin(angle)),
)
c.stroke(p)

for nx in range(ncols):
    x = (nx + 0.5) * boxsize
    for ny in range(nrows):
        y = (ncols - ny - 0.5) * boxsize
        c.text(
            x,
            y,
            rf"\textbf{{{ny * 10 + nx}}}",
            [text.halign.center, text.valign.middle],
        )

parwidth = 10.6
inputtemplate = r"\noindent\textcolor{{ex{}color}}{{\bfseries>{{}}>{{}}> {}}}"
resulttemplate = r"\textcolor{{ex{}color}}{{{}}}"
myattrs = [text.valign.top, text.parbox(parwidth)]
textblockdist = 0.5
ytop = ncols * boxsize

for s in (
    (
        inputtemplate.format("1", "a[(0,1,2,3,4), (1,2,3,4,5)]")
        + r"\\[0.1\baselineskip]"
        + resulttemplate.format("1", r"array([1, 12, 23, 34, 45])")
    ),
    (
        inputtemplate.format("2", "a[3:, [0,2,5]]")
        + r"\\[0.1\baselineskip]"
        + resulttemplate.format("2", "array([[30, 32, 35],")
        + r"\\"
        + resulttemplate.format("2", r"\hphantom{array([}[40, 42, 45],")
        + r"\\"
        + resulttemplate.format("2", r"\hphantom{array([}[50, 52, 55]])")
    ),
    (
        inputtemplate.format("3", "mask = np.array([1,0,1,0,0,1], dtype=bool)")
        + r"\\"
        + inputtemplate.format("3", "a[mask, 2]")
        + r"\\[0.1\baselineskip]"
        + resulttemplate.format("3", "array([2, 22, 52])")
    ),
):
    t = text.text(-parwidth - 0.5, ytop, s, myattrs)
    c.insert(t)
    ytop = ytop - t.bbox().height() - textblockdist


basename = os.path.splitext(sys.argv[0])[0]
c.writeGSfile(basename + ".png", resolution=150)
c.writePDFfile(basename)
