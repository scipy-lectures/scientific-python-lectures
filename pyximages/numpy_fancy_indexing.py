'''source code for figure
   numpy_fancy_indexing.(png/pdf) used in Sect. 1.3.1.7 Fancy indexing

   Requirement: PyX>=0.14  (pip installable)

'''

import os
import sys
from math import cos, radians, sin
from pyx import canvas, color, path, text, unit


def markbox(x, y, boxcolor, linewidthfactor=0.1, boxsize=1):
    '''mark box by a colored line

       the drawing is done on the global canvas called c
    '''
    linewidth = linewidthfactor*boxsize
    p = (path.rect(x, y, boxsize, boxsize)
         + path.rect(x+linewidth, y+linewidth,
                     boxsize-2*linewidth, boxsize-2*linewidth).reversed()
         )
    c.fill(p, [boxcolor])


ncols = 6
nrows = ncols
boxsize = 1
angle = radians(40)
reducedboxsize = 0.65*boxsize

text.set(text.LatexRunner)
text.preamble(r'''\usepackage[T1]{fontenc}
                  \usepackage{bera}
                  \renewcommand*\familydefault{\ttdefault}
                  \usepackage{color}
                  \definecolor{ex1}{rgb}{0, 0, 0.7}
                  \definecolor{ex2}{rgb}{0, 0.6, 0}
                  \definecolor{ex3}{rgb}{0.7, 0, 0}''')
unit.set(xscale=1.2)

c = canvas.canvas()

ex1color = color.rgb(0, 0, 0.7)
linewidth = 0.1*boxsize
for n in (0, 2, 5):
    c.fill(path.rect(n*boxsize+linewidth, linewidth,
                     boxsize-2*linewidth, 3*boxsize-2*linewidth),
           [ex1color])
    c.fill(path.rect(n*boxsize+2*linewidth, 2*linewidth,
                     boxsize-4*linewidth, 3*boxsize-4*linewidth),
           [color.grey(1)])

ex2color = color.rgb(0, 0.6, 0)
linewidth = 0.1
for n in range(5):
    markbox((n+1)*boxsize, (ncols-n-1)*boxsize, ex2color)

ex3color = color.rgb(0.7, 0, 0)
linewidth = 0.1
for n in (0, 2, 5):
    markbox(2*boxsize, (ncols-n-1)*boxsize, ex3color)

for nx in range(ncols+1):
    p = path.path(path.moveto(nx*boxsize, 0),
                  path.lineto(nx*boxsize, ncols*boxsize),
                  path.rlineto(reducedboxsize*cos(angle), reducedboxsize*sin(angle)))
    c.stroke(p)
for ny in range(nrows+1):
    p = path.path(path.moveto(0, ny*boxsize),
                  path.lineto(nrows*boxsize, ny*boxsize),
                  path.rlineto(reducedboxsize*cos(angle), reducedboxsize*sin(angle)))
    c.stroke(p)
p = path.path(path.moveto(ncols*boxsize+reducedboxsize*cos(angle),
                          reducedboxsize*sin(angle)),
              path.rlineto(0, ncols*boxsize),
              path.rlineto(-nrows*boxsize, 0),
              path.rlineto(-reducedboxsize*cos(angle), -reducedboxsize*sin(angle)))
c.stroke(p)
for nx in range(ncols):
    x = (nx+0.5)*boxsize
    for ny in range(nrows):
        y = (ncols-ny-0.5)*boxsize
        c.text(x, y, r'\textbf{{{}}}'.format(ny*10+nx), [text.halign.center, text.valign.middle])

parwidth = 10.6
s = r'''\noindent\textcolor{ex2}{\bfseries>{}>{}> a[(0,1,2,3,4), (1,2,3,4,5)]}\\[0.1\baselineskip]
        \textcolor{ex2}{array([1, 12, 23, 34, 45])}
     '''
c.text(-parwidth-0.5, ncols*boxsize, s, [text.valign.top, text.parbox(parwidth)])

s = r'''\noindent\textcolor{ex1}{{\bfseries>{}>{}> a[3:, [0,2,5]]}\\[0.1\baselineskip]
        \textcolor{ex1}{array([[30, 32, 35],}\\
        \textcolor{ex1}{\hphantom{array([}[40, 42, 45],}\\
        \textcolor{ex1}{\hphantom{array([}[50, 52, 55]])}}
     '''
c.text(-parwidth-0.5, ncols*boxsize-1.5, s, [text.valign.top, text.parbox(parwidth)])

s = r'''\noindent\textcolor{ex3}{\bfseries>{}>{}> mask = np.array([1,0,1,0,0,1], dtype=bool)}\\
        \textcolor{ex3}{\bfseries>{}>{}> a[mask, 2]}\\[0.1\baselineskip]
        \textcolor{ex3}{array([2, 22, 52])}
     '''
c.text(-parwidth-0.5, ncols*boxsize-4, s, [text.valign.top, text.parbox(parwidth)])

basename = os.path.splitext(sys.argv[0])[0]
c.writeGSfile(basename+'.png', resolution=150)
c.writePDFfile(basename)
