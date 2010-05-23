Contents
--------

This repository gathers some course materials that may be used in the
intro tutorials at Euroscipy 2010.

* course_in_french : general introduction to scientific Python, aimed
  mostly at beginners. Covers Python, Numpy, Scipy, a bit of Matplotlib.
  **Written in French** (alas), as it was given first at Dakar during the
  Python African Tour (http://www.dakarlug.org/pat/).

* translation_of_course_in_french : the ongoing translation of the course
  in French.

These documents are written with the rest markup language (.rst
extension) and built using Sphinx.

Building instructions
---------------------

In each directory (course_in_french, etc.), type

make html

to generate the html output in build/ (from the files in source/).

make pdf

should generate a pdf output as well, but there may be bugs.


