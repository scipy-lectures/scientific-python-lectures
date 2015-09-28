# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         = a4
PYTHON        = python

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d build/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .


.PHONY: help clean html web pickle htmlhelp latex changes linkcheck zip

all: html-noplot

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  pickle    to make pickle files (usable by e.g. sphinx-web)"
	@echo "  htmlhelp  to make HTML files and a HTML help project"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  pdf       to make PDF from LaTeX, you can set PAPER=a4 or PAPER=letter"
	@echo "  changes   to make an overview over all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"
	@echo "  install   to upload to github the web pages"
	@echo "  zip       to create the zip file with examples and doc"

clean:
	-rm -rf build/*
	-rm -rf intro/matplotlib/auto_examples/ advanced/mathematical_optimization/auto_examples/ advanced/advanced_numpy/auto_examples/ packages/statistics/auto_examples/ packages/scikit-image/auto_examples/

test:
	nosetests -v --with-doctest --doctest-tests --doctest-extension=rst testing.py $(shell find intro advanced packages -name \*.rst -print)

test-stop-when-failing:
	nosetests -vx --with-doctest --doctest-tests --doctest-extension=rst testing.py $(shell find intro advanced packages -name \*.rst -print)

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) build/html
	@echo
	@echo "Build finished. The HTML pages are in build/html."

html:
	mkdir -p build/html build/doctrees
	# This line makes the build a bit more lengthy, and the
	# the embedding of images more robust
	rm -rf build/html/_images
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) build/html
	@echo
	@echo "Build finished. The HTML pages are in build/html."

cleandoctrees:
	rm -rf build/doctrees

pickle:
	mkdir -p build/pickle build/doctrees
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) build/pickle
	@echo
	@echo "Build finished; now you can process the pickle files or run"
	@echo "  sphinx-web build/pickle"
	@echo "to start the sphinx-web server."

web: pickle

htmlhelp:
	mkdir -p build/htmlhelp build/doctrees
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) build/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in build/htmlhelp."

latex: cleandoctrees
	mkdir -p build/latex build/doctrees
	$(SPHINXBUILD) -b $@ $(ALLSPHINXOPTS) build/latex
	@echo
	@echo "Build finished; the LaTeX files are in build/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

latexpdf: latex
	$(MAKE) -C build/latex all-pdf

changes:
	mkdir -p build/changes build/doctrees
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) build/changes
	@echo
	@echo "The overview file is in build/changes."

linkcheck:
	mkdir -p build/linkcheck build/doctrees
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) build/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in build/linkcheck/output.txt."

pdf: latex
	cd build/latex ; make all-pdf ; pdfnup ScipyLectures.pdf
	cp build/latex/ScipyLectures.pdf ScipyLectures-simple.pdf
	cp build/latex/ScipyLectures-nup.pdf ScipyLectures.pdf

zip: html pdf
	mkdir -p build/scipy_lecture_notes ;
	cp -r build/html build/scipy_lecture_notes ;
	cp -r data build/scipy_lecture_notes ;
	cp ScipyLectures.pdf build/scipy_lecture_notes;
	zip -r build/scipy_lecture_notes.zip build/scipy_lecture_notes  

install: cleandoctrees html pdf
	rm -rf build/scipy-lectures.github.com
	cd build/ && \
	git clone git@github.com:scipy-lectures/scipy-lectures.github.com.git && \
	cp -r html/* scipy-lectures.github.com && \
	cd scipy-lectures.github.com && \
	git add * && \
	git commit -a -m 'Make install' && \
	git push
 
epub:
	$(SPHINXBUILD) -b epub $(ALLSPHINXOPTS) build/epub
	@echo
	@echo "Build finished. The epub file is in build/epub."

contributors:
	git shortlog -sn  2>&1 | awk '{print $$NF, $$0}' | sort | cut -d ' ' -f 2- | sed "s/^  *[0-9][0-9]*	/\n- /"
