# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
PYTHON        = python
SPHINXOPTS    =
SPHINXBUILD   = $(PYTHON) -m sphinx

ALLSPHINXOPTS   = -d build/doctrees $(SPHINXOPTS) .

TAG ?= HEAD

SSH_HOST=
SSH_USER=
SSH_TARGET_DIR=

.PHONY: help clean html web pickle htmlhelp latex changes linkcheck zip check-rsync-env

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
	-find . -name __pycache__ -type d | xargs rm -rf
	-rm -rf intro/scipy/auto_examples/ intro/matplotlib/auto_examples/ intro/scipy/summary-exercises/auto_examples advanced/mathematical_optimization/auto_examples/ advanced/advanced_numpy/auto_examples/ advanced/image_processing/auto_examples advanced/scipy_sparse/auto_examples packages/3d_plotting/auto_examples packages/statistics/auto_examples/ packages/scikit-image/auto_examples/ packages/scikit-learn/auto_examples intro/numpy/auto_examples guide/auto_examples
	-rm -f data/test.png face.png face.raw file.mat fname.png local_logo.png mandelbrot.png output.txt output2.txt plot.png pop.npy pop2.txt random_00.png random_01.png random_02.png random_03.png random_04.png random_05.png random_06.png random_07.png random_08.png random_09.png red_elephant.png test.png tiny_elephant.png workfile
	-rm -f ScientificPythonLectures-simple.pdf ScientificPythonLectures.pdf
	-rm -f advanced/image_processing/examples/face.png

test:
	$(PYTHON) -m pytest --doctest-glob '*.rst'

test-stop-when-failing:
	$(PYTHON) -m pytest -x --doctest-glob '*.rst'

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
	sed -i '/^   solutions.rst/d' intro/scipy/index.rst
	$(SPHINXBUILD) -b $@ $(ALLSPHINXOPTS) build/latex
	git restore intro/scipy/index.rst
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
	cd build/latex ; make all-pdf ; pdfjam --outfile ScientificPythonLectures-nup.pdf --nup 2x1 --landscape ScientificPythonLectures.pdf
	cp build/latex/ScientificPythonLectures.pdf ScientificPythonLectures-simple.pdf
	cp build/latex/ScientificPythonLectures-nup.pdf ScientificPythonLectures.pdf

zip: clean html pdf
	mkdir -p build/scipy_lecture_notes ;
	cp ScientificPythonLectures.pdf ScientificPythonLectures-simple.pdf build/html/_downloads/
	cp -r data build/html/
	cd build/html ; zip -r ../scientific-python-lectures-html-$(TAG).zip .
	cp ScientificPythonLectures.pdf build/ ;
	git archive -o build/scientific-python-lectures-source-$(TAG).zip --prefix scientific-python-lectures-$(TAG)/ $(TAG)

install: cleandoctrees html pdf
	rm -rf build/scipy-lectures.github.com
	cp ScientificPythonLectures.pdf ScientificPythonLectures-simple.pdf build/html/_downloads/
	cd build/ && \
	git clone  --no-checkout --depth 1 git@github.com:scipy-lectures/scipy-lectures.github.com.git && \
	cp -r html/* scipy-lectures.github.com && \
	cd scipy-lectures.github.com && \
	echo -n 'scipy-lectures.org' > CNAME && \
	touch .nojekyll && \
	git add * .nojekyll && \
	git commit -a -m 'Make install' && \
	git push

rsync_upload: check-rsync-env cleandoctrees html pdf
	cp ScientificPythonLectures-simple.pdf ScientificPythonLectures.pdf build/html/_downloads/
	rsync -P -auvz --delete build/html/ $(SSH_USER)@$(SSH_HOST):$(SSH_TARGET_DIR)/

check-rsync-env:
ifndef SSH_TARGET_DIR
	$(error SSH_TARGET_DIR is undefined)
endif
ifndef SSH_HOST
	$(error SSH_HOST is undefined)
endif

epub:
	$(SPHINXBUILD) -b epub $(ALLSPHINXOPTS) build/epub
	@echo
	@echo "Build finished. The epub file is in build/epub."

contributors:
	git shortlog -sn  2>&1 | awk '{print $$NF, $$0}' | sort | cut -d ' ' -f 2- | sed "s/^  *[0-9][0-9]*	/\n- /"
