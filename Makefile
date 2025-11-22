PYTHON ?= python
PIP_INSTALL_CMD ?= $(PYTHON) -m pip install
BUILD_DIR=_build/html
JL_DIR=_build/jl

html:
	# Check for ipynb files in source (should all be text - .md or .Rmd).
	if compgen -G "*.ipynb" 2> /dev/null; then (echo "ipynb files" && exit 1); fi
	jupyter-book build -W .

jl:
	# Jupyter-lite files for book build.
	$(PIP_INSTALL_CMD) -r jl-build-requirements.txt
	jljb-write-dir $(BUILD_DIR)/interact data images --jl-tmp $(JL_DIR)

lint:
	pre-commit run --all-files --show-diff-on-failure --color always

web: html jl

github: web
	ghp-import -n _build/html -p -f

clean: rm-ipynb
	rm -rf _build
	-find . -name ".ipynb_checkpoints" -exec rm -rf {} \;
	-find . -name "joblib" -exec rm -rf {} \;

rm-ipynb:
	find . -name "*.ipynb" -exec rm {} \;

test:
	pytest .

compare-optimizers:
	( cd advanced/mathematical_optimization/helper && \
		python compare_optimizers.py )
