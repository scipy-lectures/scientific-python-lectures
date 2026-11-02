# Set VENV= (empty) to use tools already on PATH instead of a uv environment.
VENV ?= .venv
PYTHON ?= $(BIN)python
BUILD_DIR=_build/html
JL_DIR=_build/jl
REQUIREMENTS = requirements.txt build_requirements.txt test_requirements.txt jl-build-requirements.txt

ifdef VENV
export VIRTUAL_ENV := $(abspath $(VENV))
export PATH := $(VIRTUAL_ENV)/bin:$(PATH)
# Make 3.81 (macOS) ignores the exported PATH when it runs a command directly.
BIN := $(VIRTUAL_ENV)/bin/
ENV_STAMP := $(VENV)/.installed
endif

html: $(ENV_STAMP)
	# Check for ipynb files in source (should all be text - .md or .Rmd).
	if compgen -G "*.ipynb" 2> /dev/null; then (echo "ipynb files" && exit 1); fi
	$(BIN)jupyter-book build -W .

jl: $(ENV_STAMP)
	# Jupyter-lite files for book build.
	rm -rf $(JL_DIR)
	mkdir $(JL_DIR)
	cp -r data images $(JL_DIR)
	$(PYTHON) _scripts/process_notebooks.py $(JL_DIR)
	$(PYTHON) -m jupyter lite build \
		--contents $(abspath $(JL_DIR)) \
		--output-dir $(abspath $(BUILD_DIR))/interact \
		--lite-dir $(abspath $(JL_DIR))

lint: $(ENV_STAMP)
	$(BIN)pre-commit run --all-files --show-diff-on-failure --color always

web: html jl

github: web
	$(BIN)ghp-import -n _build/html -p -f

clean: rm-ipynb
	rm -rf _build
	-find . -name ".ipynb_checkpoints" -exec rm -rf {} \;
	-find . -name "joblib" -exec rm -rf {} \;

rm-ipynb:
	find . -name "*.ipynb" -exec rm {} \;

# Avoid a Jupyter deprecation warning, which pytest treats as an error.
test: export JUPYTER_PLATFORM_DIRS = 1
test: $(ENV_STAMP)
	$(BIN)pytest .

compare-optimizers: $(ENV_STAMP)
	( cd advanced/mathematical_optimization/helper && \
		$(PYTHON) compare_optimizers.py )

# Reinstall only when a requirements file is newer than the stamp.
$(ENV_STAMP): $(REQUIREMENTS)
	uv venv --allow-existing $(VENV)
	uv pip install $(addprefix -r ,$(REQUIREMENTS))
	touch $@

env: $(ENV_STAMP)

.PHONY: html jl lint web github clean rm-ipynb test compare-optimizers env
