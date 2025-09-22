---
orphan: true
---

# Contributing

The Scientific Python Lectures are a community-based effort and require
constant maintenance and improvements. New contributions such as wording
improvements or inclusion of new topics are welcome.

To propose bugfixes or straightforward improvements to the lectures, see the
contribution guide below.

For new topics, read the objectives first and [open an issue on the GitHub
project](https://github.com/scipy-lectures/scientific-python-lectures/issues) to
discuss it with the editors.

## Objectives and design choices for the lectures

Contributors should keep the following objectives and design choices of
the Scientific Python Lectures in mind.

Objectives:

- Provide a self-contained introduction to Python and its primary computational
  packages, the ”Scientific Python stack“.
- Provide tutorials for a selection of widely-used and stable computational
  libraries.
  Currently, we cover pandas, statmodels, seaborn, scikit-image,
  scikit-learn, and sympy.
- Automated testing is applied to the code examples as much as possible.

Design choices:

- Each chapter should provide a useful basis for a 1‒2 h tutorial.
- The code should be readable.
- An idiomatic style should be followed, e.g. `import numpy as np`,
  preference for array operations, PEP8 coding conventions.

## Contributing guide

The directory `guide` contains instructions on how to contribute:

[Contribution guide](guide)

## Building instructions

To generate the html output for on-screen display, Type:

```
make html
```

the generated html files can be found in `build/html`

The first build takes a long time, but information is cached and
subsequent builds will be faster.

To generate the pdf file for printing:

```
make pdf
```

The pdf builder is a bit difficult and you might have some TeX errors.
Tweaking the layout in the `*.rst` files is usually enough to work
around these problems.

### Requirements

Build requirements are listed in the
{download}`requirements file <requirements.txt>`:

```{literalinclude} requirements.txt
```

Ensure that you have a [virtual environment](https://docs.python.org/3/library/venv.html) or conda environment
set up, then install requirements with:

```
pip install -r requirements.txt
```

Note that you will also need the following system packages:

> - Python C development headers (the `python3-dev` package on Debian, e.g.),
> - a C compiler like gcc,
> - [GNU Make](https://www.gnu.org/software/make/),
> - a full LaTeX distribution such as [TeX Live](https://www.tug.org/texlive/) (`texlive-latex-base`,
>   `texlive-latex-extra`, `texlive-fonts-extra`, and `latexmk`
>   on Debian/Ubuntu),
> - [dvipng](http://savannah.nongnu.org/projects/dvipng/),
> - [latexmk](https://personal.psu.edu/~jcc8/software/latexmk/),
> - [git](https://git-scm.com/).

### Updating the cover

Use inkscape to modify the cover in `images/`, then export to PDF:

```
inkscape --export-filename=cover-2025.pdf cover-2025.svg
```

Ensure that the `images/cover.pdf` symlink points to the correct
file.

## A note on processing

The pages are designed both as pages for pretty HTML output, and to be used as
interactive notebooks in e.g. JupyterLite.

There is some markup that we need for the pretty HTML output that looks ugly in
a Jupyter interface such as [JupyterLite](https://jupyterlite.readthedocs.io).
Accordingly, we post-process the pages with a script
`_scripts/process_notebooks.py` to load the pages as text notebooks, and write
out `.ipynb` files with modified markup that looks better in a Jupyter
interface.  Some of the authoring advice here is to allow that process to work
smoothly, because the `process_notebooks.py` file reads the input Myst-MD
format notebooks using [Jupytext](https://jupytext.readthedocs.io) before
converting to Jupyter `.ipynb` files.

## Notes and admonitions

Use `:::` for
`<div>` blocks ([JupyterBook allows
this](https://jupyterbook.org/en/stable/content/content-blocks.html#markdown-friendly-directives-with)):
So, for example, prefer:

~~~
::: {note}

My note

:::
~~~

to the more standard Myst markup of:

~~~
<!-- #region -->
``` {note}

My note

```
<!-- #endregion -->
~~~

Note the `region` and `endregion` markup in the second form; this makes more
sure that Jupytext does not confuse the `{note}` with a code block.  One of the
advantages of the `:::` markup is that you don't need these `#region`
demarcations.

For the same reason, prefer the `:::` form for other content blocks, such as
warnings and admonitions.  For example, prefer:

~~~
::: {admonition} A custom title

My admonition

:::
~~~


## Exercises and solutions

We use [sphinx-exercise](https://ebp-sphinx-exercise.readthedocs.io) for the exercises and solutions.

Mark *all* exercises and solutions with [gated
markers](https://ebp-sphinx-exercise.readthedocs.io/en/latest/syntax.html#alternative-gated-syntax),
like this:

~~~
::: {exercise-start}
:label: my-exercise-label
:class: dropdown
:::

My exercise.

::: {exercise-end}
:::

::: {solution-start} my-exercise-label
:class: dropdown
:::

My solution.

::: {solution-end}
:::
~~~

The gated markers (of form `solution-start` and `solution-end` etc) allow you
to embed code cells in the exercise or solution, because this allows code cells
to be at the top level of the notebook, where Jupyter needs them to be.

The gated markers also make it possible to for the `process_notebooks.py`
script to recognize exercise and solutions blocks, to parse them correctly.

## Development

Run this once, in the repository directory:

```
pip install pre_commit
pre-commit install
```

Before each commit that you will push:

```
pre-commit run --all
```

Among other things, this runs the `codespell` check, also run by CI.
