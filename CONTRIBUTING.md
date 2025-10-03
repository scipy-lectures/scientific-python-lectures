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
  Currently, we cover Pandas, Statmodels, some of Seaborn, Scikit-image,
  Scikit-learn, and Sympy.
- We would like to apply automated testing to the code examples as much as
  possible.

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

- Python C development headers (the `python3-dev` package on Debian, e.g.),
- a C compiler like gcc,
- [GNU Make](https://www.gnu.org/software/make/),
- a full LaTeX distribution such as [TeX Live](https://www.tug.org/texlive/) (`texlive-latex-base`,
  `texlive-latex-extra`, `texlive-fonts-extra`, and `latexmk`
  on Debian/Ubuntu),
- [dvipng](http://savannah.nongnu.org/projects/dvipng/),
- [latexmk](https://personal.psu.edu/~jcc8/software/latexmk/),
- [git](https://git-scm.com/).

### Updating the cover

Use Inkscape to modify the cover in `images/`, then export to PDF:

```
inkscape --export-filename=cover-2025.pdf cover-2025.svg
```

Ensure that the `images/cover.pdf` symlink points to the correct
file.

## Notes and admonitions

Use `:::` for
`<div>` blocks ([JupyterBook allows
this](https://jupyterbook.org/en/stable/content/content-blocks.html#markdown-friendly-directives-with)):
So, for example, prefer:

```
::: {note}

My note

:::
```

to the more standard Myst markup of:

````
<!-- #region -->
``` {note}

My note

```
<!-- #endregion -->
````

Note the `region` and `endregion` markup in the second form; this makes more
sure that Jupytext does not confuse the `{note}` with a code block. One of the
advantages of the `:::` markup is that you don't need these `#region`
demarcations.

For the same reason, prefer the `:::` form for other content blocks, such as
warnings and admonitions. For example, prefer:

```
::: {admonition} A custom title

My admonition

:::
```

## Exercises and solutions

We use [sphinx-exercise](https://ebp-sphinx-exercise.readthedocs.io) for the exercises and solutions.

Mark _all_ exercises and solutions with [gated
markers](https://ebp-sphinx-exercise.readthedocs.io/en/latest/syntax.html#alternative-gated-syntax),
like this:

```
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
```

The gated markers (of form `solution-start` and `solution-end` etc) allow you
to embed code cells in the exercise or solution, because this allows code cells
to be at the top level of the notebook, where Jupyter needs them to be.

The gated markers also make it possible to for the `process_notebooks.py`
script to recognize exercise and solutions blocks, to parse them correctly.

(notebook-processing)=

## A note on processing

The pages are designed both as pages for pretty HTML output, and to be used as
interactive notebooks in e.g. JupyterLite.

There is some markup that we need for the pretty HTML output that looks ugly in
a Jupyter interface such as [JupyterLite](https://jupyterlite.readthedocs.io).
To deal with this in part, we install the
[jupyterlab_myst](https://github.com/jupyter-book/jupyterlab-myst) extension by
default, so that Myst markup (mostly) appears as it should inside JupyterLab
when opened as a notebook.  Another difference we want to see between the HTML
and the notebook version is that we want to avoid putting the solutions in the
notebook version, to allow more space for thought about the exercise. Both to
modify any ugly formatting, and to remove the exercise solutions, we
post-process the pages with a script `_scripts/process_notebooks.py` to load
the pages as text notebooks, and write out `.ipynb` files with modified markup
that looks better in a Jupyter interface. Some of the authoring advice here is
to allow that process to work smoothly, because the `process_notebooks.py` file
reads the input Myst-MD format notebooks using
[Jupytext](https://jupytext.readthedocs.io) before converting to Jupyter
`.ipynb` files.

## Tests

There may well be cases where you will want to put cells in the rendered
notebook that test values, as part of the exposition.  For example, from the
`intro/scipy/index.md` notebook / page:

~~~

```{code-cell}
A_upper = np.triu(A)
A_upper
```

```{code-cell}
np.allclose(sp.linalg.solve_triangular(A_upper, b, lower=False),
            sp.linalg.solve(A_upper, b))
```
~~~

Notice that, in this case, we do want the reader to see that test, as part of
the exposition.

However, there are cases where the test would be useful, to, for example,
detect changes in the output over versions of the packages being used.  We want
to avoid the situation where the text says one thing, but the values contradict
it.  But we may not want the reader to have to read such tests as part of the
exposition.  Here, for example, is a test from the `intro/scipy/index.md`
notebook:

~~~

```{code-cell}
log_a = sp.special.gammaln(500)
log_b = sp.special.gammaln(499)
log_res = log_a - log_b
res = np.exp(log_res)
res
```

```{code-cell}
:tags: [remove-cell, test]
assert np.allclose(res, 499)
```

~~~

Note that the test confirms that Scipy is still giving the output implied in
the text.  Note too that we have given the testing code cell the tag
`remove-cell`.  This drops the cell from the HTML output, and our
[post-processing of the notebooks](notebook-processing) also drops these cells,
so someone opening the notebook in e.g. JupyterLite will not see them.
Accordingly, please make sure you are not defining anything in these test cells
that the notebook will need in later cells.

Be judicious — testing the output of `np.ones(3)` is probably not useful
— Numpy would have to break in order for that test to fail.

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
