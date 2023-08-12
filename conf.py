from datetime import date
from subprocess import PIPE, Popen
import os

import sphinx_gallery
from pygments import formatters
from sphinx import highlighting

# General configuration
# ---------------------

exclude_patterns = ["README.rst"]

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.imgmath",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
]

# See https://sphinx-copybutton.readthedocs.io/en/latest/use.html#automatic-exclusion-of-prompts-from-the-copies
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_copy_empty_lines = False

doctest_test_doctest_blocks = "true"

sphinx_gallery_conf = {
    "examples_dirs": [
        "intro/scipy/summary-exercises/examples",
        "intro/matplotlib/examples",
        "intro/numpy/examples",
        "intro/scipy/examples",
        # the following entry contains an extra level because
        # execution of the other python files causes errors
        "advanced/advanced_numpy/examples/plots",
        "advanced/image_processing/examples",
        "advanced/mathematical_optimization/examples",
        "packages/scikit-image/examples",
        "packages/scikit-learn/examples",
        "packages/statistics/examples",
        "guide/examples",
    ],
    "gallery_dirs": [
        "intro/scipy/summary-exercises/auto_examples",
        "intro/matplotlib/auto_examples",
        "intro/numpy/auto_examples",
        "intro/scipy/auto_examples",
        "advanced/advanced_numpy/auto_examples",
        "advanced/image_processing/auto_examples",
        "advanced/mathematical_optimization/auto_examples",
        "packages/scikit-image/auto_examples",
        "packages/scikit-learn/auto_examples",
        "packages/statistics/auto_examples",
        "guide/auto_examples",
    ],
    "doc_module": "scientific-python-lectures",
    # The following is necessary to get the links in the code of the
    # examples
    "backreferences_dir": "tmp",
    "plot_gallery": "1",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# General information about the project.
project = "Scientific Python Lectures"
copyright = f"{date.today().year}"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
release = "2023.1rc0"
version = release

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
language = "en"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Monkey-patch sphinx to set the lineseparator option of pygment, to
# have indented line wrapping


class MyHtmlFormatter(formatters.HtmlFormatter):
    def __init__(self, **options):
        options["lineseparator"] = '\n<div class="newline"></div>'
        formatters.HtmlFormatter.__init__(self, **options)


highlighting.PygmentsBridge.html_formatter = MyHtmlFormatter

# Our substitutions
rst_epilog = """

.. |clear-floats| raw:: html

    <div style="clear: both"></div>

.. always clear floats at the bottom to avoid having stick out in the footer

|clear-floats|

"""

# Options for HTML output
# -----------------------

# The theme to use for HTML and HTML Help pages.  Major themes that come with
# Sphinx are currently 'default' and 'sphinxdoc'.
html_theme = "scientific_python_lectures"

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = ["themes"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # 'nosidebar': 'true',
    "footerbgcolor": "#000000",
    "relbarbgcolor": "#000000",
}

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "Scientific Python Lectures"

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = ""

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "images/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["themes/scientific_python_lectures/static"]

# If false, no index is generated.
html_use_index = False

# Output file base name for HTML help builder.
htmlhelp_basename = "ScientificPythonLectures"

# Options for epub output
# ------------------------

epub_theme = "epub"
epub_theme_options = {"relbar1": False, "footer": False}
epub_show_urls = "no"
epub_tocdup = False

# Options for LaTeX output
# ------------------------

# Latex references with page numbers (only Sphinx 1.0)
latex_show_pagerefs = False

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
latex_documents = [
    (
        "index",
        "ScientificPythonLectures.tex",
        r"Scientific Python Lectures",
        r"""Scientific Python Lectures team. Editors: Gaël Varoquaux, Emmanuelle Gouillart, Olav Vahtras, Pierre de Buyl, K. Jarrod Millman, Stéfan van der Walt""",
        "manual",
    ),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
latex_logo = "images/cover.pdf"

# Latex settings
latex_toplevel_sectioning = "part"
latex_domain_indices = False

# Additional stuff for the LaTeX preamble.
preamble = r"""
\definecolor{VerbatimColor}{rgb}{0.961, .98, 1.}
\definecolor{VerbatimBorderColor}{rgb}{0.6,0.6,0.6}
\usepackage{graphics}
\usepackage[final]{pdfpages}

\setcounter{tocdepth}{1}
\usepackage{amssymb}
\usepackage{pifont}
\usepackage{multicol}
\DeclareUnicodeCharacter{2460}{\ding{182}}
\DeclareUnicodeCharacter{2461}{\ding{183}}
\DeclareUnicodeCharacter{2462}{\ding{184}}
\DeclareUnicodeCharacter{2794}{\ding{229}}

\renewenvironment{wrapfigure}[2]{\begin{figure}[H]}{\end{figure}}

\def\shadowbox#1{\rule{\linewidth}{1pt}\nopagebreak

\nopagebreak\hspace*{.02\linewidth}#1\nopagebreak

\nopagebreak\rule{\linewidth}{1pt}
}
"""

latex_elements = {
    "papersize": "a4paper",
    "preamble": preamble,
    "fontpkg": "\\usepackage{lmodern}",
    "fncychap": r"""%
        \usepackage[Sonny]{fncychap}%
        \ChRuleWidth{1.5pt}%
        \ChNumVar{\fontsize{76}{80}\sffamily\slshape}
        \ChTitleVar{\raggedleft\Huge\sffamily\bfseries}
    """,
    "classoptions": ",oneside,openany",
    "babel": r"\usepackage[english]{babel}",
    "releasename": "Edition",
    "sphinxsetup": "warningBgColor={RGB}{255,204,204}",
    "maketitle": r"""
        \includepdf[noautoscale]{cover.pdf}
        \makeatletter%
        \hypersetup{
            pdfinfo={
                Title={\@title},
                Author={\@author},
                License={CC-BY},
            }
        }%
        \makeatother%
        \newpage\newpage
    """
    # 'tableofcontents': '\\pagestyle{normal}\\pagenumbering{arabic} %\\tableofcontents',
}

_python_doc_base = "https://docs.python.org/3/"

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": (_python_doc_base, None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "imageio": ("https://imageio.readthedocs.io/en/stable/", None),
}


extlinks = {
    "simple": (_python_doc_base + "/reference/simple_stmts.html#%s", "%s"),
    "compound": (_python_doc_base + "/reference/compound_stmts.html#%s", "%s"),
}

# -- Options for imgmath ------------------------------------------------

imgmath_dvipng_args = ["-gamma 1.5", "-D 180", "-bg", "Transparent"]
immath_use_preview = True


def add_per_page_js(app, pagename, templatename, context, doctree):
    if pagename.split("/")[-1] == "index":
        # For folding table of contents
        app.add_js_file("foldable_toc.js")
        app.add_css_file("foldable_toc.css")


def setup(app):
    app.add_js_file("https://code.jquery.com/jquery-3.7.0.min.js")
    app.add_js_file("scroll_highlight_toc.js")

    app.connect("html-page-context", add_per_page_js)

    # Is this still used?
    app.add_css_file("https://unpkg.com/purecss@3.0.0/build/base-min.css")


# Generate redirect on scipy-lectures.org
domain = os.getenv("DOMAIN", "lectures.scientific-python.org")
html_context = {"domain": domain}
print(f"Building for domain: {domain}")
