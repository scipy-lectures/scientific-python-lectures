#!/usr/bin/env python3
""" Process sphinx-gallery examples in notebook
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import ast
from copy import deepcopy
import re
from pathlib import Path

import jupytext
import nbformat


HEADER = jupytext.reads('''\
---
jupyter:
  jupytext:
    formats: ipynb,Rmd
    text_representation:
      extension: .Rmd
      format_name: rmarkdown
      format_version: '1.2'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

''', fmt='Rmd')

# New Markdown cell function
NMC = nbformat.versions[HEADER['nbformat']].new_markdown_cell


def get_ref_targets(root_path, nb_ext='.Rmd', excludes=()):
    refs = []
    for nb_path in root_path.glob('**/*' + nb_ext):
        if nb_path in excludes:
            continue
        refs += re.findall(r"^\s*\(\s*([a-zA-Z0-9-_]+)\s*\)=\s*$",

                           nb_path.read_text(),
                           flags=re.MULTILINE)
    return refs


FIG_EG_RE = re.compile(r'''
^(\s*:::+|```)\s*\{(?:figure|image)\}\s*
auto_examples/.*?images/sphx_glr_(?P<stem>\w+?)_\d{3}\.png
.*?
\s*\1''', flags=re.MULTILINE | re.VERBOSE | re.DOTALL)


def get_eg_stems(nb_path):
    """ Analyze notebook for references to example output
    """
    refs = []
    nb = jupytext.read(nb_path)
    for cell in nb.cells:
        if cell['cell_type'] != 'markdown':
            continue
        for ref in [m.groupdict()['stem']
                    for m in FIG_EG_RE.finditer(cell['source'])]:
            if ref not in refs:
                refs.append(ref)
    return refs


def proc_str(s):
    s = s.strip()
    lines = s.splitlines()
    if len(lines) > 2 and re.match(r'^[=-]{2,}\s*$', lines[1]):
        title = lines[0].strip()
        lines = lines[2:]
    if len(lines) and lines[0].strip() == '':
        lines = lines[1:]
    return '\n'.join(lines), title


def process_example(eg_path, import_lines=None):
    import_lines = [] if import_lines is None else import_lines
    txt = eg_path.read_text()
    nb = jupytext.reads(txt, 'py:nomarker')
    title = None
    # Convert standalone multiline strings to Markdown cells.
    out_cells = []
    for cell in nb.cells:
        if cell['cell_type'] != 'code':
            out_cells.append(cell)
            continue
        body = ast.parse(cell.source).body
        # Multiline string.
        if (len(body) == 1 and
            isinstance(body[0], ast.Expr) and
            isinstance(body[0].value, ast.Constant) and
            isinstance(body[0].value.value, str)):
            src, cell_title = proc_str(body[0].value.value)
            cell['cell_type'] = 'markdown'
            cell['source'] = src
            title = cell_title if title is None else title
            out_cells.append(cell)
            continue
        out_lines = []
        show_cell = False
        for L in cell['source'].splitlines():
            sL = L.strip()
            if sL.startswith('plt.show'):
                show_cell = True
                continue
            if sL.startswith('import '):
                if sL in import_lines:
                    continue
                import_lines.append(sL)
            out_lines.append(L)
        if out_lines:
            cell['source'] = '\n'.join(out_lines)
            if show_cell:
                cell['metadata'] = cell.get('metadata', {})
                cell['metadata']['tags'] = list(set(
                    cell['metadata'].get('tags', [])
                ).union(['hide-input']))
            out_cells.append(cell)
    nb.cells = out_cells
    # Get title from filename if not already found.
    if title is None and (m := re.match(r'plot_(.+)\.py', eg_path.name)):
        title = m.groups()[0]
    return nb, title


def process_nb_examples(root_path, nb_path, examples_path):
    # Get all references (something)=
    ref_defs = get_ref_targets(root_path)
    # Get all examples.
    examples = {}
    nb_imp_lines = []
    eg_paths = list(examples_path.glob('plot_*.py'))
    if not eg_paths:
        raise RuntimeError(f'No examples at {examples_path}')
    for eg_path in eg_paths:
        nb, title = process_example(eg_path, nb_imp_lines)
        eg_stem = eg_path.stem
        ref = (eg_stem if title is None else
               re.sub(r'[^a-zA-Z0-9]', '-', title).lower().strip('-'))
        assert ref not in ref_defs
        examples[eg_stem] = nb, title, ref
    # Analyze notebook for references to examples
    eg_stems = get_eg_stems(nb_path)
    # Try to detect possible titles for each reference.
    # Run through examples in notebook order
    nb_out = deepcopy(HEADER)
    cells = nb_out.cells
    cells.append(NMC(f'# Examples for {nb_path}'))
    for eg_stem in eg_stems:
        cells +=  output_example(eg_stem, examples, header_level=2)
    remaining = [s for s in examples if s not in eg_stems]
    if remaining:
        cells.append(NMC('## Other examples'))
        for eg_stem in remaining:
            cells += output_example(eg_stem, examples, header_level=3)
    return nb_out


def output_example(eg_stem, examples, header_level=2):
    nb, title, ref = examples[eg_stem]
    title = ref.replace('-', ' ').title() if title is None else title
    return [NMC(f'({ref})=\n\n{'#' * header_level} {title}\n\n'
                f'<!--- {eg_stem} -->')] + nb.cells


def get_parser():
    parser = ArgumentParser(description=__doc__,  # Usage from docstring
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('nb_file', help='notebook file')
    parser.add_argument('--eg-dir', help='path to examples')
    parser.add_argument('--root-dir', help='root path to book', default='.')
    parser.add_argument('--eg-nb', help='Output notebook filename')
    return parser


def main():
    args = get_parser().parse_args()
    # Process inputs and set defaults.
    nb_pth = Path(args.nb_file)
    if not nb_pth.is_file():
        raise RuntimeError(f'Notebook {nb_pth} is not a file')
    if args.eg_dir is not None:
        eg_pth = Path(args.eg_dir)
    elif (eg_pth := nb_pth.parent / 'examples').is_dir():
        pass
    elif not (eg_pth := nb_pth.parent.parent / 'examples').is_dir():
        raise RuntimeError("Cannot find examples directory")
    eg_nb = Path(args.eg_nb) if args.eg_nb is not None else (
        nb_pth.parent / (nb_pth.stem + '_examples' + nb_pth.suffix))
    # Generate, write examples notebook.
    out_nb = process_nb_examples(Path(args.root_dir), nb_pth, eg_pth)
    jupytext.write(out_nb, eg_nb, fmt='rmarkdown')


if __name__ == '__main__':
    main()
