#!/usr/bin/env python3
""" Process sphinx-gallery examples in notebook
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import ast
import re
from pathlib import Path

import jupytext


HEADER = '''\
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

'''



def get_ref_targets(root_path, nb_ext='.Rmd', excludes=()):
    refs = []
    for nb_path in root_path.glob('**/*' + nb_ext):
        if nb_path in excludes:
            continue
        refs += re.findall(r"^\s*\(\s*([a-zA-Z0-9-_]+)\s*\)=\s*$",
                           nb_path.read_text(),
                           flags=re.MULTILINE)
    return refs


def get_eg_refs(nb_path):
    # Analyze notebook for references to examples
    # Try to detect possible titles for each reference.
    return []


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
        for L in cell['source'].splitlines():
            sL = L.strip()
            if sL.startswith('plt.show'):
                continue
            if sL.startswith('import '):
                if sL in import_lines:
                    continue
                import_lines.append(sL)
            out_lines.append(L)
        if out_lines:
            cell['source'] = '\n'.join(out_lines)
            out_cells.append(cell)
    nb.cells = out_cells
    # Get title from filename if not already found.
    if title is None and (m := re.match(r'plot_(.+)\.py', eg_path.name)):
        title = m.groups()[0]
    return nb, title


def process_nb_examples(root_path,
                        nb_path,
                        examples_path,
                        use_title=True):
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
        if title is None:
            ref = eg_path.with_suffix('').name
        else:
            ref = (re.sub(r',;:', '', title).lower()
                   .replace('  ', ' ')
                   .replace(' ', '-'))
        assert ref not in ref_defs
        title = title if use_title else eg_path.stem
        examples[ref] = (title, nb)
    # Analyze notebook for references to examples
    eg_refs = get_eg_refs(nb_path)
    # Try to detect possible titles for each reference.
    # Run through examples in notebook order
    nb_out = [HEADER, '# Examples for ' + str(nb_path)]
    for eg_ref in eg_refs:
        nb_out += ['', ''] + output_example(ref, examples, header_level=2)
    remaining = [ref for ref in examples if ref not in eg_refs]
    if remaining:
        nb_out += ['', '', '## Other examples', '']
        for ref in remaining:
            nb_out += output_example(ref, examples, header_level=3)
    return '\n'.join(nb_out)


def output_example(ref, examples, header_level=2):
    title, nb = examples[ref]
    title = ref.replace('-', ' ').title() if title is None else title
    return [f'({ref})=', '',
            '#' * header_level + ' ' + title, '',
            jupytext.writes(nb, 'Rmd')]


def get_parser():
    parser = ArgumentParser(description=__doc__,  # Usage from docstring
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('nb_file', help='notebook file')
    parser.add_argument('--eg-dir', help='path to examples')
    parser.add_argument('--root-dir', help='root path to book', default='.')
    parser.add_argument('--eg-nb', help='Output notebook filename')
    parser.add_argument('--fname-titles', action='store_true',
                        help='If set, use filesnames as titles for examples')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    nb_pth = Path(args.nb_file)
    if not nb_pth.is_file():
        raise RuntimeError(f'Notebook {nb_pth} is not a file')
    if args.eg_dir is not None:
        eg_pth = Path(args.eg_dir)
    else:
        eg_pth = nb_pth.parent / 'examples'
        if not eg_pth.is_dir():
            eg_pth = nb_pth.parent.parent / 'examples'
        if not eg_pth.is_dir():
            raise RuntimeError("Cannot find examples directory")
    if args.eg_nb is not None:
        eg_nb = Path(args.eg_nb)
    else:
        eg_nb = (nb_pth.parent / (nb_pth.stem + '_examples' + nb_pth.suffix))
    out_txt = process_nb_examples(Path(args.root_dir),
                                  nb_pth,
                                  eg_pth,
                                  not args.fname_titles)
    eg_nb.write_text(out_txt)


if __name__ == '__main__':
    main()
