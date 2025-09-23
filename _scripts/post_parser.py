#!/usr/bin/env python3
"""Post-ReST to Myst parser"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import re
import textwrap


RMD_HEADER = """\
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
"""


def process_python_block(lines, tags=()):
    if [L.strip().startswith(">>> ") for L in lines if L.strip()][0]:
        return process_doctest_block(lines)
    return [get_hdr(tags)] + lines[:] + ["```"]


_PY_BLOCK = """\
>>> 7 * 3.
21.0
>>> 2**10
1024
>>> 8 % 3
2
""".splitlines()


_EXP_PY_BLOCK = [
    "```{python}",
    "7 * 3.",
    "```",
    "",
    "```{python}",
    "2**10",
    "```",
    "",
    "```{python}",
    "8 % 3",
    "```",
]


def test_process_python_block():
    assert process_python_block(_PY_BLOCK) == _EXP_PY_BLOCK
    assert process_doctest_block(_PY_BLOCK) == _EXP_PY_BLOCK


IPY_IN = re.compile(r"In \[\d+\]: (.*)$")
IPY_OUT = re.compile(r"Out \[\d+\]: (.*)$")


def process_verbatim_block(lines):
    out_lines = []
    for line in lines:
        if line.strip() in ("@verbatim", ":verbatim:"):
            continue
        line = IPY_IN.sub(r"\1", line)
        line = IPY_OUT.sub(r"\1", line)
        out_lines.append(line)
    return ["```python", ""] + out_lines + ["```"]


_IPY_BLOCK = """\
    In [53]: a = "hello, world!"
    In [54]: a[2] = 'z'
    ---------------------------------------------------------------------------
    Traceback (most recent call last):
       File "<stdin>", line 1, in <module>
    TypeError: 'str' object does not support item assignment

    In [55]: a.replace('l', 'z', 1)
    Out[55]: 'hezlo, world!'
    In [56]: a.replace('l', 'z')
    Out[56]: 'hezzo, worzd!'
""".splitlines()


_IPY_CONT_RE = re.compile(r"\s*\.{3,}: (.*)$")


def process_ipython_block(lines):
    text = textwrap.dedent("\n".join(lines))
    if "@verbatim" in text or ":verbatim:" in text:
        return process_verbatim_block(text.splitlines())
    out_lines = ["```{python}"]
    state = "start"
    last_i = len(lines) - 1
    for i, line in enumerate(text.splitlines()):
        if state == "start" and line.strip() == "":
            continue
        if m := IPY_IN.match(line):
            if state == "output" and i != last_i:
                out_lines += ["```", "", "```{python}"]
            state = "code"
            out_lines.append(m.groups()[0])
            continue
        if state == "code" and (m := _IPY_CONT_RE.match(line)):
            out_lines.append(m.groups()[0])
            continue
        # In code, but no code input line.
        if line.strip():
            state = "output"
    return out_lines + ["```"]


def test_ipython_block():
    assert process_ipython_block(_IPY_BLOCK) == [
        "```{python}",
        'a = "hello, world!"',
        "a[2] = 'z'",
        "```",
        "",
        "```{python}",
        "a.replace('l', 'z', 1)",
        "```",
        "",
        "```{python}",
        "a.replace('l', 'z')",
        "```",
    ]


_DOCTEST_BLOCK = r"""
>>> a = "hello, world!"
>>> a[3:6] # 3rd to 6th (excluded) elements: elements 3, 4, 5
'lo,'
>>> a[2:10:2] # Syntax: a[start:stop:step]
'lo o'
>>> a[::3] # every three characters, from beginning to end
'hl r!'
""".splitlines()


def get_hdr(tags):
    if not tags:
        return "```{python}"
    joined_tags = ", ".join(f'"{t}"' for t in tags)
    return f"```{{python tags=c({joined_tags})}}"


def process_doctest_block(lines, tags=()):
    if not any(L.strip().startswith(">>> ") for L in lines):
        return process_python_block(lines, tags)
    lines = textwrap.dedent("\n".join(lines)).splitlines()
    cell_hdr = get_hdr(tags)
    out_lines = [cell_hdr]
    state = "start"
    last_i = len(lines) - 1
    for i, line in enumerate(lines):
        if state == "start" and line.strip() == "":
            continue
        if line.startswith(">>> "):
            if state == "output" and i != last_i:
                out_lines += ["```", "", cell_hdr]
            state = "code"
            out_lines.append(line[4:])
            continue
        if state == "code" and line.startswith("... "):
            out_lines.append(line[4:])
            continue
        state = "output"
    return out_lines + ["```"]


def test_doctest_block():
    assert process_doctest_block(_DOCTEST_BLOCK) == [
        "```{python}",
        'a = "hello, world!"',
        "a[3:6] # 3rd to 6th (excluded) elements: elements 3, 4, 5",
        "```",
        "",
        "```{python}",
        "a[2:10:2] # Syntax: a[start:stop:step]",
        "```",
        "",
        "```{python}",
        "a[::3] # every three characters, from beginning to end",
        "```",
    ]


def process_eval_rst_block(lines):
    return [textwrap.dedent("\n".join(lines))]


_EVAL_RST_BLOCK = """\
```{eval-rst}
.. ipython::

   In [1]: a = [1, 2, 3]

   In [2]: b = a

   In [3]: a
   Out[3]: [1, 2, 3]

   In [4]: b
   Out[4]: [1, 2, 3]

   In [5]: a is b
   Out[5]: True

   In [6]: b[1] = 'hi!'

   In [7]: a
   Out[7]: [1, 'hi!', 3]
```
""".splitlines()


def test_ipython_block_in_rst():
    assert parse_lines(_EVAL_RST_BLOCK) == [
        "```{python}",
        "a = [1, 2, 3]",
        "b = a",
        "a",
        "```",
        "",
        "```{python}",
        "b",
        "```",
        "",
        "```{python}",
        "a is b",
        "```",
        "",
        "```{python}",
        "b[1] = 'hi!'",
        "a",
        "```",
    ]


STATE_PROCESSOR = {
    "python-block": process_python_block,
    "ipython-block": process_ipython_block,
    "doctest-block": process_doctest_block,
    "eval-rst-block": process_eval_rst_block,
}


def parse_lines(lines):
    parsed_lines = []
    state = "default"
    block_lines = []
    for i, line in enumerate(lines):
        if state == "default":
            if re.match(r"```\s*\{eval-rst\}\s*$", line):
                if re.match(r"\.\.\s+ipython::", lines[i + 1]):
                    state = "ipython-block-header"
                else:
                    state = "eval-rst-block"
                # Remove all eval-rst blocks.
                continue
            LS = line.strip()
            if LS == "```":
                state = "python-block"
                continue
            if LS == "```pycon":
                state = "doctest-block"
                continue
            if LS.startswith("```"):
                state = "other-block"
                directive = line
                continue
        if state == "ipython-block-header":
            # Drop ipython line
            state = "ipython-block"
            continue
        if state.endswith("block"):
            if line.strip() != "```":
                block_lines.append(line)
                continue
            parsed_lines += (
                STATE_PROCESSOR[state](block_lines)
                if state in STATE_PROCESSOR
                else [directive] + block_lines + [line]
            )
            block_lines = []
            state = "default"
            continue
        parsed_lines.append(line)

    return parsed_lines


def strip_content(lines):
    text = "\n".join(lines)
    text = re.sub(r"^\.\.\s+currentmodule:: .*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+#\s*doctest:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"^:::\s*\{topic\}\s*\**(.*?)\**$",
        r":::{admonition} \1",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^:::\s*\{seealso\}$\n*(.*?)^:::\s*$",
        ":::{admonition} See also\n\n\\1:::\n",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return re.sub(
        r"\`\`\`\s*\{contents\}.*?^\`\`\`\s*\n",
        "",
        text,
        flags=re.MULTILINE | re.DOTALL,
    ).splitlines()


def process_percent_block(lines):
    # The first one or more lines should be considered comments.
    for i, line in enumerate(lines):
        if line.strip().startswith(">>> "):
            head_lines = [
                ">>> # " + L
                for L in lines[:i]
                if (L.strip() and "for doctest" not in L.lower())
            ]
            return process_doctest_block(head_lines + lines[i:], tags=("hide-input",))
    return ["<!---"] + lines[:] + ["-->"]


def process_percent(lines):
    out_lines = []
    block_lines = []
    state = "default"
    for line in lines:
        pct_line = line.startswith("% ")
        if state == "default":
            if not pct_line:
                out_lines.append(line)
                continue
            state = "percent-lines"
        if state == "percent-lines":
            if line.startswith("%"):
                block_lines.append(line[2:])
            else:  # End of block
                out_lines += process_percent_block(block_lines)
                assert not line.strip()
                state = "default"
                block_lines = []
    return out_lines


def process_md(fname):
    fpath = Path(fname)
    out_lines = fpath.read_text().splitlines()[:]
    for parser in [parse_lines, strip_content, process_percent]:
        out_lines = parser(out_lines)
    content = "\n".join(out_lines)
    out_path = fpath
    if fpath.suffix == ".md" and "```{python}" in content:
        out_path = fpath.with_suffix(".Rmd")
        fpath.unlink()
        content = f"{RMD_HEADER}\n{content}"
    out_path.write_text(content)


def get_parser():
    parser = ArgumentParser(
        description=__doc__,  # Usage from docstring
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument("in_md", nargs="+", help="Input Markdown files")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    for fname in args.in_md:
        process_md(fname)


if __name__ == "__main__":
    main()
