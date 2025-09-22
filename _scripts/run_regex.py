#!/usr/bin/env python3
"""Run a regex over a file"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import re

IMAGE_NOT_EXAMPLE = re.compile(
    r"""
^```{image}
\s+(?!auto_examples)
(?P<fname>\S+)$
.*?
```""",
    flags=re.DOTALL | re.MULTILINE | re.VERBOSE,
)


REPLACER = r"![](\1)"


def run_regexp(fname, regex, replacer):
    pth = Path(fname)
    in_contents = pth.read_text()
    out_contents = regex.sub(replacer, in_contents)
    pth.write_text(out_contents)


def get_parser():
    parser = ArgumentParser(
        description=__doc__,  # Usage from docstring
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument("fname", nargs="+", help="Files on which to run regexp")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    for fname in args.fname:
        run_regexp(fname, IMAGE_NOT_EXAMPLE, REPLACER)


if __name__ == "__main__":
    main()
