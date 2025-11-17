"""Test notebook parsing"""

from copy import deepcopy
import re
import sys
from pathlib import Path

import jupytext

import pytest

HERE = Path(__file__).parent
THERE = HERE.parent
EG1_NB_PATH = HERE / "eg.Rmd"
EG2_NB_PATH = HERE / "eg2.Rmd"

sys.path.append(str(THERE))

import process_notebooks as pn


def nb2rmd(nb, fmt="myst", ext=".Rmd"):
    return jupytext.writes(nb, fmt)


@pytest.mark.parametrize("nb_path", (EG1_NB_PATH, EG2_NB_PATH))
def test_process_nbs(nb_path):
    url = f"foo/{nb_path.stem}.html"
    out_nb = pn.load_process_nb(nb_path, fmt="myst", url=url)
    out_txt = nb2rmd(out_nb)
    out_lines = out_txt.splitlines()
    assert out_lines.count("**Start of exercise**") == 2
    assert out_lines.count("**End of exercise**") == 2
    assert out_lines.count(f"**See the [corresponding page]({url}) for solution**") == 2
    # A bit of solution text, should not be there after processing.
    assert "You probably spotted that" not in out_txt
    assert "Here's our hypothesis of the algorithm:" not in out_txt
    # Admonitions
    assert out_lines.count("**Start of note**") == 1
    assert out_lines.count("**End of note**") == 1
    assert out_lines.count("**Start of admonition: My title**") == 1
    assert out_lines.count("**End of admonition**") == 1
    # Labels
    assert "plot-frames" not in out_txt


@pytest.mark.parametrize("nb_path", (EG1_NB_PATH, EG2_NB_PATH))
def test_admonition_finding(nb_path):
    nb_text = nb_path.read_text()
    nb_lines = nb_text.splitlines()
    ad_lines = pn.get_admonition_lines(nb_text, nb_path)
    for first, last in ad_lines:
        assert pn._ADM_HEADER.match(nb_lines[first])
        assert pn._END_DIV_RE.match(nb_lines[last])


def test_cell_processors():
    nb = jupytext.read(EG1_NB_PATH)
    # Code cell at index 6, Markdown at index 7.
    nb_cp = deepcopy(nb)

    def null_processor(cell):
        return cell

    out = pn.process_cells(nb_cp, [null_processor])
    assert out["cells"] is not nb_cp["cells"]
    assert out["cells"] == nb_cp["cells"]

    # Label processor.
    # There is a label in the example notebook.
    labeled_indices = [i for i, c in enumerate(nb["cells"]) if ")=\n" in c["source"]]
    assert len(labeled_indices) == 1
    out = pn.process_cells(nb_cp, [pn.label_processor])
    other_in_cell = nb_cp["cells"].pop(labeled_indices[0])
    other_out_cell = out["cells"].pop(labeled_indices[0])
    # With these cells removed, the other cells compare equal.
    assert out["cells"] == nb_cp["cells"]
    # Label removed.
    assert pn._LABEL.match(other_in_cell["source"])
    assert not pn._LABEL.match(other_out_cell["source"])

    # remove-cell processor.
    nb_cp = deepcopy(nb)
    # No tagged cells in original notebook.
    out = pn.process_cells(nb_cp, [pn.remove_processor])
    assert out["cells"] == nb_cp["cells"]
    # An example code and Markdown cel.
    eg_cells = [6, 7]
    for eg_i in eg_cells:
        nb_cp["cells"][eg_i]["metadata"]["tags"] = ["remove-cell"]
    out = pn.process_cells(nb_cp, [pn.remove_processor])
    assert out["cells"] != nb_cp["cells"]
    assert len(out["cells"]) == len(nb_cp["cells"]) - len(eg_cells)
    # The two cells have been dropped.
    assert out["cells"][eg_cells[0]] == nb_cp["cells"][eg_cells[-1] + 1]


def test_admonition_processing():
    src = """
## Signal processing: {mod}`scipy.signal`

::: {note}
:class: dropdown

{mod}`scipy.signal` is for typical signal processing: 1D,
regularly-sampled signals.
:::

**Resampling** {func}`scipy.signal.resample`: resample a signal to `n`
points using FFT.

::: {admonition} Another thought

Some text.


:::

More text.
"""
    out = pn.process_admonitions(src, EG1_NB_PATH)
    exp = """
## Signal processing: {mod}`scipy.signal`

**Start of note**

{mod}`scipy.signal` is for typical signal processing: 1D,
regularly-sampled signals.

**End of note**

**Resampling** {func}`scipy.signal.resample`: resample a signal to `n`
points using FFT.

**Start of admonition: Another thought**

Some text.

**End of admonition**

More text."""
    assert exp == out
