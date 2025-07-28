""" Test notebook parsing
"""

import sys
from pathlib import Path

import jupytext

import pytest

HERE = Path(__file__).parent
THERE = HERE.parent
EG1_NB_PATH = HERE / 'eg.Rmd'
EG2_NB_PATH = HERE / 'eg2.Rmd'

sys.path.append(str(THERE))

import process_notebooks as pn


def nb2rmd(nb, fmt='myst', ext='.Rmd'):
    return jupytext.writes(nb, fmt)


@pytest.mark.parametrize('nb_path', (EG1_NB_PATH, EG2_NB_PATH))
def test_process_nbs(nb_path):
    url = url=f'foo/{nb_path.stem}.html'
    out_nb = pn.load_process_nb(nb_path, fmt='msyt', url=url)
    out_txt = nb2rmd(out_nb)
    out_lines = out_txt.splitlines()
    assert out_lines.count('**Start of exercise**') == 2
    assert out_lines.count('**End of exercise**') == 2
    assert out_lines.count(
        f'**See the [corresponding page]({url}) for solution**'
    ) == 2
    # A bit of solution text, should not be there after processing.
    assert 'You probably spotted that' not in out_txt
    assert "Here's our hypothesis of the algorithm:" not in out_txt
    # Admonitions
    assert out_lines.count('**Start of note**') == 1
    assert out_lines.count('**End of note**') == 1
    assert out_lines.count('**Start of admonition: My title**') == 1
    assert out_lines.count('**End of admonition**') == 1
    # Labels
    assert 'plot-frames' not in out_txt


@pytest.mark.parametrize('nb_path', (EG1_NB_PATH, EG2_NB_PATH))
def test_admonition_finding(nb_path):
    nb_text = nb_path.read_text()
    nb_lines = nb_text.splitlines()
    ad_lines = pn.get_admonition_lines(nb_text)
    for first, last in ad_lines:
        assert pn._ADM_HEADER.match(nb_lines[first])
        assert pn._END_DIV_RE.match(nb_lines[last])
