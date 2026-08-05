"""Fixes intolerance of backticks in IP3 traceback lexer.

See: https://github.com/ipython/ipython/issues/14142
"""

from pygments.token import Error, Text
from ipython_pygments_lexers import IPythonTracebackLexer


class NicerIPython3TracebackLexer(IPythonTracebackLexer):
    """Tolerate backticks in tracebacks (see module docstring)."""

    name = "IPython Traceback allowing backticks"
    aliases = ("ipython3tb", "ipythontb")  # Original aliases

    def get_tokens_unprocessed(self, text):
        for index, token, value in super().get_tokens_unprocessed(text):
            if token is Error and value == "`":
                yield index, Text, value
            else:
                yield index, token, value


def setup(app):
    app.add_lexer("ipython3tb", NicerIPython3TracebackLexer)
    app.add_lexer("ipythontb", NicerIPython3TracebackLexer)
