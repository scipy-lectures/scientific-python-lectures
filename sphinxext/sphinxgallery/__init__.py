"""Sphinx Gallery
"""
import os
__version__ = '0.0.10'

def path_static():
    """Returns path to packaged static files"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '_static'))
