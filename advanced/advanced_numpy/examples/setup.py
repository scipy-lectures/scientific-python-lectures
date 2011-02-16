import os
import sys
from os.path import join
from distutils.sysconfig import get_python_inc
import numpy
from numpy.distutils.misc_util import get_numpy_include_dirs

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    for filename in ['mandel.so']:
        # make sure we don't have stale files lying around
        if os.path.isfile(filename):
            os.unlink(filename)

    config = Configuration('', parent_package, top_path)
    config.add_extension('mandel', sources=['mandel.c'])
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
