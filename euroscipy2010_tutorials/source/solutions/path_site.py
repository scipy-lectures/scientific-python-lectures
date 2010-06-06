"""Script to search the PYTHONPATH for the module site.py"""

import os
import sys
import glob

def find_module(module):
    result = []
    # Loop over the list of paths in sys.path
    for subdir in sys.path:
        # Join the subdir path with the module we're searching for
        pth = os.path.join(subdir, module)
        # Use glob to test if the pth is exists
        res = glob.glob(pth)
        # glob returns a list, if it is not empty, the pth exists
        if len(res) > 0:
            result.append(res)
    return result


if __name__ == '__main__':
    result = find_module('site.py')
    print result
    
