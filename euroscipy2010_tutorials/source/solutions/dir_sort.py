"""
Script to list all the '.py' files in a directory, in the order of file
name length.
"""

import os
import sys


def filter_and_sort(file_list):
    """ Out of a list of file names, returns only the ones ending by
        '.py', ordered with increasing file name length.
    """
    file_list = [filename for filename in file_list 
                          if filename.endswith('.py')]

    def key(item):
        return len(item)

    file_list.sort(key=key)
    return file_list


if __name__ == '__main__':
    file_list = os.listdir(sys.argv[-1])
    sorted_file_list = filter_and_sort(file_list)
    print sorted_file_list

