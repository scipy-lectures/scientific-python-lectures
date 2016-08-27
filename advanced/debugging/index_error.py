"""Small snippet to raise an IndexError."""
from __future__ import print_function

def index_error():
    lst = list('foobar')
    print(lst[len(lst)])

if __name__ == '__main__':
    index_error()

