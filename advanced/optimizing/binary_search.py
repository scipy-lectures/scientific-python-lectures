#!/usr/bin/env python
# encoding: utf-8

import nose.tools as nt


def vsearch(seq, val):
    lo, hi = 0, len(seq)-1
    while lo <= hi:
        mid = (lo+hi)//2
        midval = seq[mid]
        if midval < val:
            lo = mid+1
        elif midval > val:
            hi = mid-1
        else:
            return mid
    return None


def test():
    cases = ((([1, 2, 3], 1), 0),
             (([-3, -2, -1], -1), 2),
             (([-3, -3, -3], -3), 1),
             (([1], 1), 0),
             (([1], 2), None),
             (([], 2), None),
             )
    for case, expected in cases:
        answer = vsearch(*case)
        nt.assert_equal(answer, expected,
                        "Case: %s, expected: %s, answer: %s" %
                        (repr(case), repr(expected), repr(answer)))
