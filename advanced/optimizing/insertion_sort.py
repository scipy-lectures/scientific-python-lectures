#!/usr/bin/env python
# encoding: utf-8

import nose.tools as nt


def vsort(seq):
    n = len(seq)
    for i in xrange(n):
        m = i
        for j in xrange(i+1, n):
            if seq[j] < seq[m]:
                m = j
        if i != m:
            seq[m], seq[i] = seq[i], seq[m]


def test():
    cases = (([1, 2, 3], [1, 2, 3]),
             ([-2, -1, -3], [-3, -2, -1]),
             ([1], [1]),
             ([], []),
             )
    for case, expected in cases:
        vsort(case)
        nt.assert_equal(case, expected)

