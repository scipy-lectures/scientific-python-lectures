#!/usr/bin/env python
# encoding: utf-8

import nose.tools as nt


def vmax(seq):
    ans = None
    for i in seq:
        if i > ans:
            ans = i
    return ans


def test():
    cases = (([1, 2, 3], 3),
             ([-2, -1, -3], -1),
             ([1], 1),
             ([], None),
             )
    for case, expected in cases:
        answer = vmax(case)
        nt.assert_equal(answer, expected,
                        "Case: %s, expected: %s, answer: %s" %
                        (repr(case), repr(expected), repr(answer)))
