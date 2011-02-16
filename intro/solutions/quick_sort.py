"""
Implement the quick sort algorithm.
"""

def qsort(lst):
    """ Quick sort: returns a sorted copy of the list.
    """
    if len(lst) <= 1:
        return lst
    pivot, rest    = lst[0], lst[1:]

    # Could use list comprehension:
    # less_than      = [ lt for lt in rest if lt < pivot ]

    less_than = []
    for lt in rest:
        if lt < pivot:
            less_than.append(lt)

    # Could use list comprehension:
    # greater_equal  = [ ge for ge in rest if ge >= pivot ]

    greater_equal = []
    for ge in rest:
        if ge >= pivot:
            greater_equal.append(ge)
    return qsort(less_than) + [pivot] + qsort(greater_equal)

# And now check that qsort does sort:
assert qsort(range(10)) == range(10)
assert qsort(range(10)[::-1]) == range(10)
assert qsort([1, 4, 2, 5, 3]) == sorted([1, 4, 2, 5, 3])
