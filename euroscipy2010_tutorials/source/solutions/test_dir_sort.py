"""
Test the dir_sort logic.
"""
import dir_sort

def test_filter_and_sort():
    # Test that non '.py' files are not filtered.
    file_list = ['a', 'aaa', 'aa', '', 'z', 'zzzzz']
    file_list2 = dir_sort.filter_and_sort(file_list)
    assert len(file_list2) == 0

    # Test that the otuput file list is ordered by length.
    file_list = [ n + '.py' for n in file_list]
    file_list2 = dir_sort.filter_and_sort(file_list)
    name1 = file_list2.pop(0)
    for name in file_list2:
        assert len(name1) <= len(name)


if __name__ == '__main__':
    test_filter_and_sort()

