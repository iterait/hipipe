#!/usr/bin/env python3

import numpy as np
import range_py_cpp as pycpp


def main():
    assert(list(pycpp.empty_list_range()) == [])
    assert(list(pycpp.list_range())       == [1, 1, 2, 3, 5, 8])
    assert(list(pycpp.vector_range())     == [1, 1, 2, 3, 5, 8])
    assert(list(pycpp.view_range())       == [1, 1, 2, 3, 5, 8])

    # test indexing
    # list should not be indexable
    assert(not hasattr(pycpp.list_range(), "__getitem__"))
    # vector should be indexable
    assert(hasattr(pycpp.vector_range(), "__getitem__"))

    # check many combinations and compare with python's list
    cpp_data = pycpp.vector_range()
    py_data = list(pycpp.vector_range())
    for i in range(-6, 5):
        assert(cpp_data[i] == py_data[i])

    # test slicing
    for i in range(-15, 15):
        for j in range(-15, 15):
            assert(list(cpp_data[i:j]) == py_data[i:j])
            assert(list(cpp_data[i:]) == py_data[i:])
            assert(list(cpp_data[:j]) == py_data[:j])


if __name__ == '__main__':
    main()
