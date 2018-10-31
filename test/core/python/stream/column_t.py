#!/usr/bin/env python3

import numpy as np

import column_t_py_cpp as pycpp


def main():
    assert(list(pycpp.empty_column()) == [])
    assert(list(pycpp.one_dim_column()) == [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(np.array(pycpp.two_dim_column()), np.array([[1, 2], [3, 4], [5, 6]]))


if __name__ == '__main__':
    main()
