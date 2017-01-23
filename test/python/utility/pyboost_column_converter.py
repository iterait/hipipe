#!/usr/bin/env python3

import numpy as np
import pyboost_column_converter_py_cpp as pycpp


def main():
    assert(np.array_equal(pycpp.py_vector1d_empty(), []))

    # Test conversion of 1d, 2d and 3d vectors
    # only the last dimension (i.e., the innermost vector)
    # shall be a np.ndarray. All the other dimensions
    # shall be pure pyhton lists.

    # test whole range - 1d
    assert(isinstance(pycpp.py_vector1d(), np.ndarray))
    assert(pycpp.py_vector1d().dtype == np.int32)
    assert(np.array_equal(pycpp.py_vector1d(), [1, 2, 3]))

    # test whole range - 2d
    assert(isinstance(pycpp.py_vector2d(), list))
    for arr in pycpp.py_vector2d():
        assert(isinstance(arr, np.ndarray))
        assert(arr.dtype == np.int32)
    assert(np.array_equal(pycpp.py_vector2d(), [[1, 2, 3]] * 3))

    # test whole range - 3d
    assert(isinstance(pycpp.py_vector3d(), list))
    for d1_list in pycpp.py_vector3d():
        assert(isinstance(d1_list, list))
        for d2_arr in d1_list:
            assert(isinstance(d2_arr, np.ndarray))
            assert(d2_arr.dtype == np.int32)
    assert(np.array_equal(pycpp.py_vector3d(), [[[1, 2, 3]] * 3] * 3))

    # test slicing
    assert(np.array_equal(pycpp.py_vector1d()[ 0: 3], [1, 2, 3]))
    assert(np.array_equal(pycpp.py_vector1d()[ 1: 3], [2, 3]))
    assert(np.array_equal(pycpp.py_vector1d()[ 2: 3], [3]))
    assert(np.array_equal(pycpp.py_vector1d()[ 3: 3], []))
    assert(np.array_equal(pycpp.py_vector1d()[-1: 7], [3]))
    assert(np.array_equal(pycpp.py_vector1d()[-2:-1], [2]))

    # It is not necessary to test the conversion more in here, since it is
    # already covered by cxtream::python::range,
    # cxtream::python::utility::to_ndarray or python's list.

    columns = pycpp.columns()
    assert(set(columns.keys()) == {"Int", "Double"})
    assert(list(columns["Int"]) == [1, 2])
    assert(list(columns["Double"]) == [9., 10.])


if __name__ == '__main__':
    main()
