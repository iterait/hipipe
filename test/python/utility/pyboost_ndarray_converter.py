#!/usr/bin/env python3

import numpy as np
import pyboost_ndarray_converter_py_cpp as pycpp


def main():
    test_data = pycpp.test_data()

    # list arithmetic value types
    types = [(np.bool_,      "bool"),
             (np.int8,       "std::int8_t"),
             (np.uint8,      "std::uint8_t"),
             (np.int16,      "std::int16_t"),
             (np.uint16,     "std::uint16_t"),
             (np.int32,      "std::int32_t"),
             (np.uint32,     "std::uint32_t"),
             (np.int64,      "std::int64_t"),
             (np.uint64,     "std::uint64_t"),
             (np.float32,    "float"),
             (np.float64,    "double"),
             (np.float128,   "long double")]

    # check all the arrays with the arithmetic value types
    for np_type, cpp_type in types:
        array = test_data.typed_arrays()[cpp_type]
        # check that the converted ndarray has the desired type and dtype
        assert(isinstance(array, np.ndarray))
        assert(array.dtype == np_type)
        # check that it is able to contain min and max values from c++ (or at least that
        # they are the same as those converted by boost::python)
        desired = np.zeros(4, dtype=np_type)
        desired[0] = test_data.cpp_min_values()[cpp_type]
        desired[1] = 0
        desired[2] = 1
        # We know that python fails to represent large floating values in the float type,
        # For that reason, we skip the test of maximum value for long double.
        desired[3] = test_data.cpp_max_values()[cpp_type]
        if cpp_type == "long double":
            array = array[:3]
            desired = desired[:3]
        np.testing.assert_array_almost_equal(array, desired)

    # check empty array
    assert(isinstance(array, np.ndarray))
    assert(np.array_equal(test_data.empty_array(), []))

    # check array with shared_ptr's wrapped in a class (registered in boost::python)
    ptr_array = test_data.shared_ptr_array()
    # this should have generic np.object dtype
    assert(isinstance(ptr_array, np.ndarray))
    assert(ptr_array.dtype == np.object)
    assert(len(ptr_array) == 3)
    assert([ptr.value() for ptr in ptr_array] == [1, 2, 3])
    # check that memory is properly deallocated when references disappear
    assert([ptr.use_count() for ptr in ptr_array] == [2, 2, 2])
    ptr_array2 = test_data.shared_ptr_array()
    assert([ptr.use_count() for ptr in ptr_array] == [3, 3, 3])
    del ptr_array2
    assert([ptr.use_count() for ptr in ptr_array] == [2, 2, 2])


if __name__ == '__main__':
    main()
