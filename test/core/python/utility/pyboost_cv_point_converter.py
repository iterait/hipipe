#!/usr/bin/env python3

import numpy as np
import pyboost_cv_point_converter_py_cpp as pycpp


POINT = (4.2, 42)


def main():
    # check c++ -> python
    np.testing.assert_array_almost_equal(pycpp.point(), POINT)

    # check python -> c++
    assert(pycpp.check_point(POINT))


if __name__ == '__main__':
    main()
