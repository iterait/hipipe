#!/usr/bin/env python3

import numpy as np
import pyboost_cv_converter_py_cpp as pycpp

CHESSBOARD = np.array([[1, 0, 1],
                       [0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.uint8)

RGB_SAMPLE = np.array([[[0.1, 0.2, 0.3],
                        [0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6]],
                       [[0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9],
                        [0.7, 0.8, 0.9]]], dtype=np.float32)


def main():
    # check c++ -> python
    np.testing.assert_array_equal(pycpp.chessboard(), CHESSBOARD)
    np.testing.assert_array_almost_equal(pycpp.rgb_sample(), RGB_SAMPLE)
    # check python -> c++
    assert(pycpp.check_chessboard(CHESSBOARD))
    assert(pycpp.check_rgb_sample(RGB_SAMPLE))


if __name__ == '__main__':
    main()
