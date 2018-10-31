#!/usr/bin/env python3

import numpy as np

import batch_t_py_cpp as pycpp


def main():
    assert(dict(pycpp.empty_batch()) == {})
    batch = dict(pycpp.non_empty_batch())
    assert set(batch.keys()) == {'Int', 'Double', 'IntVec'}
    assert(list(batch['Int']) == [])
    assert(list(batch['Double']) == [0.0, 1.0])
    np.testing.assert_array_equal(np.array(batch['IntVec']), np.array([[1, 2], [3, 4]]))


if __name__ == '__main__':
    main()
