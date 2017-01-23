#!/usr/bin/env python3

import numpy as np
import pyboost_fs_path_converter_py_cpp as pycpp

paths = ["path1.txt",
         "dir1/path2.txt",
         "dir1/dir2/path3.txt"]

def main():
    # check c++ -> python
    assert(pycpp.paths() == paths)
    # check python -> c++
    assert(pycpp.check_paths(paths))


if __name__ == '__main__':
    main()
