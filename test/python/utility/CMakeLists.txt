build_pyboost_test_module(pyboost_column_converter_py_cpp pyboost_column_converter.py.cpp)
add_python_test("test.python.utility.pyboost_column_converter" "pyboost_column_converter.py")

if(BUILD_PYTHON_OPENCV)
  build_pyboost_test_module(pyboost_cv_converter_py_cpp pyboost_cv_converter.py.cpp)
  add_python_test("test.python.utility.pyboost_cv_converter" "pyboost_cv_converter.py")
endif()

build_pyboost_test_module(pyboost_fs_path_converter_py_cpp pyboost_fs_path_converter.py.cpp)
add_python_test("test.python.utility.pyboost_fs_path_converter" "pyboost_fs_path_converter.py")

build_pyboost_test_module(pyboost_is_registered_py_cpp pyboost_is_registered.py.cpp)
add_python_test("test.python.utility.pyboost_is_registered" "pyboost_is_registered.py")

build_pyboost_test_module(pyboost_ndarray_converter_py_cpp pyboost_ndarray_converter.py.cpp)
add_python_test("test.python.utility.pyboost_ndarray_converter" "pyboost_ndarray_converter.py")
