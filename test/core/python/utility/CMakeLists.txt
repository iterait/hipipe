build_pyboost_test_module("test.core.python.utility" "ndim_vector_converter_py_cpp" "ndim_vector_converter.py.cpp")
add_python_test("test.core.python.utility.ndim_vector_converter" "ndim_vector_converter.py")

if(HIPIPE_BUILD_PYTHON_OPENCV)
  build_pyboost_test_module("test.core.python.utility" "pyboost_cv_converter_py_cpp" "pyboost_cv_converter.py.cpp")
  add_python_test("test.core.python.utility.pyboost_cv_converter" "pyboost_cv_converter.py")
endif()

build_pyboost_test_module("test.core.python.utility" "pyboost_fs_path_converter_py_cpp" "pyboost_fs_path_converter.py.cpp")
add_python_test("test.core.python.utility.pyboost_fs_path_converter" "pyboost_fs_path_converter.py")

build_pyboost_test_module("test.core.python.utility" "pyboost_is_registered_py_cpp" "pyboost_is_registered.py.cpp")
add_python_test("test.core.python.utility.pyboost_is_registered" "pyboost_is_registered.py")

build_pyboost_test_module("test.core.python.utility" "vector_converter_py_cpp" "vector_converter.py.cpp")
add_python_test("test.core.python.utility.vector_converter" "vector_converter.py")
