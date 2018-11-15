add_subdirectory("stream")

add_subdirectory("utility")

build_pyboost_test_module("test.core.python" "range_py_cpp" "range.py.cpp")
add_python_test("test.core.python.range" "range.py")
