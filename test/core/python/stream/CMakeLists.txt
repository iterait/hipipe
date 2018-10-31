build_pyboost_test_module("test.core.python.stream" "column_t_py_cpp" "column_t.py.cpp")
add_python_test("test.core.python.stream.column_t" "column_t.py")

build_pyboost_test_module("test.core.python.stream" "converter_py_cpp" "converter.py.cpp")
add_python_test("test.core.python.stream.converter" "converter.py")
