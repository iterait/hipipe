add_subdirectory("stream")

add_subdirectory("utility")

add_boost_test("test.core.base64" "base64.cpp" "")

add_boost_test("test.core.csv" "csv.cpp" "")

add_boost_test("test.core.dataframe" "dataframe.cpp" "")

add_boost_test("test.core.groups" "groups.cpp" "")

add_boost_test("test.core.index_mapper" "index_mapper.cpp" "")

add_boost_test("test.core.thread" "thread.cpp" "")
